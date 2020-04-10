import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np


def plot_points_3d(points, coord_min, coord_max, downsample=None):
  """
  Args:
    points (torch.Tensor) : Shape (num_points, 3).
    coord_min (float) : Used for setting axes min.
    coord_max (float) : Used for setting axes max.
  """
  assert(len(points.shape) == 2)
  assert(points.shape[1] == 3)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  if downsample is not None:
    sample_indices = np.random.choice(len(points), size=downsample)
    points = points[sample_indices]

  ax.scatter(points[:,0], points[:,1], points[:,2])

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  ax.set_xlim([coord_min, coord_max])
  ax.set_ylim([coord_min, coord_max])
  ax.set_zlim([coord_min, coord_max])

  plt.title("Distribution of 3D Points")
  plt.show()


class TwoLayerNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_units=40):
    super(TwoLayerNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_units, bias=True)
    self.fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.out = nn.Linear(hidden_units, output_dim, bias=True)

  def forward(self, x):
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    return self.out(out)


class ProjectedPointDataset(Dataset):
  def __init__(self, N, D, plane_normal, plane_offset, coord_min=-10, coord_max=10, seed=0):
    super(ProjectedPointDataset, self).__init__()
    self.N = N
    self.D = D

    # Generate N random points in R^d uniformly spaced in the hypercube.
    torch.manual_seed(seed)
    self.random_points = torch.empty(N, D).uniform_(coord_min, coord_max)

    # x' = x - (n*x + d) * n
    plane_normal = plane_normal.cpu()
    self.projected_points = project_to_plane(self.random_points, plane_normal, plane_offset)
    assert(torch.allclose(squared_distance_to_plane(self.projected_points, plane_normal, plane_offset), torch.zeros(N)))

    self.random_points.requires_grad = False
    self.projected_points.requires_grad = False

  def __len__(self):
    return self.N

  def __getitem__(self, index):
    return {
      "point_raw": self.random_points[index],
      "point_projected": self.projected_points[index]
    }


def project_to_plane(points, plane_normal, plane_offset):
  """
  points (torch.Tensor) : Shape (N, 3)
  plane_normal (torch.Tensor) : Shape (3)
  plane_offset (float) : Distance from the origin to the plane along its normal vector.
  """
  N, d = points.shape[0], points.shape[1]
  signed_plane_dist = torch.bmm(points.view(N, 1, d), plane_normal.unsqueeze(-1).unsqueeze(0).expand(N, -1, -1)).sum(axis=1) - plane_offset
  return points - signed_plane_dist*plane_normal


def squared_distance_to_plane(points, plane_normal, plane_offset):
  N, d = points.shape[0], points.shape[1]
  signed_plane_dist = torch.bmm(points.view(N, 1, d), plane_normal.unsqueeze(-1).unsqueeze(0).expand(N, -1, -1)).sum(axis=1) - plane_offset
  return signed_plane_dist**2


def train(dimensions=3):
  """
  Train a network to learn a projection function.

  min { ||x - x'||^2 }    ==> Minimize distance between x' and input point x.
  s.t. x^T p - d = 0      ==> Such that x' lies on a plane with normal p and offset d.
  """
  torch.autograd.set_detect_anomaly(True)

  device = torch.device("cuda")
  model = TwoLayerNetwork(dimensions, dimensions, hidden_units=40).to(device)

  lagrange_iters = 100
  train_iters = 100

  # NOTE: Need to normalize the vector!
  plane_normal = torch.Tensor([1, 2, 3]).to(device)
  plane_normal /= plane_normal.norm(2)
  plane_offset = 0.5

  dataset_dim_min = -10
  dataset_dim_max = 10

  batch_size = 128

  # Create a dataset of points and their planar projection.
  dataset_size = train_iters * batch_size
  train_dataset = ProjectedPointDataset(dataset_size, dimensions, plane_normal, plane_offset,
                                        coord_min=dataset_dim_min, coord_max=dataset_dim_max)
  val_dataset = ProjectedPointDataset(10 * batch_size, dimensions, plane_normal, plane_offset,
                                        coord_min=dataset_dim_min, coord_max=dataset_dim_max)
  train_loader = DataLoader(train_dataset, batch_size, True, num_workers=2)
  val_loader = DataLoader(val_dataset, batch_size, False, num_workers=2)

  # plot_points_3d(train_dataset.projected_points, dataset_dim_min, dataset_dim_max, downsample=1000)

  # Just one planar constraint in this case, so a single multiplier.
  lamda = torch.ones(1).to(device)

  optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
  multiplier_lr = 1e-4

  print("Training settings:")
  print("Lagrange iters:\n  ", lagrange_iters)
  print("Train iters:\n  ", train_iters)
  print("Plane parameters:\n  ", "Normal=", plane_normal, "Offset=", plane_offset)
  print("Dataset size:\n  ", dataset_size)
  print("Initial lagrange multipliers:\n  ", lamda)

  for li in range(lagrange_iters):
    # Train the model using the current Lagrange relaxation.
    for ti, inputs in enumerate(train_loader):
      for key in inputs:
        inputs[key] = inputs[key].to(device)

      p_hat = model(inputs["point_raw"])

      # Compute the loss using the current Lagrangian multipliers.
      supervised_loss = (p_hat - inputs["point_projected"])**2
      constraint_violation = squared_distance_to_plane(p_hat, plane_normal, plane_offset)
      lagrange_loss = supervised_loss.mean() + lamda*constraint_violation.mean()

      model.zero_grad()
      lagrange_loss.backward()
      optimizer.step()

    # Aggregate constraint violations across all of the training examples.
    with torch.no_grad():
      total_constraint_violations = torch.zeros(len(train_loader))
      for ti, inputs in enumerate(train_loader):
        for key in inputs:
          inputs[key] = inputs[key].to(device)
        p_hat = model(inputs["point_raw"])
        constraint_violation = squared_distance_to_plane(p_hat, plane_normal, plane_offset)
        total_constraint_violations[ti] = constraint_violation.sum()

      # Do a single update on the Lagrange multipliers.
      lamda = lamda + multiplier_lr*total_constraint_violations.sum()

    # Validate the model on a different sampling of points.
    mean_supervised_loss = torch.zeros(len(val_loader))
    mean_constraint_violation = torch.zeros(len(val_loader))
    mean_lagrange_loss = torch.zeros(len(val_loader))

    with torch.no_grad():
      for vi, inputs in enumerate(val_loader):
        for key in inputs:
          inputs[key] = inputs[key].to(device)

        p_hat = model(inputs["point_raw"])

        supervised_loss = (p_hat - inputs["point_projected"])**2
        constraint_violation = squared_distance_to_plane(p_hat, plane_normal, plane_offset)
        lagrange_loss = supervised_loss.mean() + lamda*constraint_violation.mean()

        mean_supervised_loss[vi] = supervised_loss.mean()
        mean_constraint_violation[vi] = constraint_violation.mean()
        mean_lagrange_loss[vi] = lagrange_loss.mean()

    print("Epoch {} | MSE Loss={} | Constraint Violation={} | Lagrange Loss={} | lambda={}".format(
        li, mean_supervised_loss.mean(), mean_constraint_violation.mean(), mean_lagrange_loss.mean(), lamda.item()))


if __name__ == "__main__":
  train()
