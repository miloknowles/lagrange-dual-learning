import math

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from utils.constants import Constants
from utils.forward_kinematics import *
from utils.ema import ExponentialMovingAverage
from models.simple_networks import *
from models.residual_network import *


class JointAngleDataset(Dataset):
  def __init__(self, N, J, seed=0):
    super(JointAngleDataset, self).__init__()
    self.N = N    # The number of examples.
    self.J = J    # The number of links on this robot.

    # Generate N random points in R^J.
    torch.manual_seed(seed)
    self.random_theta = torch.empty(N, J).uniform_(0, 2*math.pi)
    self.random_ee = ForwardKinematicsThreeLinkTorch(self.random_theta)

  def __len__(self):
    return self.N

  def __getitem__(self, index):
    return {
      "joint_theta": self.random_theta[index],
      "q_ee_desired": self.random_ee[index]
    }


def train():
  """
  Train a network to approximate inverse kinematics solutions for a three link robotic arm.

  min { ||theta||^2 }                    ==> Minimize L2 norm of joint angles.
  s.t. ||f(theta) - ee_desired|| = 0     ==> Such that end effector is at desired position.
  """
  dimensions = 3

  device = torch.device("cuda")

  num_network_inputs = 3 # Gets a (desired_x, desired_y, desired_theta) input.
  num_network_outpus = 3 # Outpus angles for each joint.
  # model = FourLayerNetwork(num_network_inputs, num_network_outpus, hidden_units=80).to(device)
  model = SixLayerNetwork(num_network_inputs, num_network_outpus, hidden_units=40).to(device)
  # model = ResidualNetwork(num_network_inputs, num_network_outpus, hidden_units=40).to(device)

  lagrange_iters = 1000
  train_iters = 500
  batch_size = 8

  dataset_size = train_iters * batch_size

  train_dataset = JointAngleDataset(dataset_size, 3)
  val_dataset = JointAngleDataset(dataset_size, 3)
  train_loader = DataLoader(train_dataset, batch_size, True, num_workers=2)
  val_loader = DataLoader(val_dataset, batch_size, False, num_workers=2)

  lamda = 40.0 * torch.ones(2).to(device)

  optimizer = Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
  multiplier_lr = 1e-4

  print("Training settings:")
  print("Lagrange iters:\n  ", lagrange_iters)
  print("Train iters:\n  ", train_iters)
  print("Dataset size:\n  ", dataset_size)
  print("Initial lagrange multipliers:\n  ", lamda)

  def process_batch(inputs):
    outputs = {}

    for key in inputs:
      inputs[key] = inputs[key].to(device)

    pred_joint_theta = model(inputs["q_ee_desired"])
    outputs["pred_joint_theta"] = pred_joint_theta

    q_ee_desired = inputs["q_ee_desired"]
    q_ee_actual = ForwardKinematicsThreeLinkTorch(pred_joint_theta)
    outputs["q_ee_actual"] = q_ee_actual

    # Compute the loss using the current Lagrangian multipliers.
    joint_angle_l2 = 0.5 * pred_joint_theta**2
    position_err_sq = 0.5 * (q_ee_desired[:,:2] - q_ee_actual[:,:2])**2

    # TODO: add orientation constraint
    lagrange_loss = joint_angle_l2.sum() + lamda[0]*position_err_sq.sum()

    outputs["joint_angle_l2"] = joint_angle_l2.sum()
    outputs["position_err_sq"] = position_err_sq.sum()
    outputs["lagrange_loss"] = lagrange_loss

    return outputs

  for li in range(lagrange_iters):
    # Train the model using the current Lagrange relaxation.

    ema = ExponentialMovingAverage(0, smoothing_factor=2)

    for ti, inputs in enumerate(train_loader):
      for key in inputs:
        inputs[key] = inputs[key].to(device)

      outputs = process_batch(inputs)
      # print(outputs["lagrange_loss"])

      model.zero_grad()
      outputs["lagrange_loss"].backward()
      optimizer.step()

      loss_detached = outputs["lagrange_loss"].detach().cpu().numpy().item()
      if ti == 0:
        ema.initialize(loss_detached)
      else:
        ema.update(ti, loss_detached)
      # print("Raw={} Smoothed={}".format(loss_detached, ema.ema))

    # Aggregate constraint violations across all of the training examples.
    with torch.no_grad():
      total_constraint_violations = torch.zeros(len(train_loader))
      for ti, inputs in enumerate(train_loader):
        outputs = process_batch(inputs)
        total_constraint_violations[ti] = outputs["position_err_sq"]

      # Do a single update on the Lagrange multipliers.
      lamda = lamda + multiplier_lr*total_constraint_violations.sum()

    # Validate the model on a different sampling of points.
    mean_supervised_loss = torch.zeros(len(val_loader))
    mean_constraint_violation = torch.zeros(len(val_loader))
    mean_lagrange_loss = torch.zeros(len(val_loader))

    with torch.no_grad():
      for vi, inputs in enumerate(val_loader):
        outputs = process_batch(inputs)
        mean_supervised_loss[vi] = outputs["joint_angle_l2"]
        mean_constraint_violation[vi] = outputs["position_err_sq"]
        mean_lagrange_loss[vi] = outputs["lagrange_loss"]

    print("Epoch {} | MSE Loss={:4f} | Constraint Violation={:4f} | Lagrange Loss={:4f} | lambda={:4f}".format(
        li, mean_supervised_loss.mean(), mean_constraint_violation.mean(), mean_lagrange_loss.mean(), lamda[0].item()))


if __name__ == "__main__":
  train()
