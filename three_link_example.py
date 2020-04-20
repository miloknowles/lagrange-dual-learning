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
    self.random_ee, _, _ = ForwardKinematicsThreeLinkTorch(self.random_theta)

  def __len__(self):
    return self.N

  def __getitem__(self, index):
    return {
      "joint_theta": self.random_theta[index],
      "q_ee_desired": self.random_ee[index]
    }


class IkOptions(object):
  def __init__(self):
    self.lagrange_iters = 1000
    self.train_iters = 500
    self.batch_size = 8
    self.dataset_size = 16000
    self.device = torch.device("cuda")
    self.multiplier_lr = 1e-4
    self.initial_lambda = 40.0


class IkLagrangeDualTrainer(object):
  """
  Train a network to approximate inverse kinematics solutions for a three link robotic arm.

  min { ||theta||^2 }                    ==> Minimize L2 norm of joint angles.
  s.t. ||f(theta) - ee_desired|| = 0     ==> Such that end effector is at desired position.
  """
  def __init__(self, opt):
    self.opt = opt

    num_network_inputs = 3 # Gets a (desired_x, desired_y, desired_theta) input.
    num_network_outputs = 3 # Outpus angles for each joint.
    # self.model = FourLayerNetwork(num_network_inputs, num_network_outpus, hidden_units=80).to(self.opt.device)
    self.model = SixLayerNetwork(num_network_inputs, num_network_outputs, hidden_units=40).to(self.opt.device)
    # self.model = ResidualNetwork(num_network_inputs, num_network_outpus, hidden_units=40).to(self.opt.device)

    train_dataset = JointAngleDataset(self.opt.dataset_size, 3)
    val_dataset = JointAngleDataset(self.opt.dataset_size, 3)
    self.train_loader = DataLoader(train_dataset, self.opt.batch_size, True, num_workers=2)
    self.val_loader = DataLoader(val_dataset, self.opt.batch_size, False, num_workers=2)

    self.optimizer = Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
    self.lamda = self.opt.initial_lambda * torch.ones(8).to(self.opt.device)

    print("==> TRAINING SETTINGS:")
    print("Lagrange iters:\n  ", self.opt.lagrange_iters)
    print("Train iters:\n  ", self.opt.train_iters)
    print("Dataset size:\n  ", self.opt.dataset_size)
    print("Initial lagrange multipliers:\n  ", self.lamda)

  def main(self):
    """
    Main training loop.
    """
    for epoch in range(self.opt.lagrange_iters):
      self.train_with_relaxation(self.lamda)
      self.lamda = self.update_multipliers(self.lamda)
      self.validate(epoch, self.lamda)

  def process_batch(self, inputs, lamda):
    """
    Process a batch of inputs through the network, compute constraint violations and losses.
    """
    outputs = {}

    for key in inputs:
      inputs[key] = inputs[key].to(self.opt.device)

    pred_joint_theta = self.model(inputs["q_ee_desired"])
    outputs["pred_joint_theta"] = pred_joint_theta

    q_ee_desired = inputs["q_ee_desired"]
    q_ee_actual, q_joint1, q_joint0 = ForwardKinematicsThreeLinkTorch(pred_joint_theta)
    outputs["q_ee_actual"] = q_ee_actual

    # Compute the loss using the current Lagrangian multipliers.
    joint_angle_l2 = 0.5 * pred_joint_theta**2
    position_err_sq = 0.5 * (q_ee_desired[:,:2] - q_ee_actual[:,:2])**2

    # Limit joint angles to +/- PI/2.
    joint0_limit_violation = torch.abs(pred_joint_theta[:,0]) - (math.pi)
    joint1_limit_violation = torch.abs(pred_joint_theta[:,1]) - (math.pi / 2)
    joint2_limit_violation = torch.abs(pred_joint_theta[:,2]) - (math.pi / 2)

    # Constrain all of the joints to avoid a box.
    obstacle_xlimits = torch.Tensor([0.4, 0.6])
    obstacle_ylimits = torch.Tensor([0.4, 0.6])

    # This makes the violation maximized at the center of the box, and decreasing to zero at the edges.
    joint0_box_viol_x = obstacle_xlimits.mean() - torch.abs(q_joint0[0] - obstacle_xlimits.mean())
    joint0_box_viol_y = obstacle_ylimits.mean() - torch.abs(q_joint0[1] - obstacle_ylimits.mean())

    joint1_box_viol_x = obstacle_xlimits.mean() - torch.abs(q_joint1[0] - obstacle_xlimits.mean())
    joint1_box_viol_y = obstacle_ylimits.mean() - torch.abs(q_joint1[1] - obstacle_ylimits.mean())

    lagrange_loss = joint_angle_l2.sum() + \
        lamda[0]*position_err_sq.sum() + \
        lamda[1]*joint0_limit_violation.sum() + \
        lamda[2]*joint1_limit_violation.sum() + \
        lamda[3]*joint2_limit_violation.sum() + \
        lamda[4]*joint0_box_viol_x.sum() + \
        lamda[5]*joint0_box_viol_y.sum() + \
        lamda[6]*joint1_box_viol_x.sum() + \
        lamda[7]*joint1_box_viol_y.sum()

    outputs["joint_angle_l2"] = joint_angle_l2.sum()

    outputs["constraint_violations"] = torch.Tensor([
      position_err_sq.sum(),
      joint0_limit_violation.sum(),
      joint1_limit_violation.sum(),
      joint2_limit_violation.sum(),
      joint0_box_viol_x.sum(),
      joint0_box_viol_y.sum(),
      lamda[6]*joint1_box_viol_x.sum(),
      lamda[7]*joint1_box_viol_y.sum()
    ])

    outputs["lagrange_loss"] = lagrange_loss

    return outputs

  def train_with_relaxation(self, lamda):
    """
    Do one training epoch of the model on a Lagrangian relaxation parameterized by the current
    multipliers lambda.
    """
    # Train the self.model using the current Lagrange relaxation.
    ema = ExponentialMovingAverage(0, smoothing_factor=2)

    for ti, inputs in enumerate(self.train_loader):
      for key in inputs:
        inputs[key] = inputs[key].to(self.opt.device)

      outputs = self.process_batch(inputs, lamda)

      self.model.zero_grad()
      outputs["lagrange_loss"].backward()
      self.optimizer.step()

      loss_detached = outputs["lagrange_loss"].detach().cpu().numpy().item()
      if ti == 0:
        ema.initialize(loss_detached)
      else:
        ema.update(ti, loss_detached)

  def update_multipliers(self, lamda):
    """
    Do a single update step on the Lagrange multipliers based on constraint violations.
    """
    # Aggregate constraint violations across all of the training examples.
    with torch.no_grad():
      total_constraint_violations = torch.zeros(len(self.train_loader), len(lamda)).to(self.opt.device)

      for ti, inputs in enumerate(self.train_loader):
        outputs = self.process_batch(inputs, lamda)
        total_constraint_violations[ti,:] = outputs["constraint_violations"]

      lamda = lamda + self.opt.multiplier_lr*total_constraint_violations.sum(axis=0)

      # If a constraint is really easy to satisfy (and always negative), then lamda could become
      # unboundedly negative, effective turning it off. Make sure this doesn't happen.
      lamda = lamda.clamp(min=0)

    return lamda

  def validate(self, epoch, lamda):
    """
    Test the model on a validation set to see if the constraint violation and loss is improving.
    """
    mean_supervised_loss = torch.zeros(len(self.val_loader))
    mean_constraint_violation = torch.zeros(len(self.val_loader), len(lamda))
    mean_lagrange_loss = torch.zeros(len(self.val_loader))

    with torch.no_grad():
      for vi, inputs in enumerate(self.val_loader):
        outputs = self.process_batch(inputs, lamda)
        mean_supervised_loss[vi] = outputs["joint_angle_l2"]
        mean_constraint_violation[vi,:] = outputs["constraint_violations"]
        mean_lagrange_loss[vi] = outputs["lagrange_loss"]

    print("==> Epoch {}\n  Orig Loss={}\n  Constraints={}\n  Lagrange Loss={}\n  Multipliers={}".format(
        epoch, mean_supervised_loss.mean(), mean_constraint_violation.mean(axis=0),
        mean_lagrange_loss.mean(), lamda))


if __name__ == "__main__":
  opt = IkOptions()
  trainer = IkLagrangeDualTrainer(opt)
  trainer.main()
