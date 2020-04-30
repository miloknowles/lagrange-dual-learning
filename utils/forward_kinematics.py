import torch
import numpy as np

from .constants import Constants


def ForwardKinematicsTwoLink(theta):
  """
  Forward kinematics math for the 2-link robot arm.

  Args:
    theta (np.array) : Shape (2,), contains the three joint angles of the robot arm.

  Returns:
    (np.array) of shape (3,) containing the resulting (x, y, theta) configuration of the EE.
  """
  x_ee = np.empty(3, dtype=object)
  x_ee[0] = Constants.R2_LENGTH1*np.cos(theta[0]) + Constants.R2_LENGTH2*np.cos(theta[0] + theta[1])
  x_ee[1] = Constants.R2_LENGTH1*np.sin(theta[0]) + Constants.R2_LENGTH2*np.sin(theta[0] + theta[1])
  x_ee[2] = theta[0] + theta[1]

  return x_ee


def ForwardKinematicsThreeLink(theta):
  """
  Forward kinematics math for the 3-link robot arm.

  Args:
    theta (np.array) : Shape (3,), contains the three joint angles of the robot arm.

  Returns:
    (np.array) of shape (3,) containing the resulting (x, y, theta) configuration of the EE.
  """
  x3_ee = np.empty(3, dtype=object)
  theta_0 = theta[0]
  theta_01 = theta_0 + theta[1]
  theta_012 = theta_01 + theta[2]

  x3_ee[0] = Constants.R3_LENGTH1*np.cos(theta_0) + Constants.R3_LENGTH2*np.cos(theta_01) + Constants.R3_LENGTH3*np.cos(theta_012)
  x3_ee[1] = Constants.R3_LENGTH1*np.sin(theta_0) + Constants.R3_LENGTH2*np.sin(theta_01) + Constants.R3_LENGTH3*np.sin(theta_012)
  x3_ee[2] = theta_012

  return x3_ee


def ForwardKinematicsThreeLinkTorch(theta):
  """
  Forward kinematics math for the 3-link robot arm.

  Args:
    theta (torch.Tensor) : Shape (b, 3), contains the three joint angles of the robot arm.

  Returns:
    (torch.Tensor) of shape (b, 3) containing the resulting (x, y, theta) configuration of the EE.
  """
  assert(len(theta.shape) == 2)
  assert(theta.shape[1] == 3)

  x3_ee = torch.zeros_like(theta)
  x2 = torch.zeros_like(theta)
  x1 = torch.zeros_like(theta)

  theta_0 = theta[:,0]
  theta_01 = theta_0 + theta[:,1]
  theta_012 = theta_01 + theta[:,2]

  # LINK 1
  x1[:,0] = Constants.R3_LENGTH1*torch.cos(theta_0)
  x1[:,1] = Constants.R3_LENGTH1*torch.sin(theta_0)
  x1[:,2] = theta[:,0]

  # LINK 2
  x2[:,0] = x1[:,0] + Constants.R3_LENGTH2*torch.cos(theta_01)
  x2[:,1] = x1[:,1] + Constants.R3_LENGTH2*torch.sin(theta_01)
  x2[:,2] = x1[:,2] + theta[:,1]

  # LINK 3 (end effector)
  x3_ee[:,0] = x2[:,0] + Constants.R3_LENGTH3*torch.cos(theta_012)
  x3_ee[:,1] = x2[:,1] + Constants.R3_LENGTH3*torch.sin(theta_012)
  x3_ee[:,2] = x2[:,2] + theta[:,2]

  return x3_ee, x2, x1


def ForwardKinematicsEightLinkTorch(theta):
  """
  Forward kinematics math for the 8-link robot arm.

  Args:
    theta (torch.Tensor) : Shape (b, 8), contains the three joint angles of the robot arm.

  Returns:
    (torch.Tensor) of shape (b, 3) containing the resulting (x, y, theta) configuration of the EE.
  """
  assert(len(theta.shape) == 2)
  assert(theta.shape[1] == 8)

  links = [torch.zeros(batch_size, 3) for _ in range(8)]
  links[0][:,0] = Constants.R8_LENGTH*torch.cos(theta[:,0])
  links[0][:,1] = Constants.R8_LENGTH*torch.sin(theta[:,0])
  links[0][:,2] = theta[:,0]

  for i in range():
    links[i][:,0] = links[i-1][:,0] + Constants.R8_LENGTH*torch.cos(theta[:,i] + links[i-1][:,2])
    links[i][:,1] = links[i-1][:,1] + Constants.R8_LENGTH*torch.sin(theta[:,i] + links[i-1][:,2])
    links[i][:,2] = theta[:,i] + links[i-1][:,2]

  links.reverse()
  return links
