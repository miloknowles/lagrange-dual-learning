import numpy as np
import pydrake.symbolic as sym

from .constants import Constants


def ForwardKinematicsTwoLinkConstraint(theta):
  """
  Does symbolic forward kinematics on the 2-link robot arm, useful for adding EE constraints to PyDrake.

  Args:
    theta (np.array) : Shape (2,), contains the three joint angles of the robot arm.

  Returns:
    tuple of (3) Variable objects
  """
  x_ee = np.empty(3, dtype=object)
  x_ee_0 = Constants.R2_LENGTH1*sym.cos(theta[0]) + Constants.R2_LENGTH2*sym.cos(theta[0] + theta[1])
  x_ee_1 = Constants.R2_LENGTH1*sym.sin(theta[0]) + Constants.R2_LENGTH2*sym.sin(theta[0] + theta[1])
  x_ee_theta = theta[0] + theta[1]

  return x_ee_0, x_ee_1, x_ee_theta


def ForwardKinematicsThreeLinkConstraint(theta):
  """
  Does symbolic forward kinematics on the 3-link robot arm, useful for adding EE constraints to PyDrake.

  Args:
    theta (np.array) : A list of (3) PyDrake Variable objects, representing the (3) joint angles

  Returns:
    tuple of (3) Variable objects
  """
  theta_0 = theta[0]
  theta_01 = theta_0 + theta[1]
  theta_012 = theta_01 + theta[2]

  x3_ee_0 = Constants.R3_LENGTH1*sym.cos(theta_0) + Constants.R3_LENGTH2*sym.cos(theta_01) + Constants.R3_LENGTH3*sym.cos(theta_012)
  x3_ee_1 = Constants.R3_LENGTH1*sym.sin(theta_0) + Constants.R3_LENGTH2*sym.sin(theta_01) + Constants.R3_LENGTH3*sym.sin(theta_012)
  x3_ee_theta = theta_012

  return x3_ee_0, x3_ee_1, x3_ee_theta
