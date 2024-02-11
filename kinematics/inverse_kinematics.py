import os, math
import numpy as np

from pydrake.all import MathematicalProgram
import pydrake.solvers.mathematicalprogram as mp

from kinematics.drake_utils import ForwardKinematicsThreeLinkConstraint


def IkThreeLinkMP(theta_guess, x_ee_desired, theta_ee_desired=None):
  """
  Use PyDrake's Mathematical Program solver for inverse-kinematics.

  Args:
    theta_guess (np.array) : Shape (3,), the initial guess for joint angles.
    x_ee_desired (np.array) : Shape (2,), desired EE position.
    theta_ee_desired (None or float) : If not None, desired end effector orientation.
  """
  theta_value = np.zeros(3)
  prog = MathematicalProgram()

  # Create decision variables (joint angles).
  theta = prog.NewContinuousVariables(3, 'theta')

  # Cost function is the 2-norm of joint angles.
  prog.AddQuadraticCost((theta**2).sum())

  # Constrain the end effector to be at the desired position.
  ee_position_x, ee_position_y, ee_theta = ForwardKinematicsThreeLinkConstraint(theta)
  prog.AddConstraint(ee_position_x == x_ee_desired[0])
  prog.AddConstraint(ee_position_y == x_ee_desired[1])

  # Optionally add orientation constraint.
  if theta_ee_desired is not None:
    prog.AddConstraint(ee_theta == theta_ee_desired)

  prog.SetInitialGuess(theta, theta_guess)
  result = mp.Solve(prog)
  theta_value = result.GetSolution(theta)

  return theta_value, result.get_solution_result()
