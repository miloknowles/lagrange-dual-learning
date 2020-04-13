import os, math
import numpy as np
from pydrake.forwarddiff import gradient
from pydrake.all import MathematicalProgram
import pydrake.symbolic as sym

from pydrake.multibody.plant import MultibodyPlant, AddMultibodyPlantSceneGraph
from pydrake.geometry import SceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.multibody import inverse_kinematics as ik
from pydrake.math import RigidTransform
import pydrake.solvers.mathematicalprogram as mp
from pydrake.math import RotationMatrix, RollPitchYaw, RigidTransform

from utils.robot_visualizer import RobotVisualizer
from utils.constants import Constants
from utils.forward_kinematics import *


def IkThreeLinkArmMathematicalProgram(x_ee_desired, theta_ee_desired, theta_initial_guess):
  '''
  x_ee_desired: numpy array of shape (2,), desired end effector position.
  theta_ee_desired: float, desired end effector orientation.
  theta_initial_guess: numpy array of shape (3,), initial guess for the decision variables.
  '''
  theta_value = np.zeros(3)
  prog = MathematicalProgram()

  # create decision variables
  theta = prog.NewContinuousVariables(3, 'theta')

  # Cost function is the 2-norm of joint angles.
  prog.AddQuadraticCost((theta**2).sum())

  # Constrain the end effector to be at the desired position.
  ee_position_x, ee_position_y, ee_theta = ForwardKinematicsThreeLinkConstraint(theta)
  prog.AddConstraint(ee_position_x == x_ee_desired[0])
  prog.AddConstraint(ee_position_y == x_ee_desired[1])
  # prog.AddConstraint(ee_theta == math.pi / 2)

  prog.SetInitialGuess(theta, theta_initial_guess)

  result = mp.Solve(prog)
  theta_value = result.GetSolution(theta)
  return theta_value, result.get_solution_result()


if __name__ == "__main__":
  three_link_viz = RobotVisualizer(num_links=3)

  x_ee_desired = np.array([1.0, 0.5])
  # x_ee_desired = np.array([0, 0])
  theta_ee_desired = np.pi/2
  theta_initial_guess = np.zeros(3)

  # [-7.72442947  1.9551931   7.34003269]
  theta_initial_guess = np.array([
    -7.72442947,  1.9551931,   7.34003269
  ])

  theta_value, result = \
      IkThreeLinkArmMathematicalProgram(x_ee_desired, theta_ee_desired, theta_initial_guess)

  print("Solution result: ", result)
  print("IK solution: ", theta_value)
  print("EE position: ", ForwardKinematicsThreeLink(theta_value))

  # show robot pose and target in meshcat
  three_link_viz.DrawRobot(theta_value)
  three_link_viz.DrawTarget(x_ee_desired)
