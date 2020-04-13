import unittest

from utils.robot_visualizer import RobotVisualizer
from utils.constants import Constants
from utils.forward_kinematics import *
from utils.inverse_kinematics import IkThreeLinkMP


class DrakeIkTest(unittest.TestCase):
  def setUp(self):
    self.viz = RobotVisualizer(num_links=3)

  def test_3link_position(self):
    x_ee_desired = np.array([1.0, 0.5])

    # NOTE: Without an orientation constraint, the solver doesn't find a feasible solution.
    # Need to start near a feasible solution.
    theta_initial_guess = np.array([-7.72442947, 1.9551931, 7.34003269])

    theta_value, result = IkThreeLinkMP(theta_initial_guess, x_ee_desired, theta_ee_desired=None)

    print("Solution result: ", result)
    print("IK solution: ", theta_value)
    print("EE position: ", ForwardKinematicsThreeLink(theta_value))

    self.viz.DrawRobot(theta_value)
    self.viz.DrawTarget(x_ee_desired)

  def test_3link_position_and_orient(self):
    x_ee_desired = np.array([1.0, 0.5])
    theta_ee_desired = np.pi/2
    theta_initial_guess = np.zeros(3)

    theta_value, result = IkThreeLinkMP(theta_initial_guess, x_ee_desired, theta_ee_desired=theta_ee_desired)

    print("Solution result: ", result)
    print("IK solution: ", theta_value)
    print("EE position: ", ForwardKinematicsThreeLink(theta_value))

    self.viz.DrawRobot(theta_value)
    self.viz.DrawTarget(x_ee_desired)
