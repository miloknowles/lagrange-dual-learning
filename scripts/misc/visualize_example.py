import os
import numpy as np

from pydrake.systems.drawing import plot_system_graphviz, plot_graphviz
from pydrake.all import MathematicalProgram
import pydrake.solvers.mathematicalprogram as mp
from pydrake.math import RotationMatrix, RollPitchYaw, RigidTransform

from utils.robot_visualizer import RobotVisualizer

# Creates visualizer for robots.
# two_link_viz = RobotVisualizer(num_links=2)
three_link_viz = RobotVisualizer(num_links=3)

# Draw the three link robot
three_link_viz.DrawRobot([np.pi/2, -np.pi/2, -np.pi/2])
