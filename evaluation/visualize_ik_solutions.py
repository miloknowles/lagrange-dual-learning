import os, sys

import torch
from matplotlib import pyplot as plt

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

# Hack, but avoids annoying import errors.
sys.path.append("/home/milo/lagrange-dual-learning/")
from ik_trainer import IkLagrangeDualTrainer, IkDataset
from ik_options import IkOptions

from utils.robot_visualizer import RobotVisualizer


def visualize(opt, trainer):
  assert(opt.num_links == 3)

  # NOTE: Need to run meshcat-server in a different terminal or this will wait forever!
  rviz = RobotVisualizer(num_links=opt.num_links)

  # Draw the static obstacles.
  shapes = rviz.vis["shapes"]

  random_obstacles = trainer.json_config["dynamic_constraints"]["random_obstacles"]
  random_obstacles_num = trainer.json_config["dynamic_constraints"]["random_obstacles_num"]

  # If obstacles are all static, render them once at the beginning.
  if not random_obstacles:
    for i, obst_params in enumerate(trainer.json_config["static_constraints"]["obstacles"]):
      # Make a box with the desired height and width.
      obst_name = "obst_{}".format(i+1)
      x, y, w, h = obst_params
      shapes[obst_name].set_object(g.Box([0.1, w, h]))
      # NOTE: I think the origin of the object is its center? So need to add half width and height.
      shapes[obst_name].set_transform(tf.translation_matrix([0, x+0.5*w, y+0.5*h]))

  with torch.no_grad():
    for i, inputs in enumerate(trainer.val_dataset):
      print("Processing example {}...".format(i+1))
      for key in inputs:
        inputs[key] = inputs[key].to(trainer.device).unsqueeze(0)

      outputs = trainer.process_batch(inputs, trainer.lamda)
      joint_angles = outputs["pred_joint_theta"].squeeze(0).cpu().numpy()
      joint_angles_gt = inputs["joint_theta"].squeeze(0).cpu().numpy()
      print("Predicted joint angles:", joint_angles)

      if opt.show_groundtruth_theta:
        rviz.DrawRobot(joint_angles_gt)
      else:
        rviz.DrawRobot(joint_angles)

      if random_obstacles:
        assert("dynamic_obstacles" in inputs)

        for j in range(random_obstacles_num):
          obst_name = "obst_{}".format(j+1)
          x, y, w, h = inputs["dynamic_obstacles"].squeeze(0)[j].cpu().numpy().tolist()
          shapes[obst_name].set_object(g.Box([0.1, w, h]))
          shapes[obst_name].set_transform(tf.translation_matrix([0, x+0.5*w, y+0.5*h]))

      ee_desired_position = inputs["q_ee_desired"].squeeze(0)[:2].cpu().numpy()
      rviz.DrawTarget(ee_desired_position, radius=0.2)

      plt.waitforbuttonpress(timeout=-1)


if __name__ == "__main__":
  opt = IkOptions()
  trainer = IkLagrangeDualTrainer(opt.parse())
  visualize(opt.parse(), trainer)

