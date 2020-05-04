import os, sys

import torch
from matplotlib import pyplot as plt

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

# Hack, but avoids annoying import errors.
sys.path.append("/home/milo/lagrange-dual-learning/")
from ik_trainer import IkLagrangeDualTrainer
from ik_dataset import IkDataset
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

      if obst_params["type"] == "circle":
        x, y, radius = obst_params["x"], obst_params["y"], obst_params["radius"]
        shapes[obst_name].set_object(g.Sphere(radius))
        shapes[obst_name].set_transform(tf.translation_matrix([0, x, y]))
      else:
        x, y, w, h = obst_params["x"], obst_params["y"], obst_params["width"], obst_params["height"]
        shapes[obst_name].set_object(g.Box([0.1, w, h]))
        # NOTE: I think the origin of the object is its center? So need to add half width and height.
        shapes[obst_name].set_transform(tf.translation_matrix([0, x+0.5*w, y+0.5*h]))

  position_err = []
  constraint_viol = []

  with torch.no_grad():
    for i, inputs in enumerate(trainer.val_dataset):
      print("Processing example {}...".format(i+1))
      for key in inputs:
        inputs[key] = inputs[key].to(trainer.device).unsqueeze(0)

      # if i == 100:
        # break

      outputs = trainer.process_batch(inputs, trainer.lamda)
      joint_angles = outputs["pred_joint_theta"].squeeze(0).cpu().numpy()
      joint_angles_gt = inputs["joint_theta"].squeeze(0).cpu().numpy()
      print("Predicted joint angles:", joint_angles)

      position_err.append(torch.sqrt(outputs["position_err_sq"]))
      constraint_viol.append(outputs["constraint_violations"].unsqueeze(0))

      if not opt.no_plot:
        if opt.show_groundtruth_theta:
          rviz.DrawRobot(joint_angles_gt)
        else:
          rviz.DrawRobot(joint_angles)

        if random_obstacles:
          assert("dynamic_obstacles" in inputs)

          for j in range(random_obstacles_num):
            obst_name = "obst_{}".format(j+1)
            x, y, radius, _ = inputs["dynamic_obstacles"].squeeze(0)[j].cpu().numpy().tolist()
            shapes[obst_name].set_object(g.Sphere(radius))
            shapes[obst_name].set_transform(tf.translation_matrix([0, x, y]))
            # shapes[obst_name].set_object(g.Box([0.1, w, h]))
            # shapes[obst_name].set_transform(tf.translation_matrix([0, x+0.5*w, y+0.5*h]))

      ee_desired_position = inputs["q_ee_desired"].squeeze(0)[:2].cpu().numpy()

      if not opt.no_plot:
        rviz.DrawTarget(ee_desired_position, radius=0.2)
        plt.waitforbuttonpress(timeout=-1)

  position_err = torch.cat(position_err).flatten().mean().cpu()
  constraint_viol = torch.cat(constraint_viol).cpu()
  num_violations = (constraint_viol > 1e-4).sum(axis=0)

  # How often are any of the joint limits violated?
  joint_limit_indices = [i for i in range(len(trainer.constraint_names)) if "JL" in trainer.constraint_names[i]]
  joint_limit_violated = (constraint_viol[:,joint_limit_indices] > 0)
  any_joint_limit_violated = (joint_limit_violated.sum(axis=1) > 0).sum().item()

  # How often are any of the obstacles violated?
  obstacle_indices = [i for i in range(len(trainer.constraint_names)) if "OB" in trainer.constraint_names[i]]
  print(obstacle_indices)
  print(trainer.constraint_names)
  obstacle_violated = (constraint_viol[:,obstacle_indices] > 0)
  print(obstacle_violated)
  any_obstacle_violated = (obstacle_violated.sum(axis=1) > 0).sum().item()

  for i, name in enumerate(trainer.constraint_names):
    print("{} | {} | {}%".format(name, num_violations[i], float(num_violations[i]) / len(trainer.val_dataset)))

  print("Avg position error =", position_err)
  print("Joint limit (%) =", float(any_joint_limit_violated) / len(trainer.val_dataset))
  print("Obstacle (%)=", float(any_obstacle_violated) / len(trainer.val_dataset))


if __name__ == "__main__":
  opt = IkOptions()
  trainer = IkLagrangeDualTrainer(opt.parse())
  visualize(opt.parse(), trainer)

