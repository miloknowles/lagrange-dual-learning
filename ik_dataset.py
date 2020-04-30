import os

import torch
from torch.utils.data import DataLoader, Dataset

from utils.forward_kinematics import *


class IkDataset(Dataset):
  def __init__(self, N, J, json_config, seed=0, cache_save_path=None):
    super(IkDataset, self).__init__()
    self.N = N    # The number of examples.
    self.J = J    # The number of links on this robot.
    self.json_config = json_config

    if cache_save_path is not None and os.path.exists(cache_save_path):
      print("[DATASET] Found cached dataset at {}, loading from there!".format(cache_save_path))
      load_dict = torch.load(cache_save_path)
      self.random_theta = load_dict["random_theta"]
      self.random_ee = load_dict["random_ee"]
      self.random_obstacles = load_dict["random_obstacles"]
      self.tensor_template = load_dict["tensor_template"]
    else:
      print("[DATASET] Initializing dataset from scratch...")
      self.initialize(N, J, json_config, seed=0, cache_save_path=cache_save_path)

      if cache_save_path is not None:
        print("[DATASET] Saving the dataset to {}".format(cache_save_path))
        save_dict = {}
        save_dict["random_theta"] = self.random_theta
        save_dict["random_ee"] = self.random_ee
        save_dict["random_obstacles"] = self.random_obstacles
        save_dict["tensor_template"] = self.tensor_template
        torch.save(save_dict, cache_save_path)

  def initialize(self, N, J, json_config, seed=0, cache_save_path=None):
    # Generate N random points in R^J.
    torch.manual_seed(seed)

    joint_angle_min = json_config["static_constraints"]["joint_limits"][0]
    joint_angle_max = json_config["static_constraints"]["joint_limits"][1]

    print("[DATASET] Joint angle limits:", joint_angle_min, joint_angle_max)

    self.random_theta = torch.empty(N, J).uniform_(joint_angle_min, joint_angle_max)

    forw_kinematics_function = {
      3: ForwardKinematicsThreeLinkTorch,
      8: ForwardKinematicsEightLinkTorch
    }[J]

    self.random_ee = forw_kinematics_function(self.random_theta)[0]
    q_all_joints = forw_kinematics_function(self.random_theta)

    dynamic_obstacles = json_config["dynamic_constraints"]["random_obstacles"]

    def no_joint_collision(q_all_joints, x, y, w, h):
      for q in q_all_joints:
        if q[0] >= x and q[0] <= (x+w) and q[1] >= y and q[1] <= (y+h):
          return False

      if 0 >= x and 0 <= (x+w) and 0 >= y and 0 <= (y+h):
        return False

      return True

    # If obstacles are randomly generated, then we can place them around the joint to avoid collisions.
    self.random_obstacles = None
    if dynamic_obstacles:
      num_obstacles = json_config["dynamic_constraints"]["random_obstacles_num"]
      width = json_config["dynamic_constraints"]["random_obstacle_width"]
      height = json_config["dynamic_constraints"]["random_obstacle_height"]
      print("[DATASET] Generating {} random obstacles for each example (w={} h={})".format(num_obstacles, width, height))

      self.random_obstacles = torch.zeros(len(self.random_ee), 4, 4)
      self.random_obstacles[:,:,2] = width
      self.random_obstacles[:,:,3] = height

      # For each training example, keep generating random obstacles until one isn't in collision.
      for i in range(len(self.random_ee)):
        q_all_joints_this_ex = [q[i] for q in q_all_joints]
        for obst_idx in range(num_obstacles):
          random_xy = torch.empty(2).uniform_(-1.5 - width, 1.5)
          while not no_joint_collision(q_all_joints_this_ex, random_xy[0], random_xy[1], width, height):
            random_xy = torch.empty(2).uniform_(-1.5 - width, 1.5)
          self.random_obstacles[i,obst_idx,:2] = random_xy

    # If obstacles are static, then remove any joint angles that would cause a collision.
    else:
      print("[DATASET] Filtering dataset with {} static obstacles".format(len(json_config["static_constraints"]["obstacles"])))
      for i in range(len(self.random_ee)):
        q_all_joints_this_ex = [q[i].squeeze(0) for q in q_all_joints]
        for static_obstacle in json_config["static_constraints"]["obstacles"]:
          x, y, w, h = static_obstacle
          while not no_joint_collision(q_all_joints_this_ex, x, y, w, h):
            self.random_theta[i] = torch.empty(J).uniform_(joint_angle_min, joint_angle_max)
            q_all_joints_this_ex = [q.squeeze(0) for q in forw_kinematics_function(self.random_theta[i].unsqueeze(0))]
        self.random_ee[i] = q_all_joints_this_ex[0]

    # Make a placeholder tensor that network inputs will be made out of.
    joint_limits = torch.Tensor(self.json_config["static_constraints"]["joint_limits"])

    inputs_to_concat = [
      torch.zeros(2),
      joint_limits,
      torch.zeros(16)
    ]

    for i, params in enumerate(self.json_config["static_constraints"]["obstacles"]):
      inputs_to_concat[2][4*i:4*i+4] = torch.Tensor(params)

    self.tensor_template = torch.cat(inputs_to_concat)

  def __len__(self):
    return self.random_theta.shape[0]

  def __getitem__(self, index):
    output = {}
    output["input_tensor"] = self.tensor_template.clone()
    output["input_tensor"][0:2] = self.random_ee[index,:2]
    output["joint_theta"] = self.random_theta[index]
    output["q_ee_desired"] = self.random_ee[index]

    # If the obstacles are dynamic, replace them in the input tensor.
    if self.random_obstacles is not None:
      output["dynamic_obstacles"] = self.random_obstacles[index]
      output["input_tensor"][4:] = self.random_obstacles[index].flatten()

    return output
