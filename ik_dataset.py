import os

import torch
from torch.utils.data import DataLoader, Dataset

from utils.forward_kinematics import *
from utils.training_utils import *


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

    # If obstacles are randomly generated, then we can place them around the joint to avoid collisions.
    self.random_obstacles = None
    if dynamic_obstacles:
      num_obstacles = json_config["dynamic_constraints"]["random_obstacles_num"]
      print("[DATASET] Generating {} random obstacles for each example".format(num_obstacles))

      # NOTE: Each obstacle can have up to 4 params. In the case of the circle, only 3.
      self.random_obstacles = torch.zeros(len(self.random_ee), 4, 4)
      obst_template = json_config["dynamic_constraints"]["random_obstacles_template"]
      print("[DATASET] Obstacle template:\n", obst_template)

      if obst_template["type"] == "circle":
        self.random_obstacles[:,:,2] = obst_template["radius"]
      elif obst_template["type"] == "rectangle":
        self.random_obstacles[:,:,2] = obst_template["width"]
        self.random_obstacles[:,:,3] = obst_template["height"]
      else:
        raise NotImplementedError("Unknown obstacle type {}".format(obst_template["type"]))

      no_collision_fn = {
        "rectangle": no_joint_collisions_rectangle,
        "circle": no_joint_collisions_circle
      }[obst_template["type"]]

      # For each training example, keep generating random obstacles until one isn't in collision.
      for i in range(len(self.random_ee)):
        q_all_joints_this_ex = [q[i] for q in q_all_joints]
        for obst_idx in range(num_obstacles):
          random_xy = torch.empty(2).uniform_(-1.5, 1.5)
          while not no_collision_fn(q_all_joints_this_ex, random_xy[0], random_xy[1], obst_template):
            random_xy = torch.empty(2).uniform_(-1.5, 1.5)
          self.random_obstacles[i,obst_idx,:2] = random_xy

    # If obstacles are static, keep sampling joint configurations until there are no collisions.
    else:
      print("[DATASET] Filtering dataset with {} static obstacles".format(len(json_config["static_constraints"]["obstacles"])))
      for i in range(len(self.random_ee)):
        q_all_joints_this_ex = [q[i].squeeze(0) for q in q_all_joints]
        for obst_json in json_config["static_constraints"]["obstacles"]:
          no_collision_fn = {
            "rectangle": no_joint_collisions_rectangle,
            "circle": no_joint_collisions_circle
          }[obst_json["type"]]

          while not no_joint_collisions_circle(q_all_joints_this_ex, obst_json["x"], obst_json["y"], obst_json):
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
      if params["type"] == "circle":
        inputs_to_concat[2][4*i:4*i+4] = torch.Tensor([params["x"], params["y"], params["radius"], -1])
      else:
        raise NotImplementedError()

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
