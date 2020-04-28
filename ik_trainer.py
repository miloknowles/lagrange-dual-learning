import math, os, json, time

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from utils.constants import Constants
from utils.forward_kinematics import *
from utils.ema import ExponentialMovingAverage
from utils.training_utils import *
from models.simple_networks import *
from models.residual_network import *

import git
from tensorboardX import SummaryWriter


class JointAngleDataset(Dataset):
  def __init__(self, N, J, json_config, seed=0):
    super(JointAngleDataset, self).__init__()
    self.N = N    # The number of examples.
    self.J = J    # The number of links on this robot.
    self.json_config = json_config

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
      return True

    # If obstacles are randomly generated, then we can place them around the joint to avoid collisions.
    self.random_obstacles = None
    if dynamic_obstacles:
      num_obstacles = json_config["dynamic_constraints"]["random_obstacles_num"]
      width = json_config["dynamic_constraints"]["random_obstacle_width"]
      height = json_config["dynamic_constraints"]["random_obstacle_height"]
      print("[DATASET] Generating {} random obstacles for each example".format(num_obstacles))

      self.random_obstacles = torch.zeros(len(self.random_ee), 4, 4)
      self.random_obstacles[:,:,2] = width
      self.random_obstacles[:,:,3] = height

      # For each training example, keep generating random obstacles until one isn't in collision.
      for i in range(len(self.random_ee)):
        q_all_joints_this_ex = [q[i] for q in q_all_joints]
        for obst_idx in range(num_obstacles):
          random_xy = torch.empty(2).uniform_(-1.5 - width, 1.5)
          while not no_joint_collision(q_all_joints_this_ex, random_xy[0], random_xy[1], width, height):
            random_xy = torch.empty(2).uniform_(-2 - width, 2)
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

      # valid_mask = torch.ones(len(self.random_ee)).long()
      # for static_obstacle in json_config["static_constraints"]["obstacles"]:
      #   x, y, w, h = static_obstacle
      #   for q in q_all_joints:
      #     mask_x = (q[:,0] < x) | (q[:,0] > (x+w))
      #     mask_y = (q[:,1] < y) | (q[:,1] > (y+h))
      #     valid_mask *= (mask_x & mask_y)
      # print("[DATASET] Had to remove {} examples that violate obstacle constraints".format((valid_mask == 0).sum()))

      # valid_indices = torch.from_numpy(np.argwhere(valid_mask.numpy())).squeeze(1)
      # self.random_theta = self.random_theta[valid_indices]
      # self.random_ee = self.random_ee[valid_indices]

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


class IkLagrangeDualTrainer(object):
  """
  Train a network to approximate inverse kinematics solutions for a three link robotic arm.

  min { ||theta||^2 }                    ==> Minimize L2 norm of joint angles.
  s.t. ||f(theta) - ee_desired|| = 0     ==> Such that end effector is at desired position.

  Network Inputs:
   - desired x
   - desired y
   - joint limit min
   - joint limit max
   - obstacle 1 params (4)
   - obstacle 2 params (4)
   - obstacle 3 params (4)
   - obstacle 4 params (4)

  Total = 20
  """
  def __init__(self, opt):
    torch.backends.cudnn.benchmark = True

    self.opt = opt

    config_file_path = os.path.join("/home/milo/lagrange-dual-learning/", self.opt.config_file)
    with open(config_file_path, 'r') as f:
      self.json_config = json.load(f)

    self.device = torch.device("cuda")

    num_network_inputs = 20

    # The network outputs a joint angle for each of the links.
    self.model = EightLayerNetwork(num_network_inputs, self.opt.num_links, hidden_units=self.opt.hidden_units).to(self.device)

    self.optimizer = Adam(self.model.parameters(), lr=self.opt.optimizer_lr, betas=(0.9, 0.999))

    # Figure out how many constraints/multipliers will be needed. Store human-readable names for them.
    num_lagrange_multipliers = 1
    self.constraint_names = ["EE"]

    if self.json_config["enforce_joint_limits"] == True:
      print("NOTE: Enforcing joint limit constraints")
      num_lagrange_multipliers += self.opt.num_links
      self.constraint_names.extend(["JL{}".format(i+1) for i in range(self.opt.num_links)])

    if self.json_config["enforce_obstacles"] == True:
      print("NOTE: Enforcing obstacle constraints")
      if self.json_config["dynamic_constraints"]["random_obstacles"] == True:
        num_dynamic_obstacles = self.json_config["dynamic_constraints"]["random_obstacles_num"]
        print("NOTE: Config file says to generate {} dynamic obstacles".format(num_dynamic_obstacles))
        num_lagrange_multipliers += 2*self.opt.num_links*num_dynamic_obstacles
        for oi in range(num_dynamic_obstacles):
          for ji in range(self.opt.num_links):
            for x_or_y in ("x", "y"):
              self.constraint_names.append("DYN_OB{}_J{}_{}".format(oi+1, ji+1, x_or_y))
      else:
        num_static_obstacles = len(self.json_config["static_constraints"]["obstacles"])
        print("NOTE: Found {} STATIC obstacles in config file".format(num_static_obstacles))
        num_lagrange_multipliers += 2*self.opt.num_links*num_static_obstacles
        for oi in range(num_static_obstacles):
          for ji in range(self.opt.num_links):
            for x_or_y in ("x", "y"):
              self.constraint_names.append("STAT_OB{}_J{}_{}".format(oi+1, ji+1, x_or_y))

    assert(num_lagrange_multipliers == len(self.constraint_names))

    self.lamda = self.opt.initial_lambda * torch.ones(num_lagrange_multipliers).to(self.device)

    self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)
    os.makedirs(self.log_path, exist_ok=True)

    # https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    self.opt.commit_has = sha
    with open(os.path.join(self.log_path, "opt.json"), "w") as f:
      f.write(json.dumps(self.opt.__dict__, sort_keys=True, indent=4) + "\n")
    with open(os.path.join(self.log_path, "config.json"), "w") as f:
      f.write(json.dumps(self.json_config, indent=4) + "\n")
    with open(os.path.join(self.log_path, "constraint_names.txt"), "w") as f:
      for i, name in enumerate(self.constraint_names):
        f.write("{},{}\n".format(i, name))

    self.writer = SummaryWriter(logdir=self.log_path)

    print("========== TRAINING SETTINGS ==========")
    print("Model parameters:\n  ", count_parameters(self.model))
    print("Lagrange iters:\n  ", self.opt.lagrange_iters)
    print("Train iters:\n  ", self.opt.train_iters)
    print("Train dataset size:\n  ", self.opt.train_dataset_size)
    print("Validation dataset size:\n", self.opt.val_dataset_size)
    print("Num workers:\n  ", self.opt.num_workers)
    print("Num Lagrange multipliers:\n  ", len(self.lamda))
    print("Initial multiplier values:\n  ", self.opt.initial_lambda)
    print("Logging to:\n  ", self.log_path)
    print("Config file:\n  ", self.opt.config_file)
    print("Load weights folder:\n  ", self.opt.load_weights_folder)
    print("Load adam state?\n  ", self.opt.load_adam)
    print("=======================================\n")

    if self.opt.load_weights_folder is not None:
      print("NOTE: Load weights path given, going to load model...")
      load_model(self.model, self.optimizer if self.opt.load_adam else None,
                os.path.join(self.opt.load_weights_folder, "model.pth"),
                os.path.join(self.opt.load_weights_folder, "adam.pth"))

    # print("==> PROBLEM CONSTRAINT CONFIGURATION:")
    # print(json.dumps(self.json_config, indent=2))

    self.train_dataset = JointAngleDataset(self.opt.train_dataset_size, self.opt.num_links, self.json_config)
    self.val_dataset = JointAngleDataset(self.opt.val_dataset_size, self.opt.num_links, self.json_config)

    self.train_loader = DataLoader(self.train_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers)
    self.val_loader = DataLoader(self.val_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers)

  def main(self):
    """
    Main training loop.
    """
    for epoch in range(self.opt.lagrange_iters):
      epoch_start_time = time.time()

      self.train_with_relaxation(self.lamda)
      train_time = time.time() - epoch_start_time

      self.lamda = self.update_multipliers(self.lamda)
      mult_time = time.time() - epoch_start_time - train_time

      self.validate(epoch, self.lamda, train_time, mult_time)

      # Periodically save the model weights, multipliers and Adam state.
      if epoch % self.opt.model_save_hz == 0 and epoch > 0:
        save_model(self.model, self.optimizer, os.path.join(self.log_path, "models"), epoch)
        save_multipliers(self.lamda, os.path.join(self.log_path, "models"), epoch)

  def process_batch(self, inputs, lamda):
    """
    Process a batch of inputs through the network, compute constraint violations and losses.
    """
    outputs = {}

    for key in inputs:
      inputs[key] = inputs[key].to(self.device)

    this_batch_size = len(inputs["q_ee_desired"])

    joint_limits = torch.Tensor(self.json_config["static_constraints"]["joint_limits"]).unsqueeze(0).expand(this_batch_size, -1).to(self.device)

    pred_joint_theta = self.model(inputs["input_tensor"])
    outputs["pred_joint_theta"] = pred_joint_theta

    q_ee_desired = inputs["q_ee_desired"]

    forw_kinematics_function = {
      3: ForwardKinematicsThreeLinkTorch,
      8: ForwardKinematicsEightLinkTorch
    }[self.opt.num_links]

    q_all_joints = forw_kinematics_function(pred_joint_theta)
    q_ee_actual = q_all_joints[0]
    outputs["q_ee_actual"] = q_ee_actual

    # Compute the loss using the current Lagrangian multipliers.
    joint_angle_l2 = 0.5 * pred_joint_theta**2
    outputs["joint_angle_l2"] = joint_angle_l2.mean()

    position_err_sq = 0.5 * (q_ee_desired[:,:2] - q_ee_actual[:,:2])**2

    viol_to_concat = [position_err_sq.sum().unsqueeze(0)]

    # Optionally constraint joint angles to be within a certain range.
    if self.json_config["enforce_joint_limits"] == True:
      joint_limit_violations = torch.zeros(self.opt.num_links).to(self.device)
      middle_of_joint_limits = joint_limits.mean(axis=1)
      joint_limit_range = torch.abs(joint_limits[:,1] - joint_limits[:,0])
      for joint_idx in range(self.opt.num_links):
        # Negative if joints are within the joint limits and zero at limit. Positive outside.
        violation_this_joint = torch.abs(pred_joint_theta[:,joint_idx] - middle_of_joint_limits) - 0.5*joint_limit_range
        joint_limit_violations[joint_idx] = violation_this_joint.sum()
      viol_to_concat.append(joint_limit_violations)

    # Optionally constrain joints to avoid obstacles.
    ctr = 0
    if self.json_config["enforce_obstacles"] == True:
      if self.json_config["dynamic_constraints"]["random_obstacles"] == True:
        assert("dynamic_obstacles" in inputs)
        num_obstacles = self.json_config["dynamic_constraints"]["random_obstacles_num"]
        obstacle_params = inputs["dynamic_obstacles"] # Shape (b, 4, 4).
      else:
        num_obstacles = len(self.json_config["static_constraints"]["obstacles"])
        obstacle_params = torch.Tensor(self.json_config["static_constraints"]["obstacles"]).to(self.device)
        obstacle_params = obstacle_params.unsqueeze(0).expand(this_batch_size, -1, -1) # Shape (b, 4, 4).

      # Handle static and dynamic objects with the same penalty.
      obstacle_violations = torch.zeros(2*self.opt.num_links*num_obstacles).to(self.device)

      for obst_idx in range(num_obstacles):
        params = obstacle_params[:,obst_idx]
        midpoint_x = params[:,0] + 0.5*params[:,2]
        midpoint_y = params[:,1] + 0.5*params[:,3]

        for joint_idx in range(self.opt.num_links):
          # Negative if joints are outside of obstacle, zero at boundary, positive inside.
          viol_this_joint_x = 0.5*params[:,2] - torch.abs(q_all_joints[joint_idx][:,0] - midpoint_x)
          viol_this_joint_y = 0.5*params[:,3] - torch.abs(q_all_joints[joint_idx][:,1] - midpoint_y)
          obstacle_violations[ctr] = viol_this_joint_x.sum()
          obstacle_violations[ctr+1] = viol_this_joint_y.sum()
          ctr += 2
      viol_to_concat.append(obstacle_violations)

    # Combine all of the constraint violations into a 1D Tensor.
    outputs["constraint_violations"] = torch.cat(viol_to_concat).to(self.device)

    # Each constraint violation is weighted by its corresponding multiplier.
    outputs["lagrange_loss"] = (lamda * outputs["constraint_violations"]).sum()

    return outputs

  def train_with_relaxation(self, lamda):
    """
    Do one training epoch of the model on a Lagrangian relaxation parameterized by the current
    multipliers lambda.
    """
    # Train the self.model using the current Lagrange relaxation.
    # random_indices = torch.randperm(len(self.train_loader))[:self.opt.train_iters]

    for ti, inputs in enumerate(self.train_loader):
      if ti >= self.opt.train_iters:
        break
      outputs = self.process_batch(inputs, lamda)
      self.model.zero_grad()
      outputs["lagrange_loss"].backward()
      self.optimizer.step()

  def update_multipliers(self, lamda):
    """
    Do a single update step on the Lagrange multipliers based on constraint violations.
    """
    # Aggregate constraint violations across all of the training examples.
    with torch.no_grad():
      total_constraint_violations = torch.zeros(len(self.val_loader), len(lamda)).to(self.device)

      for ti, inputs in enumerate(self.val_loader):
        outputs = self.process_batch(inputs, lamda)
        total_constraint_violations[ti,:] = outputs["constraint_violations"]

      lamda = lamda + self.opt.multiplier_lr*total_constraint_violations.sum(axis=0)

      # If a constraint is really easy to satisfy (and always negative), then lamda could become
      # unboundedly negative, effective turning it off. Make sure this doesn't happen.
      lamda = lamda.clamp(min=0)

    return lamda

  def validate(self, epoch, lamda, train_time, mult_time):
    """
    Test the model on a validation set to see if the constraint violation and loss is improving.
    """
    mean_supervised_loss = torch.zeros(len(self.val_loader))
    mean_constraint_violation = torch.zeros(len(self.val_loader), len(lamda))
    mean_lagrange_loss = torch.zeros(len(self.val_loader))

    with torch.no_grad():
      for vi, inputs in enumerate(self.val_loader):
        outputs = self.process_batch(inputs, lamda)
        mean_supervised_loss[vi] = outputs["joint_angle_l2"]
        mean_constraint_violation[vi,:] = outputs["constraint_violations"]
        mean_lagrange_loss[vi] = outputs["lagrange_loss"]

    mean_constraint_violation = mean_constraint_violation.mean(axis=0)

    print("==> Epoch {} (Train Time = {:.3f} sec, Mult Time = {:.3f} sec)\n  Orig Loss={}\n  Constraints={}\n  Lagrange Loss={}\n  Multipliers={}".format(
        epoch, train_time, mult_time, mean_supervised_loss.mean(), mean_constraint_violation, mean_lagrange_loss.mean(), lamda))

    self.writer.add_scalar("loss/supervised", mean_supervised_loss.mean(), epoch)
    self.writer.add_scalar("loss/lagrange", mean_lagrange_loss.mean(), epoch)

    # Add all of the lagrange multipliers to a single plot.
    self.writer.add_scalars(
        "multipliers",
        {self.constraint_names[mult_idx]: mult_val for (mult_idx, mult_val) in enumerate(lamda)},
        global_step=epoch)

    # Add all of the constraints to a single plot.
    self.writer.add_scalars(
        "constraints",
        {self.constraint_names[cst_idx]: cst_val for (cst_idx, cst_val) in enumerate(mean_constraint_violation)},
        global_step=epoch)

    self.writer.close()
