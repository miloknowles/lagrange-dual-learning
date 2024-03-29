import sys; sys.path.append(".."); sys.path.append("../../")
import os, json, time

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils.training_utils import (
  get_best_system_device, save_model, save_multipliers, load_model, count_parameters
)
import utils.paths as paths
from models.simple_network import SimpleNetwork
from kinematics.forward_kinematics import ForwardKinematicsEightLinkTorch, ForwardKinematicsThreeLinkTorch
from kinematics.dataset import IkDataset
from kinematics.penalties import piecewise_joint_limit_penalty, piecewise_circle_penalty

import git
from tensorboardX import SummaryWriter

device = get_best_system_device()


class IkLagrangeDualTrainer(object):
  """
  Train a network to approximate inverse kinematics solutions for a three link
  robotic arm.

  Notes
  -----
  The training objective is to:

  Minimize the L2 norm of the arm's joint angles:
    `min { || theta ||^2 }`              
  
  Such that end effector is at the desired position:
    `|| f(theta) - ee_desired || = 0`

  Input dimensionality (total 20):
   - desired x (float)
   - desired y (float)
   - joint limit min (float)
   - joint limit max (float)
   - obstacle 1 params (4 x float)
   - obstacle 2 params (4 x float)
   - obstacle 3 params (4 x float)
   - obstacle 4 params (4 x float)

  For more information, please see the report PDF.
  """
  def __init__(self, opt):
    torch.backends.cudnn.benchmark = True

    self.opt = opt

    with open(self.opt.config_file, 'r') as f:
      self.json_config = json.load(f)

    input_dim = 20

    # The network outputs a joint angle for each of the links.
    self.model = SimpleNetwork(
      input_dim,
      self.opt.num_links,
      hidden_units=self.opt.hidden_units,
      depth=8,
      dropout=0.05
    ).to(device)

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
        num_lagrange_multipliers += self.opt.num_links*num_dynamic_obstacles
        for oi in range(num_dynamic_obstacles):
          for ji in range(self.opt.num_links):
            self.constraint_names.append("DYN_OB{}_J{}".format(oi+1, ji+1))
      else:
        num_static_obstacles = len(self.json_config["static_constraints"]["obstacles"])
        print("NOTE: Found {} STATIC obstacles in config file".format(num_static_obstacles))
        num_lagrange_multipliers += self.opt.num_links*num_static_obstacles
        for oi in range(num_static_obstacles):
          for ji in range(self.opt.num_links):
            self.constraint_names.append("STAT_OB{}_J{}".format(oi+1, ji+1))

    assert(num_lagrange_multipliers == len(self.constraint_names))

    self.lambda_multipliers = self.opt.initial_lambda * torch.ones(num_lagrange_multipliers).to(device)

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

    print("=======================================\n")
    print("OPTIONS:")
    print(" Model parameters:\n  ", count_parameters(self.model))
    print(" Lagrange iters:\n  ", self.opt.lagrange_iters)
    print(" Train iters:\n  ", self.opt.train_iters)
    print(" Train dataset size:\n  ", self.opt.train_dataset_size)
    print(" Validation dataset size:\n", self.opt.val_dataset_size)
    print(" Num workers:\n  ", self.opt.num_workers)
    print(" Num Lagrange multipliers:\n  ", len(self.lambda_multipliers))
    print(" Initial multiplier values:\n  ", self.opt.initial_lambda)
    print(" Logging to:\n  ", self.log_path)
    print(" Config file:\n  ", self.opt.config_file)
    print(" Load weights folder:\n  ", self.opt.load_weights_folder)
    print(" Load adam state?\n  ", self.opt.load_adam)
    print("=======================================\n")

    if self.opt.load_weights_folder is not None:
      print("NOTE: Load weights path given, going to load model...")
      load_model(
        self.model,
        self.optimizer if self.opt.load_adam else None,
        os.path.join(self.opt.load_weights_folder, "model.pth"),
        os.path.join(self.opt.load_weights_folder, "adam.pth")
      )

    if self.opt.cache_save_path is not None:
      cache_save_path_fmt = os.path.join(self.opt.cache_save_path, "{}_{}.pt")
      cache_save_path_train = cache_save_path_fmt.format(self.opt.model_name, "train")
      cache_save_path_val = cache_save_path_fmt.format(self.opt.model_name, "val")
    else:
      cache_save_path_train = cache_save_path_val = None

    self.train_dataset = IkDataset(self.opt.train_dataset_size, self.opt.num_links, self.json_config,
                                   cache_save_path=cache_save_path_train)
    self.val_dataset = IkDataset(self.opt.val_dataset_size, self.opt.num_links, self.json_config,
                                 cache_save_path=cache_save_path_val)

    self.train_loader = DataLoader(self.train_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers)
    self.val_loader = DataLoader(self.val_dataset, self.opt.batch_size, True, num_workers=self.opt.num_workers)


  def process_batch(self, inputs, lamda):
    """
    Process a batch of inputs through the network, compute constraint violations and losses.
    """
    outputs = {}

    for key in inputs:
      inputs[key] = inputs[key].to(device)

    this_batch_size = len(inputs["q_ee_desired"])

    joint_limits = torch.Tensor(self.json_config["static_constraints"]["joint_limits"]).unsqueeze(0).expand(this_batch_size, -1).to(device)

    pred_joint_theta = self.model(inputs["input_tensor"])
    outputs["pred_joint_theta"] = pred_joint_theta

    q_ee_desired = inputs["q_ee_desired"]

    forw_kinematics_function = {
      3: ForwardKinematicsThreeLinkTorch,
      # 8: ForwardKinematicsEightLinkTorch
    }[self.opt.num_links]

    q_all_joints = forw_kinematics_function(pred_joint_theta)
    q_ee_actual = q_all_joints[0]
    outputs["q_ee_actual"] = q_ee_actual

    # Compute the loss using the current Lagrangian multipliers.
    joint_angle_l2 = 0.5 * pred_joint_theta**2

    # NOTE: The joint angle L2 loss isn't that important, so downweight it...
    outputs["joint_angle_l2"] = 0.01 * joint_angle_l2.mean()

    position_err_sq = 0.5 * (q_ee_desired[:,:2] - q_ee_actual[:,:2])**2
    outputs["position_err_sq"] = position_err_sq

    viol_to_concat = [position_err_sq.sum().unsqueeze(0)]

    # Optionally constraint joint angles to be within a certain range.
    if self.json_config["enforce_joint_limits"] == True:
      joint_limit_violations = torch.zeros(self.opt.num_links).to(device)
      for joint_idx in range(self.opt.num_links):
        joint_limit_violations[joint_idx] = piecewise_joint_limit_penalty(
          pred_joint_theta[:,joint_idx], joint_limits[:,0], joint_limits[:,1],
          inside_slope=self.opt.penalty_good_slope,
          outside_slope=self.opt.penalty_bad_slope).sum()
      viol_to_concat.append(joint_limit_violations)

    # Optionally constrain joints to avoid obstacles.
    ctr = 0
    if self.json_config["enforce_obstacles"] == True:
      if self.json_config["dynamic_constraints"]["random_obstacles"] == True:
        assert("dynamic_obstacles" in inputs)
        num_obstacles = self.json_config["dynamic_constraints"]["random_obstacles_num"]
        obstacle_params = inputs["dynamic_obstacles"] # Shape (b, 4, 4).
        obstacle_type = self.json_config["dynamic_constraints"]["random_obstacles_template"]["type"]
      else:
        num_obstacles = len(self.json_config["static_constraints"]["obstacles"])
        obstacle_type = None if num_obstacles == 0 else self.json_config["static_constraints"]["obstacles"][0]["type"]
        obstacle_params = torch.zeros(this_batch_size, 4, 4)
        for obst_idx, obst_json in enumerate(self.json_config["static_constraints"]["obstacles"]):
          assert(obst_json["type"] == "circle")
          obstacle_params[:,obst_idx,0] = obst_json["x"]
          obstacle_params[:,obst_idx,1] = obst_json["y"]
          obstacle_params[:,obst_idx,2] = obst_json["radius"]
          obstacle_params[:,obst_idx,3] = -1

      obstacle_params = obstacle_params.to(device)

      # Handle static and dynamic objects with the same penalty.
      obstacle_violations = torch.zeros(self.opt.num_links*num_obstacles).to(device)

      for obst_idx in range(num_obstacles):
        params = obstacle_params[:,obst_idx]
        ox, oy, radius, _ = params[:,0], params[:,1], params[:,2], params[:,3]
        for joint_idx in range(self.opt.num_links):
          viol_this_joint = piecewise_circle_penalty(
            q_all_joints[joint_idx][:,0], q_all_joints[joint_idx][:,1], ox, oy, radius,
            inside_slope=self.opt.penalty_bad_slope, outside_slope=self.opt.penalty_good_slope
          )
          obstacle_violations[ctr] = viol_this_joint.sum()
          ctr += 1
      viol_to_concat.append(obstacle_violations)

    # Combine all of the constraint violations into a 1D Tensor.
    outputs["constraint_violations"] = torch.cat(viol_to_concat).to(device)

    # Each constraint violation is weighted by its corresponding multiplier.
    outputs["lagrange_loss"] = (lamda * outputs["constraint_violations"]).sum()

    return outputs

  def train_with_relaxation(self, lamda):
    """
    Do one training epoch of the model on a Lagrangian relaxation parameterized
    by the current multipliers lambda.
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
    Do a single update step on the Lagrange multipliers based on constraint
    violations.
    """
    # Aggregate constraint violations across all of the training examples.
    with torch.no_grad():
      total_constraint_violations = torch.zeros(len(self.val_loader), len(lamda)).to(device)

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

  def main(self):
    """The main training loop."""
    for epoch in range(self.opt.lagrange_iters):
      epoch_start_time = time.time()

      self.train_with_relaxation(self.lambda_multipliers)
      train_time = time.time() - epoch_start_time

      self.lambda_multipliers = self.update_multipliers(self.lambda_multipliers)
      mult_time = time.time() - epoch_start_time - train_time

      self.validate(epoch, self.lambda_multipliers, train_time, mult_time)

      # Periodically save the model weights, multipliers and Adam state.
      if epoch % self.opt.model_save_hz == 0 and epoch > 0:
        save_model(self.model, self.optimizer, os.path.join(self.log_path, "models"), epoch)
        save_multipliers(self.lambda_multipliers, os.path.join(self.log_path, "models"), epoch)
