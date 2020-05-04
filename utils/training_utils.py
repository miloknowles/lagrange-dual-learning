import os
import math
import torch


def save_model(model, adam, folder, epoch):
  os.makedirs(os.path.join(folder, "weights_{}".format(epoch)), exist_ok=True)

  model_save_path = os.path.join(folder, "weights_{}".format(epoch), "model.pth")
  torch.save(model.state_dict(), model_save_path)

  if adam is not None:
    adam_save_path = os.path.join(folder, "weights_{}".format(epoch), "adam.pth")
    torch.save(adam.state_dict(), adam_save_path)

  return model_save_path, None if adam is None else adam_save_path


def load_model(model, adam, model_path, adam_path):
  model_dict = model.state_dict()
  saved_model_dict = torch.load(model_path)
  saved_model_dict = {k: v for k, v in saved_model_dict.items() if k in model_dict}
  model_dict.update(saved_model_dict)
  model.load_state_dict(model_dict)

  if adam is not None:
    adam_dict = adam.state_dict()
    saved_adam_dict = torch.load(adam_path)
    adam_dict.update(saved_adam_dict)
    adam.load_state_dict(adam_dict)

  return model, adam


def save_multipliers(lamda, folder, epoch):
  os.makedirs(os.path.join(folder, "weights_{}".format(epoch)), exist_ok=True)
  mult_save_path = os.path.join(folder, "weights_{}".format(epoch), "multipliers.pth")
  torch.save(lamda, mult_save_path)

  return mult_save_path


# From: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def piecewise_circle_penalty(jx, jy, ox, oy, radius, inside_slope=1.0, outside_slope=0.1):
  """
  Determines the penalty for the point (jx, jy) being inside the circular obstacle parameterized
  by the point (ox, oy) and radius.
  """
  assert(inside_slope >= 0 and outside_slope >= 0)

  d = torch.sqrt((jx - ox)**2 + (jy - oy)**2)

  viol_inside = -inside_slope * (d - radius)
  viol_outside = -outside_slope * (d - radius)

  return torch.max(viol_inside, viol_outside)


def piecewise_obstacle_penalty(jx, jy, ox, oy, ow, oh, inside_slope=1.0, outside_slope=0.1):
  """
  Determines the penalty for the point (jx, jy) being inside the obstacle parameterized by
  (ox, oy, ow, oh). The penalty is piecewise linear.
  """
  assert(inside_slope >= 0 and outside_slope >= 0)

  raise NotImplementedError()

  # midpoint_x = ox + 0.5*ow
  # midpoint_y = oy + 0.5*oh

  # viol_x_inside = -inside_slope*torch.abs(jx - midpoint_x) + inside_slope*0.5*ow
  # viol_y_inside = -inside_slope*torch.abs(jy - midpoint_y) + inside_slope*0.5*oh

  # viol_x_outside = -outside_slope*torch.abs(jx - midpoint_x) + outside_slope*0.5*ow
  # viol_y_outside = -outside_slope*torch.abs(jy - midpoint_y) + outside_slope*0.5*oh

  # viol_x = torch.max(viol_x_inside, viol_x_outside)
  # viol_y = torch.max(viol_y_inside, viol_y_outside)

  # return viol_x, viol_y


def piecewise_joint_limit_penalty(theta, limit_min, limit_max, inside_slope=0.1, outside_slope=1.0):
  """
  Determines the penalty for a joint command "theta" being outside of the limits given by
  limit_min and limit_max.
  """
  assert(inside_slope >= 0 and outside_slope >= 0)

  limit_mid = (limit_min + limit_max) / 2.0
  limit_range = torch.abs(limit_max - limit_min)
  viol_inside = inside_slope*torch.abs(theta - limit_mid) - inside_slope*0.5*limit_range
  viol_outside = outside_slope*torch.abs(theta - limit_mid) - outside_slope*0.5*limit_range

  return torch.max(viol_inside, viol_outside)


def no_joint_collisions_rectangle(q_all_joints, x, y, json):
  assert("width" in json and "height" in json)
  w = json["width"]
  h = json["height"]

  for q in q_all_joints:
    if q[0] >= x and q[0] <= (x+w) and q[1] >= y and q[1] <= (y+h):
      return False

  if 0 >= x and 0 <= (x+w) and 0 >= y and 0 <= (y+h):
    return False

  return True


def no_joint_collisions_circle(q_all_joints, x, y, json):
  """
  q_all_joint (list of Tensor) : Each tensor should have shape (3,), containing (x, y, theta).
  x (float) : The x-coord of the circle.
  y (float) : The y-coord of the circle.
  """
  assert("radius" in json)
  radius = json["radius"]

  for q in q_all_joints:
    d = torch.sqrt((q[0] - x)**2 + (q[1] - y)**2)
    if d <= radius:
      return False

  dist_from_origin = math.sqrt(x**2 + y**2)
  if dist_from_origin <= radius:
    return False

  return True
