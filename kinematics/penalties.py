import math
import torch


def piecewise_circle_penalty(jx, jy, ox, oy, radius, inside_slope=1.0, outside_slope=0.1) -> torch.Tensor:
  """
  Determines the penalty for the point `(jx, jy)` being inside the circular
  obstacle parameterized by the point `(ox, oy)` and radius.
  """
  assert(inside_slope >= 0 and outside_slope >= 0)

  d = torch.sqrt((jx - ox)**2 + (jy - oy)**2)

  viol_inside = -inside_slope * (d - radius)
  viol_outside = -outside_slope * (d - radius)

  return torch.max(viol_inside, viol_outside)


def piecewise_obstacle_penalty(jx, jy, ox, oy, ow, oh, inside_slope=1.0, outside_slope=0.1) -> torch.Tensor:
  """
  Determines the penalty for the point `(jx, jy)` being inside the obstacle
  parameterized by `(ox, oy, ow, oh)`. The penalty is piecewise linear.
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
