import unittest
import math
import torch

from kinematics.penalties import piecewise_circle_penalty, piecewise_joint_limit_penalty


def test_piecewise_circle_penalty():
  jx = torch.Tensor([0.0, 0.5, 0.75, 0.5, -1.0])
  jy = torch.Tensor([0.0, 0.5, 0.5, 0.75, 0.5])

  ox = torch.Tensor([0.5])
  oy = torch.Tensor([0.5])
  radius = torch.Tensor([0.25])

  penalty = piecewise_circle_penalty(jx, jy, ox, oy, radius, inside_slope=1.0, outside_slope=0.1)

  # self.assertAlmostEqual(penalty[0], -0.1*(math.sqrt(0.5**2 + 0.5**2) - 0.25))
  # self.assertAlmostEqual(penalty[1], 0.25)
  # self.assertAlmostEqual(penalty[2], 0.0)
  # self.assertAlmostEqual(penalty[3], 0.0)
  # self.assertAlmostEqual(penalty[4], -0.1*1.25)


def test_piecewise_joint_limit_penalty():
  theta = torch.Tensor([-math.pi, math.pi, 0, 0.1])
  limit_min = torch.Tensor([-2*math.pi / 3])
  limit_max = torch.Tensor([2*math.pi / 3])
  viol = piecewise_joint_limit_penalty(theta, limit_min, limit_max, inside_slope=0.1, outside_slope=1.0)

  # self.assertAlmostEqual(viol[0].item(), math.pi / 3)
  # self.assertAlmostEqual(viol[1].item(), math.pi / 3)
  # self.assertAlmostEqual(viol[2].item(), -0.1*2*math.pi / 3)
  # self.assertAlmostEqual(viol[3].item(), -0.1*2*math.pi / 3 + 0.1*0.1)
