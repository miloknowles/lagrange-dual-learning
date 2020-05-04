import unittest
import math
import torch

from utils.training_utils import *


class TrainingUtilsTest(unittest.TestCase):
  # def test_piecewise_obstacle_penalty(self):
  #   # In order: (0, 0), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0), (-1, 0.5).
  #   jx = torch.Tensor([0.0, 0.5, 0.75, 1.0, -1.0])
  #   jy = torch.Tensor([0.0, 0.5, 0.75, 1.0, 0.5])

  #   # Obtacle from (0.5, 0.5) to (1.0, 1.0).
  #   ox = torch.Tensor([0.5])
  #   oy = torch.Tensor([0.5])
  #   ow = torch.Tensor([0.5])
  #   oh = torch.Tensor([0.5])

  #   vx, vy = piecewise_obstacle_penalty(jx, jy, ox, oy, ow, oh, inside_slope=1.0, outside_slope=0.1)

  #   # The 2nd and 4th points should have zero penalty (at object boundaries).
  #   self.assertEqual(vx[1], 0)
  #   self.assertEqual(vy[1], 0)
  #   self.assertEqual(vx[3], 0)
  #   self.assertEqual(vy[3], 0)

  #   # The point (0, 0) should have a negative penalty of 0.5*0.1
  #   self.assertAlmostEqual(vx[0].item(), -0.1*0.5)
  #   self.assertAlmostEqual(vy[0].item(), -0.1*0.5)

  #   # The point (0.75, 0.75) should have a max penalty of 0.25.
  #   self.assertAlmostEqual(vx[2].item(), 0.25)
  #   self.assertAlmostEqual(vy[2].item(), 0.25)

  #   # The last point (-1, 0.5) should have an x penalty of 0.1*1.5 and y penalty of 0.
  #   self.assertAlmostEqual(vx[4].item(), -0.15, places=3)
  #   self.assertAlmostEqual(vy[4].item(), 0)

  def test_piecewise_circle_penalty(self):
    jx = torch.Tensor([0.0, 0.5, 0.75, 0.5, -1.0])
    jy = torch.Tensor([0.0, 0.5, 0.5, 0.75, 0.5])

    ox = torch.Tensor([0.5])
    oy = torch.Tensor([0.5])
    radius = torch.Tensor([0.25])

    penalty = piecewise_circle_penalty(jx, jy, ox, oy, radius, inside_slope=1.0, outside_slope=0.1)

    self.assertAlmostEqual(penalty[0], -0.1*(math.sqrt(0.5**2 + 0.5**2) - 0.25))
    self.assertAlmostEqual(penalty[1], 0.25)
    self.assertAlmostEqual(penalty[2], 0.0)
    self.assertAlmostEqual(penalty[3], 0.0)
    self.assertAlmostEqual(penalty[4], -0.1*1.25)

  def test_piecewise_joint_limit_penalty(self):
    theta = torch.Tensor([-math.pi, math.pi, 0, 0.1])
    limit_min = torch.Tensor([-2*math.pi / 3])
    limit_max = torch.Tensor([2*math.pi / 3])
    viol = piecewise_joint_limit_penalty(theta, limit_min, limit_max, inside_slope=0.1, outside_slope=1.0)

    self.assertAlmostEqual(viol[0].item(), math.pi / 3)
    self.assertAlmostEqual(viol[1].item(), math.pi / 3)
    self.assertAlmostEqual(viol[2].item(), -0.1*2*math.pi / 3)
    self.assertAlmostEqual(viol[3].item(), -0.1*2*math.pi / 3 + 0.1*0.1)

if __name__ == "__main__":
  unittest.main()
