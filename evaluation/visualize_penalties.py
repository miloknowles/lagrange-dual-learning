import math
import numpy as np
import matplotlib.pyplot as plt


def plot_joint_limit(alpha_in, alpha_out):
  x = np.linspace(-3, 3, 500)
  limit = 2*math.pi / 3
  inside = -alpha_in*(limit - np.abs(x))
  outside = -alpha_out*(limit - np.abs(x))
  y = np.maximum(inside, outside)
  plt.plot(x, y, label="Penalty Amount", color="black")
  plt.title("Joint Limit Penalty Function")
  plt.xlabel("theta")
  plt.ylabel("Penalty")
  plt.vlines(-limit, -10, 10, linestyles="dashed", colors="red", label="Angle Limits")
  plt.vlines(limit, -10, 10, linestyles="dashed", colors="red")
  plt.ylim([-0.5, 0.9])
  plt.grid()
  plt.legend()
  plt.show()


def plot_obstacle(alpha_in, alpha_out):
  x = np.linspace(0, 5, 500)
  radius = 1.0
  inside = alpha_in*(radius - np.abs(x))
  outside = alpha_out*(radius - np.abs(x))
  y = np.maximum(inside, outside)
  plt.plot(x, y, label="Penalty Amount", color="black")
  plt.title("Obstacle Collision Penalty Function")
  plt.xlabel("Distance from Center")
  plt.ylabel("Penalty")
  plt.vlines(radius, -10, 10, linestyles="dashed", colors="red", label="Obstacle Radius")
  plt.ylim([-0.8, 1.0])
  plt.grid()
  plt.legend()
  plt.show()


if __name__ == "__main__":
  # plot_joint_limit(0.2, 1.0)
  plot_obstacle(1.0, 0.2)
