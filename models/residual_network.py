import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(BasicBlock, self).__init__()
    self.lru = nn.LeakyReLU(negative_slope=0.01)
    self.fc1 = nn.Linear(input_dim, output_dim)
    self.fc2 = nn.Linear(output_dim, output_dim)

  def forward(self, x):
    identity = x
    out = self.lru(self.fc1(x))
    out = self.fc2(out)
    out += identity
    return self.lru(out)


class ResidualNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_units=40):
    super(ResidualNetwork, self).__init__()
    self.resize = nn.Linear(input_dim, hidden_units)
    self.bb1 = BasicBlock(hidden_units, hidden_units)
    self.bb2 = BasicBlock(hidden_units, hidden_units)
    self.bb3 = BasicBlock(hidden_units, hidden_units)
    self.bb4 = BasicBlock(hidden_units, hidden_units)
    self.bb5 = BasicBlock(hidden_units, hidden_units)
    self.final_fc = nn.Linear(hidden_units, output_dim)

  def forward(self, x):
    out = self.resize(x)
    out = self.bb1(out)
    out = self.bb2(out)
    out = self.bb3(out)
    out = self.bb4(out)
    out = self.bb5(out)
    out = self.final_fc(out)
    return out
