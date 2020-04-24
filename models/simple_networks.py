import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoLayerNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_units=40):
    super(TwoLayerNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_units, bias=True)
    self.fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.out = nn.Linear(hidden_units, output_dim, bias=True)

  def forward(self, x):
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    return self.out(out)


class FourLayerNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_units=40):
    super(FourLayerNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_units, bias=True)
    self.fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc3 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc4 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.out = nn.Linear(hidden_units, output_dim, bias=True)

  def forward(self, x):
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    out = F.relu(self.fc3(out))
    out = F.relu(self.fc4(out))
    return self.out(out)


class SixLayerNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_units=40):
    super(SixLayerNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_units, bias=True)
    self.fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc3 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc4 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc5 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc6 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.out = nn.Linear(hidden_units, output_dim, bias=True)

  def forward(self, x):
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    out = F.relu(self.fc3(out))
    out = F.relu(self.fc4(out))
    out = F.relu(self.fc5(out))
    out = F.relu(self.fc6(out))
    return self.out(out)


class EightLayerNetwork(nn.Module):
  def __init__(self, input_dim, output_dim, hidden_units=40):
    super(EightLayerNetwork, self).__init__()
    self.fc1 = nn.Linear(input_dim, hidden_units, bias=True)
    self.fc2 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc3 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc4 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc5 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc6 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc7 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.fc8 = nn.Linear(hidden_units, hidden_units, bias=True)
    self.out = nn.Linear(hidden_units, output_dim, bias=True)

  def forward(self, x):
    out = F.relu(self.fc1(x))
    out = F.relu(self.fc2(out))
    out = F.relu(self.fc3(out))
    out = F.relu(self.fc4(out))
    out = F.relu(self.fc5(out))
    out = F.relu(self.fc6(out))
    out = F.relu(self.fc7(out))
    out = F.relu(self.fc8(out))
    return self.out(out)
