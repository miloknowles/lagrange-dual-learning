import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
  """An extremely simple fully-connected network."""
  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    hidden_units: int = 40,
    depth: int = 8,
    dropout: float = 0.05
  ):
    super(SimpleNetwork, self).__init__()

    self.input = nn.Linear(input_dim, hidden_units, bias=True)
    self.hidden = nn.ModuleList(
      nn.Sequential(
        nn.Linear(hidden_units, hidden_units, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout)
      ) for _ in range(depth)
    )
    self.output = nn.Linear(hidden_units, output_dim, bias=True)

  def forward(self, x):
    x = F.relu(self.input(x))
    for layer in self.hidden:
      x = layer(x)
    return self.output(x)
