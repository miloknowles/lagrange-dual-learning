import time
import torch

from models.simple_network import SimpleNetwork


def test_timing():
  device = torch.device("mps")
  model = SimpleNetwork(20, 3, hidden_units=100, depth=0, dropout=0.05).to(device)

  trials = 1000

  for batch_size in [1, 8, 64, 512]:
    input_tensor = torch.zeros(batch_size, 20).to(device)
    t0 = time.time()
    with torch.no_grad():
      for t in range(trials):
        _ = model(input_tensor)
    elap = time.time() - t0

    avg = 1e6 * elap / (trials * batch_size)
    print("Avg time = {} microsec (batch size = {})".format(avg, batch_size))
