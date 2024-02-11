class ExponentialMovingAverage(object):
  def __init__(self, initial_value, smoothing_factor=2):
    self.ema = initial_value
    self.smoothing_factor = smoothing_factor

  def initialize(self, v):
    self.ema = v

  def update(self, t, v):
    """
    Update the EMA with a new value v at timestep t.
    """
    new_factor = self.smoothing_factor / (1 + t)
    old_factor = 1 - new_factor
    self.ema = new_factor*v + old_factor*self.ema

    return self.ema
