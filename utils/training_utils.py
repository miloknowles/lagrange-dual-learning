import os
import torch


def save_model(
  model: torch.nn.Module, adam: torch.optim.Optimizer, folder: str, epoch: int
) -> str:
  """Save model and optimizer parameters.
  
  Returns
  -------
  The path where the model and optimizer are saved.
  """
  os.makedirs(os.path.join(folder, "weights_{}".format(epoch)), exist_ok=True)

  model_save_path = os.path.join(folder, "weights_{}".format(epoch), "model.pth")
  torch.save(model.state_dict(), model_save_path)

  if adam is not None:
    adam_save_path = os.path.join(folder, "weights_{}".format(epoch), "adam.pth")
    torch.save(adam.state_dict(), adam_save_path)

  return model_save_path, None if adam is None else adam_save_path


def load_model(
  model: torch.nn.Module, adam: torch.optim.Optimizer, model_path: str, adam_path: str
) -> tuple[torch.nn.Module, torch.optim.Adam]:
  """Load a model and optimizer from checkpoints.
  
  Returns
  -------
  The loaded model and Adam optimizer.
  """
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


def save_multipliers(lambda_multipliers: torch.Tensor, folder: str, epoch: int) -> str:
  """Save Lagrange multipliers.
  
  Returns
  -------
  The path where the multipliers are saved.
  """
  os.makedirs(os.path.join(folder, "weights_{}".format(epoch)), exist_ok=True)
  outpath = os.path.join(folder, "weights_{}".format(epoch), "multipliers.pth")
  torch.save(lambda_multipliers, outpath)

  return outpath


# From: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model) -> int:
  """Count the parameters of a model."""
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_best_system_device() -> str:
  """Detect the best available device on this machine."""
  device = "cpu"
  if torch.cuda.is_available():
    device = "cuda"
  elif torch.backends.mps.is_available():
    device = "mps"
  return device