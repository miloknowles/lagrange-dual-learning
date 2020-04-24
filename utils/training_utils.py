import os
import torch


def save_model(model, adam, folder, epoch):
  os.makedirs(os.path.join(folder, "weights_{}".format(epoch)), exist_ok=True)

  model_save_path = os.path.join(folder, "weights_{}".format(epoch), "model.pth")
  torch.save(model.state_dict(), model_save_path)

  if adam is not None:
    adam_save_path = os.path.join(folder, "weights_{}".format(epoch), "adam.pth")
    torch.save(adam.state_dict(), adam_save_path)

  return model_save_path, None if adam is None else adam_save_path


def load_model(model, adam, model_path, adam_path):
  model_dict = model.state_dict()
  saved_model_dict = torch.load(model_path)
  saved_model_dict = {k: v for k, v in saved_model_dict.items() if k in model_dict}
  model_dict.update(saved_model_dict)

  if adam is not None:
    adam_dict = adam.state_dict()
    saved_adam_dict = torch.load(adam_path)
    adam_dict.update(saved_adam_dict)

  return model, adam


# From: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/8
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)
