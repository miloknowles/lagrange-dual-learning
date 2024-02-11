import sys; sys.path.append(".."); sys.path.append("../../")
import argparse
from utils.paths import top_folder

class Options(object):
  def __init__(self):
    self.parser = argparse.ArgumentParser(description="Options for the IkLagrangeDualTrainer")

    self.parser.add_argument("--model_name", type=str, help="The name of this experiment/model")
    self.parser.add_argument("--log_dir", type=str, default=top_folder("training_logs"),
                             help="Where to save models and tensorboard events")
    self.parser.add_argument("--lagrange_iters", type=int, default=1000, help="Number of lagrange dual epochs")
    self.parser.add_argument("--train_iters", type=int, default=500, help="Number of training iterations for each relaxation")
    self.parser.add_argument("--batch_size", type=int, default=8, help="Use a small batch size i.e 8")
    self.parser.add_argument("--train_dataset_size", type=int, default=32000, help="Number of training examples to generate")
    self.parser.add_argument("--val_dataset_size", type=int, default=4000, help="Number of validation examples to generate")
    self.parser.add_argument("--multiplier_lr", type=float, default=1e-4, help="Update LR for the multipliers")
    self.parser.add_argument("--optimizer_lr", type=float, default=1e-4, help="Learning rate for Adam")
    self.parser.add_argument("--initial_lambda", type=float, default=40, help="Initial value for the Lagrange multipliers")
    self.parser.add_argument("--num_links", type=int, default=3, choices=[3, 8], help="Number of links in the robotic arm")
    self.parser.add_argument("--config_file", type=str, help="Path to the configuration file that specifies constraints")
    self.parser.add_argument("--num_workers", type=int, default=4, help="Number of data workers to fetch training samples")
    self.parser.add_argument("--model_save_hz", type=int, default=200, help="Save the model after this many epochs")
    self.parser.add_argument("--load_weights_folder", type=str, default=None, help="Path containing a model.pth")
    self.parser.add_argument("--load_adam", action="store_true", default=False, help="Should the adam state be loaded?")
    self.parser.add_argument("--hidden_units", type=int, default=40, help="The number of hidden units for each network layer")
    self.parser.add_argument("--cache_save_path", type=str, default=None,
                             help="If given, load/save the dataset to the specified path for fast loading in the future")
    self.parser.add_argument("--penalty_bad_slope", type=float, default=1.0, help="Slope for piecewise linear constraint penalties.")
    self.parser.add_argument("--penalty_good_slope", type=float, default=0.1, help="Slope for piecewise linear constraint penalties.")

    # Visualization / evaluation arguments.
    self.parser.add_argument("--show_groundtruth_theta", action="store_true", default=False,
                             help="If True, visualize the joint angles from the dataset instead of the network prediction")
    self.parser.add_argument("--no_plot", action="store_true", default=False, help="If True, skipping plotting during evaluation")

  def parse(self):
    self.options = self.parser.parse_args()
    return self.options

  def parse_default(self):
    self.options = self.parser.parse_args(
      ["--model_name", "default",
       "--config_file", "/home/milo/lagrange-dual-learning/resources/cfg_joint_limits_and_4obs_static.json",
       "--train_dataset_size", "1000",
       "--val_dataset_size", "100",
       "--train_iters", "10"])
    return self.options
