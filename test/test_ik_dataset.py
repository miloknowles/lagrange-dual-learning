import sys
import shutil
import json
import unittest
import os

sys.path.append("/home/milo/lagrange-dual-learning")
from scripts.trainer import IkLagrangeDualTrainer
from kinematics.dataset import IkDataset
from scripts.options import Options


class IkDatasetTest(unittest.TestCase):
  def test_init_no_cache_no_save(self):
    with open("/home/milo/lagrange-dual-learning/resources/cfg_joint_limits_and_4obs_dynamic.json", "r") as f:
      json_config = json.load(f)
    dataset = IkDataset(100, 3, json_config, seed=0, cache_save_path=None)

  def test_init_no_cache_save(self):
    cache_save_path = "/home/milo/lagrange-dual-learning/resources/datasets/ik_dataset_test.pt"
    if os.path.exists(cache_save_path):
      os.remove(cache_save_path)
    with open("/home/milo/lagrange-dual-learning/resources/cfg_joint_limits_and_4obs_dynamic.json", "r") as f:
      json_config = json.load(f)
    dataset = IkDataset(100, 3, json_config, seed=0, cache_save_path=cache_save_path)
    self.assertTrue(os.path.exists(cache_save_path))

    dataset = IkDataset(100, 3, json_config, seed=0, cache_save_path=cache_save_path)
    for i, inputs in enumerate(dataset):
      print(inputs["input_tensor"])

  def test_random_loader(self):
    opt = Options().parse_default()
    trainer = IkLagrangeDualTrainer(opt)

    unique_examples = set()

    for epoch in range(10):
      print("Doing epoch {}".format(epoch+1))
      for ti, inputs in enumerate(trainer.train_loader):
        if ti > 10:
          break
        print(ti, inputs["q_ee_desired"][0])
        unique_examples.add(str(inputs["q_ee_desired"]))

    print("Unique batches:", len(unique_examples))

if __name__ == "__main__":
  unittest.main()
