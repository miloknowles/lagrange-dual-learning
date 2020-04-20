from ik_options import IkOptions
from ik_trainer import IkLagrangeDualTrainer


if __name__ == "__main__":
  opt = IkOptions()
  trainer = IkLagrangeDualTrainer(opt.parse())
  trainer.main()
