import sys; sys.path.append(".."); sys.path.append("../../")
from scripts.options import Options
from scripts.trainer import IkLagrangeDualTrainer


if __name__ == "__main__":
  opt = Options()
  trainer = IkLagrangeDualTrainer(opt.parse())
  trainer.main()
