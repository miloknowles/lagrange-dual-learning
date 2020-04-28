../cleanup.sh ik_threelink_reg

python ../ik_trainer_main.py \
  --model_name ik_threelink_reg \
  --config_file ./resources/cfg_joint_limits.json \
  --num_links 3 \
  --multiplier_lr 1e-4 \
  --optimizer_lr 1e-4 \
  --initial_lambda 40 \
  --lagrange_iters 1000 \
  --train_iters 500 \
  --batch_size 8 \
  --train_dataset_size 44000 \
  --val_dataset_size 4000 \
  --model_save_hz 2 \
