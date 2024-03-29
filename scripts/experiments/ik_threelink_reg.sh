../cleanup.sh ik_threelink_reg

python ../train.py \
  --model_name ik_threelink_reg \
  --config_file ./resources/cfg_joint_limits_and_4obs_static.json \
  --num_links 3 \
  --multiplier_lr 1e-4 \
  --optimizer_lr 1e-4 \
  --initial_lambda 40 \
  --lagrange_iters 1000 \
  --train_iters 500 \
  --batch_size 8 \
  --train_dataset_size 40000 \
  --val_dataset_size 4000 \
  --model_save_hz 2 \
