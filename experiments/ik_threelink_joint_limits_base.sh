python ../ik_trainer_main.py \
  --model_name ik_threelink_joint_limits_base_03 \
  --config_file ./resources/cfg_joint_limits.json \
  --num_links 3 \
  --multiplier_lr 1e-4 \
  --optimizer_lr 5e-5 \
  --initial_lambda 1 \
  --lagrange_iters 1000 \
  --train_iters 500 \
  --batch_size 8 \
  --train_dataset_size 40000 \
  --val_dataset_size 4000 \
  --model_save_hz 25 \
  --hidden_units 100 \
  --load_weights_folder /home/milo/lagrange-dual-learning/resources/pretrained_models/ik_threelink_base/weights_200 \
  --penalty_good_slope 0.001 \
  --penalty_bad_slope 1.0 \
