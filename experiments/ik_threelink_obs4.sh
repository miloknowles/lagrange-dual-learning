python ../ik_trainer_main.py \
  --model_name ik_threelink_obs4_circ_01 \
  --config_file ./resources/cfg_joint_limits_and_4obs_static.json \
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
  --penalty_good_slope 0.005 \
  --penalty_bad_slope 1.0 \
  --cache_save_path /home/milo/lagrange-dual-learning/resources/datasets/ \
  # --load_weights_folder /home/milo/lagrange-dual-learning/training_logs/ik_threelink_joint_limits_base_03/models/weights_200/ \
