python visualize_ik_solutions.py \
  --model_name ik_threelink_joint_limits_viz \
  --config_file ./resources/cfg_joint_limits.json \
  --num_links 3 \
  --multiplier_lr 1e-4 \
  --optimizer_lr 5e-5 \
  --initial_lambda 1 \
  --lagrange_iters 1000 \
  --train_iters 500 \
  --batch_size 1 \
  --train_dataset_size 40000 \
  --val_dataset_size 4000 \
  --model_save_hz 50 \
  --hidden_units 100 \
  --load_weights_folder /home/milo/lagrange-dual-learning/training_logs/ik_threelink_joint_limits_base_03/models/weights_200/ \
  --no_plot \
  # --cache_save_path /home/milo/lagrange-dual-learning/resources/datasets/ \
# 0.1 ==> /home/milo/lagrange-dual-learning/training_logs/ik_threelink_joint_limits_base_01/models/weights_50/ \
# 0.01 ==> /home/milo/lagrange-dual-learning/training_logs/ik_threelink_joint_limits_base_02/models/weights_150/ \
# --load_weights_folder /home/milo/lagrange-dual-learning/resources/pretrained_models/ik_threelink_joint_limits_base/weights_200 \
