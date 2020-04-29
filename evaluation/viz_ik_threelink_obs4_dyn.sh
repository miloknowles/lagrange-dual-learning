python visualize_ik_solutions.py \
  --model_name ik_threelink_4obs_dynamic_viz \
  --config_file ./resources/cfg_joint_limits_and_4obs_dynamic.json \
  --num_links 3 \
  --multiplier_lr 1e-4 \
  --optimizer_lr 5e-5 \
  --initial_lambda 20 \
  --lagrange_iters 1000 \
  --train_iters 500 \
  --batch_size 8 \
  --train_dataset_size 40000 \
  --val_dataset_size 4000 \
  --model_save_hz 50 \
  --hidden_units 100 \
  --cache_save_path /home/milo/lagrange-dual-learning/resources/datasets/ \
  --load_weights_folder /home/milo/lagrange-dual-learning/resources/pretrained_models/ik_threelink_obs4_dynamic_base/weights_200 \
  --no_plot \
  # --show_groundtruth_theta \
