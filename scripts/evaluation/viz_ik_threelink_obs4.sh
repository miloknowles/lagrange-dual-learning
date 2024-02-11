python visualize_ik_solutions.py \
  --model_name ik_threelink_4obs_viz \
  --config_file ./resources/cfg_joint_limits_and_4obs_static.json \
  --num_links 3 \
  --batch_size 1 \
  --train_dataset_size 40000 \
  --val_dataset_size 4000 \
  --hidden_units 100 \
  --cache_save_path ../../resources/datasets \
  --load_weights_folder /home/milo/lagrange-dual-learning/training_logs/ik_threelink_obs4_circ_02/models/weights_950 \
  # --no_plot \
  # --load_weights_folder /home/milo/lagrange-dual-learning/training_logs/ik_threelink_obs4_base_05/models/weights_975 \
