python visualize_ik_solutions.py \
  --model_name ik_threelink_obs1_viz \
  --config_file ./resources/cfg_joint_limits_and_1obs_static.json \
  --num_links 3 \
  --batch_size 1 \
  --train_dataset_size 40000 \
  --val_dataset_size 4000 \
  --hidden_units 100 \
  --cache_save_path /home/milo/lagrange-dual-learning/resources/datasets/ \
  --no_plot \
  --load_weights_folder /home/milo/lagrange-dual-learning/training_logs/ik_threelink_obs1_circ_02/models/weights_950/
  # --load_weights_folder /home/milo/lagrange-dual-learning/training_logs/ik_threelink_obs1_circ_01/models/weights_850/ \
