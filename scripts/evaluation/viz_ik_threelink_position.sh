python visualize_ik_solutions.py \
  --model_name ik_threelink_position_viz \
  --config_file ./resources/cfg_no_constraints.json \
  --num_links 3 \
  --batch_size 1 \
  --train_dataset_size 40000 \
  --val_dataset_size 4000 \
  --hidden_units 100 \
  --cache_save_path ../../resources/datasets \
  --no_plot \
  --load_weights_folder /home/milo/lagrange-dual-learning/training_logs/ik_threelink_position_01/models/weights_600/
