# Clean up any previous training sessions under the same name.
../cleanup.sh ik_threelink_obs1_base_02

# Now run the training pipeline.
python ../ik_trainer_main.py \
  --model_name ik_threelink_obs1_base_02 \
  --config_file ./resources/cfg_joint_limits_and_1obs_static.json \
  --num_links 3 \
  --multiplier_lr 1e-4 \
  --optimizer_lr 5e-5 \
  --initial_lambda 20 \
  --lagrange_iters 251 \
  --train_iters 251 \
  --batch_size 8 \
  --train_dataset_size 40000 \
  --val_dataset_size 4000 \
  --model_save_hz 50 \
  --hidden_units 100 \
  --load_weights_folder /home/milo/lagrange-dual-learning/resources/pretrained_models/ik_threelink_joint_limits_base/weights_100 \
  --cache_save_path /home/milo/lagrange-dual-learning/resources/datasets/ \
