# Clean up any previous training sessions under the same name.
../cleanup.sh ik_threelink_joint_limits_base_01

# Now run the training pipeline.
python ../ik_trainer_main.py \
  --model_name ik_threelink_joint_limits_base_01 \
  --config_file ./resources/cfg_joint_limits.json \
  --num_links 3 \
  --multiplier_lr 1e-4 \
  --optimizer_lr 5e-5 \
  --initial_lambda 20 \
  --lagrange_iters 1000 \
  --train_iters 500 \
  --batch_size 8 \
  --train_dataset_size 44000 \
  --val_dataset_size 4000 \
  --model_save_hz 50 \
  --hidden_units 100 \
  --load_weights_folder /home/milo/lagrange-dual-learning/resources/pretrained_models/ik_threelink_base/weights_200 \
