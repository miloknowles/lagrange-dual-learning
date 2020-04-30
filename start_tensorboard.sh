# NOTE: $1 should be a model name, corresponding to a folder found in training_logs/
python /home/milo/.local/lib/python3.7/site-packages/tensorboard/main.py \
  --logdir=/home/milo/lagrange-dual-learning/training_logs/$1 \
  --bind_all --port 6006
