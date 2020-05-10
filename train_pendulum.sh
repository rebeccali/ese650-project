#!/bin/bash
echo "Running training for Pendulum"
set -xe # exit if a test fails
echo "Training the dummy models we don't use"
python experiment_single_embed/train.py --num_points 5 --baseline --total_steps 2
python experiment_single_embed/train.py --num_points 5 --total_steps 2
echo "Training structure..."
python experiment_single_embed/train.py --num_points 5 --verbose --structure --total_steps 600
echo "Training naive..."
python experiment_single_embed/train.py --num_points 5 --verbose --naive --total_steps 600