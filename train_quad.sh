#!/bin/bash
echo "Running training for Quadrotor"
set -xe # exit if a test fails
echo "Training the dummy models we don't use"
python experiment_simple_quad/train.py --baseline --total_steps 2
python experiment_simple_quad/train.py --total_steps 2
echo "Training structure..."
python experiment_simple_quad/train.py --verbose --structure --total_steps 600
echo "Training naive..."
python experiment_simple_quad/train.py --verbose --naive --total_steps 600