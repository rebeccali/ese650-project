#!/bin/bash
set -xe # exit if a test fails
python experiment_single_embed/train.py --test --baseline --total_steps 2
python experiment_single_embed/train.py --test --total_steps 2
python experiment_single_embed/train.py --test --structure --total_steps 2
python experiment_single_embed/train.py --test --naive --total_steps 2
python ddp_main.py --test --env LearnedPendulum-v0
python ddp_main.py --test --env DDP-Pendulum-v0
python quad_main.py --test
python analyze-single-embed.py --test
python scripts/verify_pendulum_dynamics.py --test --horizon 0.2
set +x
GREEN='\033[0;32m'
printf "${GREEN}Completed all code tests successfully.\n"
