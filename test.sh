
set -xe # exit if a test fails
python ddp_main.py --test --env LearnedPendulum-v0
python ddp_main.py --test --env DDP-Pendulum-v0
python quad_main.py --test
python experiment_single_embed/train.py --verbose --baseline --total_steps 2
python experiment_single_embed/train.py --verbose --total_steps 2
python experiment_single_embed/train.py --verbose --structure --total_steps 2
python experiment_single_embed/train.py --verbose --naive --total_steps 2
python scripts/run_learned_pendulum.py
python analyze-single-embed.py --test
set +x
GREEN='\033[0;32m'
printf "${GREEN}Completed all code tests successfully.\n"
