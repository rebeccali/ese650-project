set -e # exit if a test fails
python ddp_main.py --test
python quad_main.py --test
python experiment_single_embed/train.py --verbose --baseline --total_steps 10
python experiment_single_embed/train.py --verbose --total_steps 10
python experiment_single_embed/train.py --verbose --structure --total_steps 10
python experiment_single_embed/train.py --verbose --naive --total_steps 10
python scripts/run_learned_pendulum.py
