# Test run the whole thing:
Run `bash test.sh` in the command line.

# Code setup:
`ddp` containts the ddp algorithm and functions.

`environments` contain the custom `gym` environments. If you add one, you need to update the `environments/__init__.py`

Each folder might have its own `params`, which should have a specialized name. Please don't use `import params` or 
`import blank as params` as we have some unlimited imports (`from x import *`). In general, try not to have the same name as other files.

## Training new models:
A simple fully connected model:
`python experiment-single-force/train.py --verbose --baseline --total_steps 100`

Learn hamiltonian without symplectic structure:
`python experiment-single-force/train.py --verbose --total_steps 50`

Take advantage of hamiltonian structure:
`python experiment-single-force/train.py --verbose --structure --total_steps 50`

You can also just run the training script, which will only train naive baseline and the structured symplectic models:
`./train_pendulum.sh`

## Adding a dependency:
You should be using Conda. After installing your new dependency using conda or pip, run at the top level:

`conda export env > environment.yml`
OR
`conda env update --prefix ./env --file environment.yml  --prune`

and commit the changes. TODO(Rebecca): find out which of these makes more sense to use.

## Installation:

1. Install conda https://docs.conda.io/en/latest/miniconda.html
2. In a terminal, run:
```
conda env create -f environment.yml -n sympodenet
conda activate sympodenet
```
3. To test that everything works, run
`./test.sh`
