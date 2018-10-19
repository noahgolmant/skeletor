# <Project Name> 

A brief description of the purpose of this repository.

## Setup

This setup requires a linux x64 box.
Necessary packages are listed in `setup.py`.

1. Fill out the basic information required in the `.env`.
2. Fill out the setup info by running `scripts/setup.sh`
3. Create a conda environment for this project:
    a. `conda create -y -n <proj-name> python=3.5`
    b. `conda activate <proj-name>`
4. Install any machine-specific dependencies
    a. `./scripts/install.sh`
5. Install the requirements and mark this as a local package.
    a. `pip install --no-cache-dir --editable .`

## Scripts

All scripts are available in `scripts/`, and should be run from the repo root.

| script | purpose |
| ------ | ------- |
| `lint.sh` | invokes `pylint` with approppriate flags for repo |
| `tests.sh` | runs all tests |
| `install.sh` | use python and cuda info to install packages (e.g. pytorch) |



