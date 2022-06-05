# Implicit Language Q Learning

Official code from the paper "Offline RL for Natural Language Generation with Implicit Language Q Learning"

[project site](https://sea-snell.github.io/ILQL_site/) | [arxiv]()

## Setup

### Preprocessed Data

Download the data zip from the Google drive folder [here](https://drive.google.com/drive/folders/1ltO6e4sP3waGPJoGFGuiHt7mJt8_2eP3?usp=sharing). Place the downloaded and unzipped folder, "data", at the root of the repo.

### Dependencies and PYTHONPATH

This repo was designed for python 3.9.1

``` shell
pip install -r requirements.txt
export PYTHONPATH="$PWD/src/"
```

### Visual Dialogue Environment

To run the Visual Dialogue experiments, you need to serve the Visual Dialogue environment on localhost by following the instructions [here](https://github.com/Sea-Snell/visdial-rl).

*Make sure that the port is set fo 5000 in the config for the Visual Dialogue script you are running.*

## Repo Overview

* All data is provided pre-processed in the `data/` folder.
* `scripts/` contains all scripts for running training, evaluation, and data pre-processing steps in the paper. Scripts are organized into subfolders corresponding to the dataset used.
* `config/` contains .yaml configs for each script. This reop uses [hydra](https://hydra.cc/docs/intro/) to manage configs. Configs are organized into subfolders corresponding to the dataset used. Most config files are named the same as their corresponding script, but if you are unsure which config corresponds to a script, check the line `@hydra.main(config_path="some_path", config_name="some_name")` to see which config file the script corresponds to.
* `src/` contains all the core implementations. See `src/models/` for all model implementations. See `src/data/` for all base data processing and MDP abstraction code. See `src/utils/` for various utility functions. See `src/wordle/`, `src/visdial`, and `src/toxicity/` for all Wordle, Visual Dialogue, and Reddit comment dataset specific code respectively.

## Running Experiments

`scripts/` contains all experiment scripts. To run any script in `scripts/`:
1. Navigate to the script's directory.
2. `python script_name.py`

Optional:
* Edit the config file corresponding to the script as you desire.
* Provide commandline args [hydra](https://hydra.cc/docs/intro/) style like: `python script_name.py eval.bsize=5 train.lr=1e-6 wandb.use_wandb=false`
* Run data parallel training or evaluation on multiple GPUs like: `python -m torch.distributed.launch --nproc_per_node [N_GPUs] --use_env script_name.py arg1=a arg2=b`





