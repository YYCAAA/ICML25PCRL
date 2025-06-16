# PCRL_PPO Modular Project

A modular implementation of Preference-Conditioned Reinforcement Learning with PPO for Multi-Objective RL.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m main.run/run_Reacher --seed 1 --r 6 --m PreCo
```

## Project Structure

- `agent/` : Agent, models, buffer, min-norm solver
- `utils/` : Config, training, testing, plotting, helpers, seeding, env setup
- `main/`  : Entry point script

## Requirements
See `requirements.txt`.

## Notes
- Make sure your Python path includes the project root, or run with `python -m main.run ...` from the project root.
- The environment name and reward dimension can be changed in `utils/config.py` or via command line. 