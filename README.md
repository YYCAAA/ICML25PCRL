# PCRL_PreCo

An implementation of Preference-Conditioned Reinforcement Learning with PreCo for Multi-Objective RL.

[Paper Link](https://openreview.net/pdf?id=49g4c8MWHy)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

On-policy implementation
```bash
python -m main.run/run_Reacher --seed 1 --r 6 --m PreCo
```

For off-policy implementation with HER
```bash
cd Offpolicy_with_sample_efficiency_techniques/
python off_policy_preco_reacher.py
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

## 📖 Citation

If you use this work, please cite:

```bibtex
@inproceedings{yang2025preference,
  title     = {Preference Controllable Reinforcement Learning with Advanced Multi-Objective Optimization},
  author    = {Yang, Yucheng and Zhou, Tianyi and Pechenizkiy, Mykola and Fang, Meng},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  url       = {https://openreview.net/forum?id=49g4c8MWHy}
}


