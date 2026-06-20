# PSTO-AAV-Pursuit

This repository contains the code release for **Decentralized End-to-End Multi-AAV Pursuit Using
Predictive Spatio-Temporal Observation via Deep Reinforcement Learning**.

The implementation is based on OmniDrones/Isaac Sim and includes the PSTO observation construction,
the dual-stream policy network, LSTM-based evader trajectory prediction, and MAPPO training with
centralized critic and decentralized actor execution.

## Contents

- `train.py`: MAPPO training entry point.
- `evaluate.py`: checkpoint evaluation entry point.
- `env.py`: Isaac Sim pursuit environment and PSTO observation construction.
- `mappo.py`: MAPPO policy, value update, checkpoint I/O, and LSTM trajectory predictor training.
- `lstm.py`: evader trajectory predictor used by the PSTO heatmap.
- `models/model.py`: dual-stream LiDAR/intent-heatmap encoder and actor feature decoder.
- `nav_utils.py`: distributions, GAE, value normalization, evaluation, and frame transforms.
- `configs/`: training and evaluation configs.

## Training

```bash
python train.py
```

To resume from a checkpoint:

```bash
python train.py checkpoint_path=/path/to/checkpoint.pt resume_optimizer=true
```

## Evaluation

```bash
python evaluate.py eval.checkpoint_path=/path/to/checkpoint.pt
```

The default evaluation config uses 150 parallel environments, matching the paper's reported
evaluation protocol.

## Dependency on OmniDrones

This code is implemented as paper-specific training and evaluation scripts on top of
[OmniDrones](https://github.com/btx0424/OmniDrones). The OmniDrones framework itself is not vendored
in this repository. Install OmniDrones first, then run these scripts from this release directory.

Recommended setup:

```bash
git clone https://github.com/btx0424/OmniDrones.git
cd OmniDrones
pip install -e .
```

OmniDrones currently documents Isaac Sim 4.1.0, Python 3.10, Linux support, and `torchrl==0.3.1` in
its public repository. Please cite OmniDrones together with this paper when using the code.

## Notes

This release keeps the paper-facing training and evaluation code. Experiment logs, W&B run folders,
checkpoint files, and cache files are excluded; checkpoints should be provided explicitly through
Hydra config overrides.

## License

This code is released under the MIT License. See `LICENSE`.

## Acknowledgements

This implementation builds on [OmniDrones](https://github.com/btx0424/OmniDrones). Please cite
OmniDrones together with the corresponding paper when using this code.
