# Environment

This repository contains paper-specific code built on top of OmniDrones. It does not vendor
OmniDrones, Isaac Sim, or NVIDIA Omniverse packages.

## Tested dependency stack

Use the dependency stack documented by OmniDrones:

- Ubuntu/Linux
- Python 3.10
- Isaac Sim 4.1.0
- OmniDrones from <https://github.com/btx0424/OmniDrones>
- TorchRL 0.3.1, as required by OmniDrones

## Setup

Install OmniDrones first:

```bash
git clone https://github.com/btx0424/OmniDrones.git
cd OmniDrones
pip install -e .
```

Then place or keep this paper-code repository in an environment where `omni_drones` is importable.
Avoid reinstalling PyTorch blindly inside an Isaac Sim Python environment; Isaac Sim usually
constrains the compatible PyTorch/CUDA stack.

## Runtime notes

- Training and evaluation require Isaac Sim/Omniverse runtime access.
- Checkpoints are not included in this repository.
- Set checkpoint paths through the Hydra config or command-line overrides.
