# DDPM Diffusion Model

Implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on MNIST for the DALLE-EM project.

## Quick Start

### Train
```bash
python training_code.py
```

### Generate Samples
```bash
python sampling_code.py
```

## Files
- `training_code.py` — Training script (Cell 3 from Colab)
- `sampling_code.py` — Sampling/inference script (Cell 4 from Colab)
- `DDPM_README.md` — Detailed documentation
- `requirements.txt` — Dependencies

## Installation
```bash
pip install -r requirements.txt
```

## Results
- Trained on 60k MNIST digits for 50 epochs
- Final loss: 0.0151
- Generates realistic handwritten digits from pure noise

## DALLE-EM Connection
This DDPM implementation proves the concept for EM design automation:
- Instead of diffusing images, diffuse EM design parameters
- Train on real antenna/circuit designs + performance specs
- Sample to generate new designs that meet target specifications
