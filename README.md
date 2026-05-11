# DDPM with MNIST

## Train
```bash
python training_code.py
```

## Generate Samples
```bash
python sampling_code.py
```

## Files
- `training_code.py` — Training script
- `sampling_code.py` — Sampling/inference script
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

## How it Connects to DALLE-EM
This DDPM implementation is proof of concept for EM design automation:
- Instead of diffusing images, diffuse EM design parameters
- Train on antenna/circuit designs and performance specs
- Sample to generate new designs that meet target specs
