# DDPM Diffusion Model Implementation

Hi Amin,

I built a working Denoising Diffusion Probabilistic Models (DDPM) implementation to understand how diffusion models work and prove the concept for DALLE-EM.

## What This Does

Trains a U-Net neural network to generate realistic handwritten digits from pure noise by learning the reverse diffusion process.

**Process:**
1. Forward: Add noise to images step-by-step (1000 steps)
2. Train network to predict the noise added at each step
3. Reverse: Start with pure noise, iteratively remove it → realistic digit

## Results

- **Dataset:** 60k MNIST digits
- **Epochs:** 50
- **Loss:** 0.2850 → 0.0151 (excellent convergence)
- **Generated samples:** 16 realistic handwritten digits from pure noise

## Files

- `training_code.py` — Full training script (50 epochs on GPU: ~8-10 min)
- `sampling_code.py` — Generate new samples from trained model (~2-3 min)
- `requirements.txt` — Dependencies
- `DDPM_README.md` — Detailed technical documentation

## Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python training_code.py
```

### Generate Samples
```bash
python sampling_code.py
```

## DALLE-EM Connection

This DDPM implementation proves the core concept for EM design automation:

**Current (MNIST):**
- Diffuse images
- Train to predict noise at each step
- Sample: noise → realistic digit

**DALLE-EM (next step):**
- Diffuse EM design parameters (antenna dimensions, impedance, frequency, etc.)
- Train on real antenna designs + performance specs
- Sample: noise → new antenna design meeting target specs

**Benefits:**
- Fast design generation (seconds instead of hours of simulation)
- Learns from existing designs
- Can constrain to meet specifications

## Architecture

- **Model:** U-Net with attention layers
- **Parameters:** 15.8M
- **Input:** 32×32 grayscale images
- **Noise schedule:** Linear, 1000 timesteps
- **Optimizer:** AdamW with cosine annealing

## Next Steps

1. Train on actual EM design data instead of MNIST
2. Condition on target specs (frequency, impedance, etc.)
3. Compare generated designs vs simulated performance
4. Optimize for real hardware constraints

---

GitHub: https://github.com/pr3nav/ddpm-diffusion-model

Let me know what EM data you want to train on next!
