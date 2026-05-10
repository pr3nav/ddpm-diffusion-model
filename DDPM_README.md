# DDPM Diffusion Model - DALLE-EM Project

A from-scratch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** trained on MNIST handwritten digits.

## 📋 What This Does

1. **Trains** a U-Net neural network to learn the reverse of the diffusion process (denoising)
2. **Generates** new MNIST-like digits by starting from pure noise and iteratively denoising
3. **Demonstrates** the core technique behind DALLE-EM: using diffusion models to generate new designs

## 🚀 Quick Start

### 1. Train

```bash
python3 ddpm_train.py --epochs 50 --batch-size 128
```

**Options:**
- `--epochs`: Number of training epochs (default 10)
- `--batch-size`: Batch size (default 128)
- `--learning-rate`: Learning rate (default 1e-4)
- `--output-dir`: Where to save checkpoints (default `./ddpm_output`)

**Expected output:**
- Loss decreases (1.0+ → 0.3-0.5 after 50 epochs)
- Model checkpoints saved every 5 epochs
- Final model saved as `model_final.pt`

### 2. Sample (Generate New Images)

```bash
python3 ddpm_sample.py --model-path ./ddpm_output/model_final.pt --num-samples 16
```

**Output:**
- `generated_samples.png` — 4x4 grid of generated digits
- `sample_0.png`, `sample_1.png`, etc. — Individual samples

## 🧠 How DDPM Works

### Forward Process (Training)
1. Take a real image
2. Add random noise step-by-step (1000 steps)
3. Train network to **predict the noise** added at each step

### Reverse Process (Sampling)
1. Start with pure random noise
2. For each step: feed to network → it predicts noise → remove it
3. After 1000 denoising steps → realistic new image

### Why It Works
- The network learns a smooth mapping: `noisy_image → less_noisy_image`
- By applying this 1000 times, you get: `random_noise → realistic_image`

## 📊 Architecture

- **Model:** U-Net with attention layers
- **Parameters:** 15.8 million
- **Input:** 32x32 grayscale images
- **Training data:** 60,000 MNIST digits
- **Device:** Auto-detects CUDA, falls back to CPU

## 🔗 Connection to DALLE-EM

**DALLE-EM (Amin's paper):**
- Instead of images, diffuse **EM design parameters** (antenna dimensions, impedance, etc.)
- Training data: real antenna designs + their performance specs
- Sampling: "Generate antenna designs that achieve 5GHz resonance + 50Ω impedance"
- **Benefit:** Fast design generation instead of hours of simulation

**This MNIST demo proves:**
1. You understand the diffusion process
2. You can implement DDPM from scratch (mostly via diffusers library)
3. You can adapt it to new domains (images → EM parameters)

## 📁 Files

- `ddpm_train.py` — Main training script
- `ddpm_sample.py` — Inference/sampling script
- `ddpm_output/` — Checkpoint directory
  - `model_epoch_5.pt`, `model_epoch_10.pt`, ... — Checkpoints
  - `model_final.pt` — Final trained model
  - `samples.png` — Samples generated during training
  - `generated_samples.png` — Samples from sampling script

## ⚡ Performance Notes

### CPU vs GPU
- **CPU (MacBook):** ~2-2.5s per batch, ~1 hour for 50 epochs
- **GPU:** ~0.2s per batch, ~6 minutes for 50 epochs

### Sampling Speed
- **1000 steps:** ~5-10 minutes on CPU, ~10 seconds on GPU
- **100 steps (faster, lower quality):** ~30 seconds on CPU
  ```bash
  python3 ddpm_sample.py --num-inference-steps 100
  ```

## 🎯 Next Steps (For Amin)

1. Train for 50+ epochs (loss should reach 0.3-0.5)
2. Generate samples and visually inspect quality
3. Write brief explanation:
   - How diffusion models work (2-3 sentences)
   - How DALLE-EM adapts this to EM design
   - What you implemented and why it works
4. Create GitHub repo, push code
5. Email Amin with:
   - GitHub link
   - Generated sample images
   - 1-paragraph summary
   - Explanation of next steps (training on actual EM data)

## 📚 Further Reading

- **DDPM Paper:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- **DALLE-EM Paper:** (Amin's work on EM design with diffusion)
- **Diffusers Library:** [Hugging Face Diffusers](https://huggingface.co/docs/diffusers)

## 🐛 Troubleshooting

**Q: Training is very slow**
- A: You're on CPU. This is normal (2.5s/batch). Consider:
  - Using smaller model: `--batch-size 32`
  - Fewer epochs: `--epochs 10`
  - Or use Google Colab with GPU

**Q: CUDA not found / GPU not detected**
- A: Code auto-falls back to CPU. To use GPU:
  - Install PyTorch with CUDA support: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

**Q: Generated samples look like noise**
- A: Model needs more training. Train for 50+ epochs and check loss → should be < 0.5

**Q: Out of memory**
- A: Reduce `--batch-size` (e.g., to 32 or 64)

---

Built for DALLE-EM project with Amin. Have fun! 🚀
