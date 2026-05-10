#!/usr/bin/env python3
"""
Simple DDPM (Denoising Diffusion Probabilistic Model) training script.
Trains on MNIST, generates image samples.

Usage:
    python3 ddpm_train.py --epochs 50 --batch-size 128
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Config
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--learning-rate", type=float, default=1e-4)
parser.add_argument("--output-dir", type=str, default="./ddpm_output")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ============ 1. DATA ============
print("Loading MNIST...")
transforms_list = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize to 32x32 for faster training
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.MNIST(
    root="./mnist_data",
    train=True,
    download=True,
    transform=transforms_list
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0
)

print(f"Loaded {len(train_dataset)} training samples")

# ============ 2. MODEL ============
print("Initializing DDPM components...")

# Scheduler: controls noise schedule (how much noise to add at each step)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)

# Model: U-Net that learns to denoise
model = UNet2DModel(
    sample_size=32,  # input image size
    in_channels=1,   # grayscale
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 128, 256),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    attention_head_dim=8,
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

# Learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_loader) * args.epochs,
)

# ============ 3. TRAINING LOOP ============
print(f"Training for {args.epochs} epochs...")

for epoch in range(args.epochs):
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for step, (clean_images, _) in enumerate(progress_bar):
        # Move to device
        clean_images = clean_images.to(device)
        
        # Sample random timesteps
        batch_size = clean_images.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        # Add noise to images (forward diffusion process)
        noise = torch.randn_like(clean_images)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # Predict noise
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        
        # MSE loss between predicted noise and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint every few epochs
    if (epoch + 1) % 5 == 0:
        checkpoint_path = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# Save final model
final_model_path = os.path.join(args.output_dir, "model_final.pt")
torch.save(model.state_dict(), final_model_path)
print(f"Saved final model: {final_model_path}")

# ============ 4. SAMPLING (Inference) ============
print("\nGenerating samples...")

model.eval()
num_samples = 16

with torch.no_grad():
    # Start with pure noise
    x = torch.randn(num_samples, 1, 32, 32).to(device)
    
    # Denoise iteratively (reverse diffusion process)
    for t in tqdm(reversed(range(1000)), total=1000, desc="Sampling"):
        timesteps = torch.full((num_samples,), t, device=device).long()
        
        # Predict noise
        noise_pred = model(x, timesteps, return_dict=False)[0]
        
        # Use scheduler to remove noise
        x = noise_scheduler.step(noise_pred, t, x, return_dict=False)[0]

# Denormalize
x = (x + 1) / 2  # [-1, 1] -> [0, 1]
x = torch.clamp(x, 0, 1)

# Save generated images
from torchvision.utils import save_image
sample_path = os.path.join(args.output_dir, "samples.png")
save_image(x, sample_path, nrow=4)
print(f"Saved generated samples: {sample_path}")

print("\n✅ Done! Check the generated images in:", args.output_dir)
