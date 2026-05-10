import torch
from diffusers import DDPMScheduler, UNet2DModel
from torchvision.utils import save_image
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="./ddpm_output/model_final.pt")
parser.add_argument("--num-samples", type=int, default=16)
parser.add_argument("--output-dir", type=str, default="./ddpm_output")
parser.add_argument("--num-inference-steps", type=int, default=1000)
args = parser.parse_args()

print("Loading model...")

model = UNet2DModel(
    sample_size=32,
    in_channels=1,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(32, 64, 128, 256),
    down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    attention_head_dim=8,
).to(device)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

print(f"Loaded model from: {args.model_path}")

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)

print(f"\nGenerating {args.num_samples} samples...")

with torch.no_grad():
    x = torch.randn(args.num_samples, 1, 32, 32).to(device)

    steps_to_use = min(args.num_inference_steps, 1000)
    
    for t in range(999, -1, -1):
        if t % 100 == 0:
            print(f"  Step {1000-t}/1000")
        
        timesteps = torch.full((args.num_samples,), t, device=device).long()
        
        noise_pred = model(x, timesteps, return_dict=False)[0]
        
        x = noise_scheduler.step(noise_pred, t, x, return_dict=False)[0]

x = (x + 1) / 2 
x = torch.clamp(x, 0, 1)

sample_path = os.path.join(args.output_dir, "generated_samples.png")
save_image(x, sample_path, nrow=4)
print(f"Saved: {sample_path}")

for i in range(min(args.num_samples, 5)):
    individual_path = os.path.join(args.output_dir, f"sample_{i}.png")
    save_image(x[i:i+1], individual_path)
    print(f"   {individual_path}")
