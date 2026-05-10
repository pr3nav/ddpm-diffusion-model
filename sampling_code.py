import torch
from diffusers import DDPMScheduler, UNet2DModel
from torchvision.utils import save_image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear"
)

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

model.load_state_dict(torch.load("./ddpm_output/model_final.pt"))
model.eval()

print("Generating 16 samples...")

with torch.no_grad():
    x = torch.randn(16, 1, 32, 32).to(device)

    for t in tqdm(reversed(range(1000)), total=1000, desc="Sampling"):
        timesteps = torch.full((16,), t, device=device).long()
        noise_pred = model(x, timesteps, return_dict=False)[0]
        x = noise_scheduler.step(noise_pred, t, x, return_dict=False)[0]

x = (x + 1) / 2
x = torch.clamp(x, 0, 1)

save_image(x, "./ddpm_output/generated_samples.png", nrow=4)
print("Samples generated and saved!")
