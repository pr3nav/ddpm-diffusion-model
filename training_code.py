import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Downloading MNIST...")
train_dataset = datasets.MNIST(
    root='./mnist_data',
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2
)

print(f"Dataset loaded: {len(train_dataset)} images")
print(f"Batches per epoch: {len(train_loader)}")

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

print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(train_loader) * 50,
)

os.makedirs("./ddpm_output", exist_ok=True)
print("Starting training (50 epochs)")

for epoch in range(50):
    model.train()
    total_loss = 0.0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50")

    for step, (clean_images, _) in enumerate(progress_bar):
        clean_images = clean_images.to(device)

        batch_size_actual = clean_images.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (batch_size_actual,), device=device
        ).long()

        noise = torch.randn_like(clean_images)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    if (epoch + 1) % 5 == 0:
        checkpoint_path = f"./ddpm_output/model_epoch_{epoch+1}.pt"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: model_epoch_{epoch+1}.pt")

final_model_path = "./ddpm_output/model_final.pt"
torch.save(model.state_dict(), final_model_path)
print(f"Training complete: Model saved to {final_model_path}")
