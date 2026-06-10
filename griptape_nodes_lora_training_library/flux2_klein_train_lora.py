#!/usr/bin/env python3
"""FLUX.2 Klein LoRA training script using diffusers + PEFT."""
import argparse
import gc
import os
from pathlib import Path

import torch
from diffusers import Flux2KleinPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from PIL import Image
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

MODEL_CONFIGS = {
    "black-forest-labs/FLUX.2-klein-base-4B": {
        "guidance_scale": 4.0,
    },
    "black-forest-labs/FLUX.2-klein-4B": {
        "guidance_scale": 1.0,
    },
}


class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir, resolution=512, num_repeats=1):
        self.image_dir = Path(image_dir)
        self.resolution = resolution
        self.num_repeats = num_repeats
        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        self.image_files = [f for f in self.image_dir.iterdir() if f.suffix.lower() in self.image_extensions]
        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_files) * self.num_repeats

    def __getitem__(self, idx):
        img_idx = idx % len(self.image_files)
        img_path = self.image_files[img_idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        caption_path = img_path.with_suffix(".txt")
        caption = caption_path.read_text().strip() if caption_path.exists() else ""
        return {"pixel_values": image, "caption": caption}


def parse_args():
    parser = argparse.ArgumentParser(description="FLUX.2 Klein LoRA Training")
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_name", default="flux2_klein_lora")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--network_dim", type=int, default=16)
    parser.add_argument("--network_alpha", type=int, default=16)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "no"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    repo_id = args.pretrained_model_name_or_path
    config = MODEL_CONFIGS.get(repo_id, MODEL_CONFIGS["black-forest-labs/FLUX.2-klein-base-4B"])

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading pipeline from {repo_id}...")
    pipe = Flux2KleinPipeline.from_pretrained(
        repo_id, torch_dtype=dtype, device_map="balanced"
    )

    print("Pre-encoding captions...")
    dataset = ImageCaptionDataset(args.dataset_path, resolution=args.resolution, num_repeats=args.num_repeats)

    caption_cache = {}
    for i in range(len(dataset.image_files)):
        img_path = dataset.image_files[i]
        caption_path = img_path.with_suffix(".txt")
        caption = caption_path.read_text().strip() if caption_path.exists() else ""
        if caption not in caption_cache:
            with torch.no_grad():
                prompt_embeds, text_ids = pipe.encode_prompt(prompt=caption, device=device, num_images_per_prompt=1)
                caption_cache[caption] = (prompt_embeds.cpu(), text_ids.cpu())

    pipe.text_encoder.to("cpu")
    del pipe.text_encoder
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    print(f"Cached {len(caption_cache)} unique captions")

    vae = pipe.vae.to(device)
    transformer = pipe.transformer.to(device)
    transformer.enable_gradient_checkpointing()

    print("Applying LoRA to transformer...")
    lora_config = LoraConfig(
        r=args.network_dim,
        lora_alpha=args.network_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    vae.requires_grad_(False)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(100, args.max_train_steps // 10),
        num_training_steps=args.max_train_steps,
    )

    def convert_peft_to_diffusers(state_dict):
        """Convert PEFT LoRA keys to diffusers-compatible format."""
        converted = {}
        for k, v in state_dict.items():
            new_key = k.replace("base_model.model.", "transformer.")
            new_key = new_key.replace(".lora_A.default.weight", ".lora_A.weight")
            new_key = new_key.replace(".lora_B.default.weight", ".lora_B.weight")
            converted[new_key] = v
        return converted

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    progress_bar = tqdm(total=args.max_train_steps, desc="Training")
    transformer.train()

    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            captions = batch["caption"]
            caption = str(captions[0]) if isinstance(captions, (list, tuple)) else str(captions)

            with torch.no_grad():
                latent_dist = vae.encode(pixel_values).latent_dist
                latents = latent_dist.sample()

                prompt_embeds, text_ids = caption_cache[caption]
                prompt_embeds = prompt_embeds.to(device)
                text_ids = text_ids.to(device)

            bsz, c, h, w = latents.shape

            # FLUX.2 Klein 2x2 patchification
            latents_packed = latents.view(bsz, c, h // 2, 2, w // 2, 2)
            latents_packed = latents_packed.permute(0, 1, 3, 5, 2, 4)
            latents_packed = latents_packed.reshape(bsz, c * 4, h // 2, w // 2)

            # Batch normalization operates on packed latents (c*4=128 channels)
            with torch.no_grad():
                bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents_packed.device, latents_packed.dtype)
                bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
                    latents_packed.device, latents_packed.dtype
                )
                latents_packed = (latents_packed - bn_mean) / bn_std

            latents_flat = latents_packed.permute(0, 2, 3, 1).reshape(bsz, (h // 2) * (w // 2), c * 4)

            # U-shaped timestep distribution
            u = torch.rand(bsz, device=device, dtype=dtype)
            a = 4.0
            timesteps = (torch.exp(a * u) - 1) / (torch.exp(torch.tensor(a, device=device)) - 1)
            timesteps = torch.where(torch.rand(bsz, device=device) < 0.5, timesteps, 1 - timesteps)
            timesteps = timesteps.clamp(0.001, 0.999)

            noise = torch.randn_like(latents_flat)
            noisy_latents = (1 - timesteps.view(-1, 1, 1)) * latents_flat + timesteps.view(-1, 1, 1) * noise

            t_ids = torch.zeros(1, device=device, dtype=torch.long)
            h_ids = torch.arange(h // 2, device=device, dtype=torch.long)
            w_ids = torch.arange(w // 2, device=device, dtype=torch.long)
            l_ids = torch.zeros(1, device=device, dtype=torch.long)
            img_ids = torch.stack(torch.meshgrid(t_ids, h_ids, w_ids, l_ids, indexing="ij"), dim=-1)
            img_ids = img_ids.reshape(1, -1, 4).expand(bsz, -1, -1).to(dtype)

            guidance = torch.full((bsz,), config["guidance_scale"], device=device, dtype=dtype)

            model_pred = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                img_ids=img_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

            target = noise - latents_flat
            loss = torch.nn.functional.mse_loss(model_pred, target, reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

            if args.save_every_n_steps > 0 and global_step % args.save_every_n_steps == 0:
                save_path = Path(args.output_dir) / f"{args.output_name}-step{global_step}.safetensors"
                ckpt_state_dict = {k: v for k, v in transformer.state_dict().items() if "lora" in k.lower()}
                ckpt_state_dict = convert_peft_to_diffusers(ckpt_state_dict)
                save_file(ckpt_state_dict, str(save_path))
                print(f"\nCheckpoint: {save_path}")

    progress_bar.close()

    final_path = Path(args.output_dir) / f"{args.output_name}.safetensors"
    lora_state_dict = {k: v for k, v in transformer.state_dict().items() if "lora" in k.lower()}
    lora_state_dict = convert_peft_to_diffusers(lora_state_dict)
    metadata = {
        "model_version": repo_id,
        "network_dim": str(args.network_dim),
        "network_alpha": str(args.network_alpha),
        "resolution": str(args.resolution),
    }
    save_file(lora_state_dict, str(final_path), metadata=metadata)
    print(f"\nTraining complete! LoRA saved to {final_path}")


if __name__ == "__main__":
    main()
