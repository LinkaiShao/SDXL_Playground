#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paired LoRA training on SDXL with FROZEN IP-Adapter conditioning.

- Target: learn a "studio catalog" style while preserving garment identity
- Conditioning: text (catalog caption) + IP-Adapter image features from scraggly photo(s)
- Trainable: UNet LoRA (optionally TE LoRA if you flip --no_unet_only)
- Diffusers: 0.36.0.dev0 (expects load_ip_adapter + prepare_ip_adapter_image_embeds)
"""

import os, re, argparse, itertools, warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from datasets import Dataset
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from diffusers import (
    StableDiffusionXLPipeline,
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers

from peft import LoraConfig, get_peft_model, get_peft_model_state_dict


# ----------------------------
# Args
# ----------------------------
@dataclass
class Args:
    pretrained_model: str = "/home/link/Desktop/Code/fashion gen testing/sdxl-base"
    data_root: str = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data"
    output_dir: str = "./sdxl_lora_out_ip"

    # Which scraggly sources to use as guidance
    use_ground: bool = True
    use_hanged: bool = True

    resolution: int = 1024
    center_crop: bool = True
    random_flip: bool = False

    # LoRA
    rank: int = 16
    alpha: int = 16
    unet_only: bool = True
    init_lora_weights: str = "gaussian"

    # Optim
    lr: float = 5e-5          # lower LR for stability with small dataset
    weight_decay: float = 0.01  # small weight decay for regularization
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    noise_offset: float = 0.0

    # Training
    train_batch_size: int = 1
    grad_accum: int = 4
    max_train_steps: int = 1500
    mixed_precision: str = "bf16"     # or "fp16"
    gradient_checkpointing: bool = True
    seed: int = 42

    # Scheduler
    lr_scheduler: str = "cosine"
    warmup_ratio: float = 0.1         # 10% warmup

    # Loss
    use_min_snr: bool = True
    min_snr_gamma: float = 5.0

    # IP-Adapter
    ip_repo: str = "h94/IP-Adapter"
    ip_subfolder: str = "sdxl_models"
    ip_weight_name: str = "ip-adapter_sdxl.bin"
    ip_scales: str = "0.6,0.6"        # one scale per guidance image (ground, hanged)

    # Saving/Logging
    checkpointing_steps: int = 500
    log_every: int = 10

    # Debug cap
    max_train_samples: Optional[int] = None


def parse_args() -> Args:
    p = argparse.ArgumentParser()
    for k, v in Args().__dict__.items():
        t = type(v)
        if t is tuple: t = float
        if isinstance(v, bool):
            if v:
                p.add_argument(f"--no_{k}", dest=k, action="store_false", default=v)
            else:
                p.add_argument(f"--{k}", dest=k, action="store_true", default=v)
        else:
            p.add_argument(f"--{k}", type=t, default=v)
    a = p.parse_args()
    return a


# ----------------------------
# Data utils
# ----------------------------
def _extract_num(path: str) -> Optional[int]:
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None


def load_paired_list(data_root: str, use_ground: bool, use_hanged: bool, max_samples: Optional[int]=None):
    ground_dir = os.path.join(data_root, "on_ground_white_bg")
    hanged_dir = os.path.join(data_root, "hanged_white_bg")
    target_dir = os.path.join(data_root, "straightened")

    def list_imgs(d):
        if not os.path.isdir(d): return {}
        exts = (".jpg", ".jpeg", ".png", ".webp")
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(exts)]
        return { _extract_num(f): f for f in files }

    g = list_imgs(ground_dir) if use_ground else {}
    h = list_imgs(hanged_dir) if use_hanged else {}
    t = list_imgs(target_dir)

    keys = set(t.keys())
    if use_ground: keys &= set(g.keys())
    if use_hanged: keys &= set(h.keys())
    keys = sorted([k for k in keys if k is not None])

    items = []
    for k in keys:
        guidance_paths = []
        if use_ground: guidance_paths.append(g[k])
        if use_hanged: guidance_paths.append(h[k])
        items.append({"guidance_paths": guidance_paths, "target_path": t[k], "num": k})

    if max_samples:
        items = items[:max_samples]
    return items


def make_transforms(resolution: int, center_crop: bool, random_flip: bool):
    tfm = [
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
    ]
    if random_flip:
        tfm.append(transforms.RandomHorizontalFlip())
    tfm += [
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
    return transforms.Compose(tfm)


# ----------------------------
# IP-Adapter Encoding Helper
# ----------------------------
def encode_ip_embeds_from_pils(pipe, batch_of_lists_of_pils, device):
    """
    batch_of_lists_of_pils: List[List[PIL.Image]]   # e.g., [[ground, hanged], [ground], ...]
    Returns: torch.Tensor [B, D] in fp32 (averaged per sample)
    """
    pipe.image_encoder.eval()
    embeds_per_sample = []
    with torch.no_grad():
        for pil_list in batch_of_lists_of_pils:
            sample_embeds = []
            for img in pil_list:
                # Feature-extract → encode
                fe = pipe.feature_extractor(images=img, return_tensors="pt").pixel_values.to(device)
                # image_encoder may output ModelOutput with .image_embeds or a raw tensor
                out = pipe.image_encoder(fe)
                img_embed = getattr(out, "image_embeds", out)
                if isinstance(img_embed, (list, tuple)):
                    img_embed = img_embed[0]
                sample_embeds.append(img_embed)
            # average refs (ground+hanged) → [1, D]
            avg = torch.stack(sample_embeds, dim=0).mean(dim=0)
            # ensure shape [1, D]
            if avg.ndim == 1:
                avg = avg.unsqueeze(0)
            embeds_per_sample.append(avg)
    embeds = torch.cat(embeds_per_sample, dim=0).to(torch.float32)  # [B, D]
    return embeds


# ----------------------------
# Save
# ----------------------------
def save_lora(pipe: StableDiffusionXLPipeline, unet, te1, te2, args: Args, tag: str):
    state: Dict[str, Any] = {}
    unet = Accelerator().unwrap_model(unet)
    unet_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    state.update({f"unet.{k}": v for k, v in unet_sd.items()})

    if te1 is not None and te2 is not None:
        te1 = Accelerator().unwrap_model(te1)
        te2 = Accelerator().unwrap_model(te2)
        te1_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(te1))
        te2_sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(te2))
        state.update({f"text_encoder.{k}": v for k, v in te1_sd.items()})
        state.update({f"text_encoder_2.{k}": v for k, v in te2_sd.items()})

    weight_name = f"lora_sdxl_ip_rank{args.rank}_steps{args.max_train_steps}_{tag}.safetensors"
    pipe.save_lora_weights(
        save_directory=args.output_dir,
        weight_name=weight_name,
        unet_lora_layers={k.replace("unet.", "", 1): v for k, v in state.items() if k.startswith("unet.")},
        text_encoder_lora_layers={k.replace("text_encoder.", "", 1): v for k, v in state.items() if k.startswith("text_encoder.")} if te1 is not None else None,
        text_encoder_2_lora_layers={k.replace("text_encoder_2.", "", 1): v for k, v in state.items() if k.startswith("text_encoder_2.")} if te2 is not None else None,
        safe_serialization=True,
    )
    print(f"[saved] {os.path.join(args.output_dir, weight_name)}")


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    # ip scales
    ip_scales = [float(x) for x in str(args.ip_scales).split(",")]
    if not (args.use_ground or args.use_hanged):
        raise ValueError("At least one of --use_ground or --use_hanged must be True")
    if args.use_ground and args.use_hanged and len(ip_scales) == 1:
        ip_scales = [ip_scales[0], ip_scales[0]]  # duplicate for two refs

    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, "logs"))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision=args.mixed_precision,
        log_with=None,
        project_config=project_config,
    )
    device = accelerator.device

    # Pipeline + components
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model,
        torch_dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16,
        add_watermarker=False,
    )
    vae: AutoencoderKL = pipe.vae
    unet: UNet2DConditionModel = pipe.unet
    tokenizer = pipe.tokenizer
    tokenizer_2 = pipe.tokenizer_2
    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Freeze everything but LoRA adapters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    # Load IP-Adapter (frozen)
    try:
        pipe.load_ip_adapter(args.ip_repo, subfolder=args.ip_subfolder, weight_name=args.ip_weight_name)
        print("[info] IP-Adapter loaded (frozen).")
    except Exception as e:
        warnings.warn(f"Could not load IP-Adapter via diffusers API: {e}")
        raise

    # Keep conditioners in eval (no dropout), and on device
    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    if hasattr(pipe, 'image_encoder') and pipe.image_encoder is not None:
        pipe.image_encoder.to(device)  # keep default dtype (often fp32)
        pipe.image_encoder.eval()

    # Scheduler
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model, subfolder="scheduler")

    # LoRA on UNet (and optionally TEs)
    unet_lcfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha,
        init_lora_weights=args.init_lora_weights,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet = get_peft_model(unet, unet_lcfg)

    if not args.unet_only:
        te_lcfg = LoraConfig(
            r=max(4, args.rank // 2), lora_alpha=max(4, args.rank // 2),
            init_lora_weights=args.init_lora_weights,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )
        text_encoder = get_peft_model(text_encoder, te_lcfg)
        text_encoder_2 = get_peft_model(text_encoder_2, te_lcfg)
        if args.gradient_checkpointing:
            text_encoder.gradient_checkpointing_enable()
            text_encoder_2.gradient_checkpointing_enable()

    # Upcast trainable LoRA params to fp32 for stability
    for m in [unet, *( [] if args.unet_only else [text_encoder, text_encoder_2] )]:
        for p in m.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

    # Dataset
    items = load_paired_list(args.data_root, args.use_ground, args.use_hanged, args.max_train_samples)
    if len(items) == 0:
        raise RuntimeError(f"No paired items found under {args.data_root}")
    captions = ["studio catalog photo, white seamless background, centered garment, evenly lit, sharp edges, no wrinkles, product-only"] * len(items)
    ds = Dataset.from_dict({
        "guidance_paths": [x["guidance_paths"] for x in items],
        "target_path": [x["target_path"] for x in items],
        "caption": captions,
        "num": [x["num"] for x in items],
    })

    tx = make_transforms(args.resolution, args.center_crop, args.random_flip)

    def preprocess(examples):
        timgs = [Image.open(p).convert("RGB") for p in examples["target_path"]]
        examples["pixel_values"] = [tx(im) for im in timgs]
        ids_1 = tokenizer(examples["caption"], max_length=tokenizer.model_max_length,
                          padding="max_length", truncation=True, return_tensors="pt").input_ids
        ids_2 = tokenizer_2(examples["caption"], max_length=tokenizer_2.model_max_length,
                            padding="max_length", truncation=True, return_tensors="pt").input_ids
        examples["input_ids_1"] = ids_1
        examples["input_ids_2"] = ids_2
        return examples

    with accelerator.main_process_first():
        ds = ds.with_transform(preprocess)

    def collate(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch]).to(memory_format=torch.contiguous_format).float()
        input_ids_1 = torch.stack([b["input_ids_1"] for b in batch])
        input_ids_2 = torch.stack([b["input_ids_2"] for b in batch])
        guidance_paths = [b["guidance_paths"] for b in batch]
        return {"pixel_values": pixel_values, "input_ids_1": input_ids_1, "input_ids_2": input_ids_2,
                "guidance_paths": guidance_paths}

    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.train_batch_size, shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True
    )

    # Optimizer/scheduler
    params = itertools.chain(
        (p for p in unet.parameters() if p.requires_grad),
        ([] if args.unet_only else (p for p in text_encoder.parameters() if p.requires_grad)),
        ([] if args.unet_only else (p for p in text_encoder_2.parameters() if p.requires_grad)),
    )
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=args.betas, weight_decay=args.weight_decay, eps=args.eps)
    max_train_steps = args.max_train_steps
    warmup = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=warmup, num_training_steps=max_train_steps
    )

    # Prepare
    unet, optimizer, loader, lr_scheduler = accelerator.prepare(unet, optimizer, loader, lr_scheduler)
    if not args.unet_only:
        text_encoder, text_encoder_2 = accelerator.prepare(text_encoder, text_encoder_2)

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    vae.to(device, dtype=dtype)
    text_encoder.to(device, dtype=dtype)
    text_encoder_2.to(device, dtype=dtype)

    # Cache for IP-Adapter embeds to reduce re-encoding jitter
    ip_cache: Dict[Tuple[Tuple[str, ...], ...], Any] = {}

    global_step = 0
    unet.train()
    if not args.unet_only:
        text_encoder.train(); text_encoder_2.train()

    accelerator.print(f"[pairs] {len(ds)} | steps {max_train_steps} | batch {args.train_batch_size} | accum {args.grad_accum}")

    loss_ma, ma_n = 0.0, 0

    while global_step < max_train_steps:
        for batch in loader:
            with accelerator.accumulate(unet):
                # --- targets -> latents ---
                pixel_values = batch["pixel_values"].to(device, dtype=dtype)
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor

                noise = torch.randn_like(latents)
                if args.noise_offset > 0.0:
                    noise = noise + args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1),
                                                                    device=latents.device, dtype=latents.dtype)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # --- text cond (fp32) ---
                input_ids_1 = batch["input_ids_1"].to(device)
                input_ids_2 = batch["input_ids_2"].to(device)
                enc1 = text_encoder(input_ids_1, output_hidden_states=True)
                enc2 = text_encoder_2(input_ids_2, output_hidden_states=True)
                prompt_embeds = torch.cat([enc1.hidden_states[-2], enc2.hidden_states[-2]], dim=-1).to(torch.float32)
                pooled_embeds = enc2.text_embeds.to(torch.float32)
                add_time_ids = torch.tensor(
                    [[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]],
                    device=device, dtype=torch.float32
                ).repeat(bsz, 1)

                # --- IP-Adapter embeds (FROZEN) ---
                # Build nested list of PIL images and cache key
                batch_key = []
                for paths in batch["guidance_paths"]:
                    batch_key.append(tuple(paths))
                batch_key_t = tuple(batch_key)

                # Cache IP embeds across steps
                if batch_key_t in ip_cache:
                    ip_embeds = ip_cache[batch_key_t]
                else:
                    nested_guidance_images: List[List[Image.Image]] = []
                    for paths in batch["guidance_paths"]:
                        imgs = [Image.open(p).convert("RGB") for p in paths]
                        nested_guidance_images.append(imgs)

                    # Encode & cache
                    ip_embeds = encode_ip_embeds_from_pils(pipe, nested_guidance_images, device)  # [B, D], fp32
                    ip_cache[batch_key_t] = ip_embeds

                # Use a single scalar scale since we averaged multiple refs into one embedding
                ip_scale = float(str(args.ip_scales).split(",")[0])

                # --- UNet forward ---
                # UNet expects 'image_embeds' with this config
                model_pred = unet(
                    noisy_latents, timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={
                        "text_embeds": pooled_embeds,       # SDXL pooled (fp32)
                        "time_ids": add_time_ids,           # SDXL time ids (fp32)
                        "image_embeds": ip_embeds * ip_scale,  # IP-Adapter embeds with scaling
                    },
                ).sample

                # --- loss ---
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.use_min_snr:
                    base = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(dim=(1,2,3))
                    with torch.no_grad():
                        ac = noise_scheduler.alphas_cumprod.to(device)
                        snr = (ac[timesteps] / (1 - ac[timesteps])).clamp(min=1e-8)
                    w = (snr + 1) / (snr + args.min_snr_gamma)
                    loss = (w * base).mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # sigma-normalized diagnostic (more comparable across t)
                with torch.no_grad():
                    ac = noise_scheduler.alphas_cumprod.to(device)
                    sigma = (1.0 - ac[timesteps]).sqrt().view(-1,1,1,1)
                sigma_norm_loss = F.mse_loss(
                    (model_pred - target).float() / (sigma + 1e-8),
                    torch.zeros_like(model_pred.float()), reduction="mean"
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    lora_params = [p for p in unet.parameters() if p.requires_grad]
                    accelerator.clip_grad_norm_(lora_params, args.max_grad_norm)
                    if not args.unet_only:
                        accelerator.clip_grad_norm_([p for p in text_encoder.parameters() if p.requires_grad], args.max_grad_norm)
                        accelerator.clip_grad_norm_([p for p in text_encoder_2.parameters() if p.requires_grad], args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                loss_ma += loss.item(); ma_n += 1
                if accelerator.is_main_process and args.checkpointing_steps and global_step % args.checkpointing_steps == 0:
                    save_lora(pipe, unet, text_encoder if not args.unet_only else None,
                              text_encoder_2 if not args.unet_only else None, args, tag=f"step{global_step}")
                if global_step % args.log_every == 0:
                    avg = loss_ma / max(ma_n, 1)
                    accelerator.print(f"step {global_step}/{max_train_steps} | loss {loss.item():.4f} | avg {avg:.4f} | sigma {sigma_norm_loss.item():.4f} | lr {lr_scheduler.get_last_lr()[0]:.2e}")
                    loss_ma, ma_n = 0.0, 0

            if global_step >= max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lora(pipe, unet, text_encoder if not args.unet_only else None,
                  text_encoder_2 if not args.unet_only else None, args, tag="final")
    accelerator.end_training()


if __name__ == "__main__":
    main()
