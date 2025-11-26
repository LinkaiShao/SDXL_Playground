#!/usr/bin/env python3
"""
Test script for the trained SDXL LoRA with IP-Adapter.

Pipeline:
1. Load raw input image
2. Use BiRefNet to remove background
3. Add white background
4. Load SDXL pipeline with IP-Adapter
5. Load trained LoRA weights
6. Generate straightened catalog image using the processed image as guidance
"""

import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline

# ====== PATHS ======
OUTPUT_DIR = "/home/link/Desktop/Code/fashion gen testing/lora_to_catalog_test"
RAW_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data/raw/8861882df78e45de96b38e2423fa0fc5.webp"
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/sdxl-base"
LORA_DIR = "/home/link/Desktop/Code/fashion gen testing/sdxl_lora_out_ip"

# Test multiple checkpoints
LORA_CHECKPOINTS = [
    ("step500", "lora_sdxl_ip_rank16_steps1500_step500.safetensors"),
    ("step1000", "lora_sdxl_ip_rank16_steps1500_step1000.safetensors"),
    ("step1500", "lora_sdxl_ip_rank16_steps1500_step1500.safetensors"),
]

TEMP_WHITE_BG_PATH = os.path.join(OUTPUT_DIR, "temp_white_bg.png")

# IP-Adapter settings (same as training)
IP_REPO = "h94/IP-Adapter"
IP_SUBFOLDER = "sdxl_models"
IP_WEIGHT_NAME = "ip-adapter_sdxl.bin"
IP_SCALE = 0.3  # Lower to prevent exact copying (was 0.6)

# ====== PARAMETERS ======
RESOLUTION = 1024
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
LORA_SCALE = 1.5  # Increase for stronger LoRA effect (default 1.0, range 0.0-2.0)
DENOISE_STRENGTH = 0.85  # For img2img mode: how much to change (0=no change, 1=complete change)
SEED = 42

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def remove_background_birefnet(image_path, output_path):
    """
    Use BiRefNet to remove background and add white background.
    """
    print("Loading BiRefNet...")
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet", trust_remote_code=True
    ).to(device).eval()

    tx = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    print(f"Processing image: {image_path}")
    im = Image.open(image_path).convert("RGB")
    orig_size = im.size

    with torch.no_grad():
        x = tx(im).unsqueeze(0).to(device)
        # Model returns a list/tuple; the last item is the finest mask
        pred = birefnet(x)[-1].sigmoid().squeeze().detach().cpu()
        mask = transforms.ToPILImage()(pred).resize(orig_size, Image.BICUBIC)

    # Create image with white background
    white_bg = Image.new("RGB", orig_size, (255, 255, 255))
    im_rgba = im.copy()
    im_rgba.putalpha(mask)
    white_bg.paste(im_rgba, (0, 0), im_rgba)

    white_bg.save(output_path)
    print(f"Saved white background image: {output_path}")

    return white_bg


def load_pipeline_with_lora(model_dir, lora_path, ip_repo, ip_subfolder, ip_weight_name):
    """
    Load SDXL pipeline with IP-Adapter and LoRA.
    """
    from safetensors.torch import load_file
    from safetensors.torch import save_file
    import os
    import tempfile

    print("Loading SDXL pipeline...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe = pipe.to(device)

    # Load IP-Adapter
    print("Loading IP-Adapter...")
    try:
        pipe.load_ip_adapter(ip_repo, subfolder=ip_subfolder, weight_name=ip_weight_name)
        pipe.set_ip_adapter_scale(IP_SCALE)
        print(f"IP-Adapter loaded with scale {IP_SCALE}")
    except Exception as e:
        print(f"Warning: Could not load IP-Adapter: {e}")
        return None

    # Load LoRA weights manually and fix the key format
    print(f"Loading LoRA from: {lora_path}")
    try:
        # Load the safetensors file
        state_dict = load_file(lora_path)

        # Convert PEFT format to diffusers format
        # Keys are like: unet.base_model.model.down_blocks.X.attn.to_k.lora.down.weight
        # Should be: unet.down_blocks.X.attn.to_k.lora_A.weight (for lora.down) or lora_B.weight (for lora.up)
        converted_state_dict = {}
        for key, value in state_dict.items():
            # Only process unet keys
            if not key.startswith('unet.'):
                continue

            # Remove "base_model.model." but keep "unet." prefix
            new_key = key.replace('unet.base_model.model.', 'unet.')

            # Convert lora.down -> lora_A and lora.up -> lora_B
            new_key = new_key.replace('.lora.down.weight', '.lora_A.weight')
            new_key = new_key.replace('.lora.up.weight', '.lora_B.weight')

            converted_state_dict[new_key] = value

        # Save to temporary file and load via diffusers
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_lora_path = os.path.join(tmpdir, "temp_lora.safetensors")
            save_file(converted_state_dict, tmp_lora_path)

            # Load using diffusers
            pipe.load_lora_weights(tmpdir, weight_name="temp_lora.safetensors")

        # Fuse LoRA weights into the model for inference
        pipe.fuse_lora(lora_scale=LORA_SCALE)

        print(f"LoRA weights loaded and fused with scale {LORA_SCALE}")
    except Exception as e:
        print(f"Error loading LoRA: {e}")
        import traceback
        traceback.print_exc()
        return None

    return pipe


def generate_straightened_image(pipe, guidance_image, prompt, seed=42, num_steps=50, guidance_scale=7.5):
    """
    Generate straightened catalog image using img2img with IP-Adapter guidance and LoRA.
    """
    print("Generating straightened image...")
    print(f"  - Steps: {num_steps}")
    print(f"  - Guidance scale: {guidance_scale}")
    print(f"  - Denoise strength: {DENOISE_STRENGTH}")
    print(f"  - LoRA scale: {LORA_SCALE}")
    print(f"  - IP-Adapter scale: {IP_SCALE}")
    print(f"  - Seed: {seed}")

    generator = torch.Generator(device=device).manual_seed(seed)

    # Resize guidance image to match resolution
    guidance_image = guidance_image.resize((RESOLUTION, RESOLUTION), Image.BICUBIC)

    # Convert pipeline to img2img for better control
    img2img_pipe = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    img2img_pipe.register_to_config(**pipe.config)

    # Copy IP-Adapter components if they exist
    if hasattr(pipe, 'image_encoder'):
        img2img_pipe.image_encoder = pipe.image_encoder
        img2img_pipe.feature_extractor = pipe.feature_extractor

    # Set IP-Adapter scale
    if hasattr(img2img_pipe, 'set_ip_adapter_scale'):
        img2img_pipe.set_ip_adapter_scale(IP_SCALE)

    # Generate with img2img + IP-Adapter + LoRA
    result = img2img_pipe(
        prompt=prompt,
        negative_prompt="wrinkled, folded, hanging, creased, distorted, shadows, blurry, low quality, deformed",
        image=guidance_image,
        ip_adapter_image=guidance_image if hasattr(img2img_pipe, 'image_encoder') else None,
        strength=DENOISE_STRENGTH,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    return result


def main():
    print("=" * 60)
    print("Testing SDXL LoRA Checkpoints with IP-Adapter")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")

    # Step 1: Remove background once (reuse for all checkpoints)
    print("\n[Step 1] Removing background with BiRefNet...")
    white_bg_image = remove_background_birefnet(RAW_IMAGE_PATH, TEMP_WHITE_BG_PATH)

    # Step 2 & 3: Test each checkpoint
    for checkpoint_name, checkpoint_file in LORA_CHECKPOINTS:
        print(f"\n{'=' * 60}")
        print(f"[Testing {checkpoint_name}]")
        print(f"{'=' * 60}")

        lora_path = os.path.join(LORA_DIR, checkpoint_file)
        output_path = os.path.join(OUTPUT_DIR, f"output_{checkpoint_name}.png")

        # Load pipeline with this checkpoint
        print(f"\n[Step 2] Loading SDXL pipeline with IP-Adapter and LoRA...")
        pipe = load_pipeline_with_lora(MODEL_DIR, lora_path, IP_REPO, IP_SUBFOLDER, IP_WEIGHT_NAME)

        if pipe is None:
            print(f"Failed to load {checkpoint_name}. Skipping.")
            continue

        # Generate straightened output
        print(f"\n[Step 3] Generating straightened catalog image...")
        prompt = "a garment laid flat on white background, straightened and dewrinkled, professional product photo, studio catalog, centered, evenly lit, sharp edges"

        output_image = generate_straightened_image(
            pipe,
            white_bg_image,
            prompt,
            seed=SEED,
            num_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE
        )

        # Save output
        output_image.save(output_path)
        print(f"\n[Done] Saved: {output_path}")

        # Clean up to free memory before loading next checkpoint
        # Unfuse LoRA before deleting
        try:
            pipe.unfuse_lora()
        except:
            pass  # In case unfuse fails

        del pipe
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print("[Done] All checkpoints tested")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
