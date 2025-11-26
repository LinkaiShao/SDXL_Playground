#!/usr/bin/env python3
"""
Test different combinations of LoRA and IP-Adapter weights in a grid.
"""

import os
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from safetensors.torch import load_file, save_file
import tempfile
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
MODEL_DIR = "/home/link/Desktop/Code/fashion gen testing/sdxl-base"
LORA_DIR = "/home/link/Desktop/Code/fashion gen testing/sdxl_lora_out_ip"
RAW_IMAGE_PATH = "/home/link/Desktop/Code/fashion gen testing/straighten_image_data/raw/8861882df78e45de96b38e2423fa0fc5.webp"
OUTPUT_DIR = "/home/link/Desktop/Code/fashion gen testing/lora_grid_test"
TEMP_WHITE_BG_PATH = os.path.join(OUTPUT_DIR, "temp_white_bg.png")

# IP-Adapter settings
IP_REPO = "h94/IP-Adapter"
IP_SUBFOLDER = "sdxl_models"
IP_WEIGHT_NAME = "ip-adapter_sdxl.bin"

# Generation settings
SEED = 42
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5
DENOISE_STRENGTH = 0.85

# Grid parameters - test different weight combinations
LORA_CHECKPOINT = "lora_sdxl_ip_rank16_steps1500_step1500.safetensors"  # Use final checkpoint
LORA_SCALES = [0.5, 1.0, 1.5, 2.0]
IP_SCALES = [0.2, 0.4, 0.6, 0.8]


def remove_background_birefnet(image_path, output_path):
    """Remove background using BiRefNet and add white background."""
    print("Loading BiRefNet...")
    birefnet = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        trust_remote_code=True
    )
    birefnet.to(device)

    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = birefnet(input_tensor)[-1].sigmoid().cpu()

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image.size, Image.Resampling.BILINEAR)

    # Create white background
    white_bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    image_rgba = image.convert("RGBA")

    # Apply mask
    image_rgba.putalpha(mask)
    white_bg.paste(image_rgba, (0, 0), image_rgba)

    # Convert to RGB
    white_bg = white_bg.convert("RGB")
    white_bg.save(output_path)
    print(f"Saved white background image: {output_path}")

    del birefnet
    torch.cuda.empty_cache()

    return white_bg


def load_pipeline_with_lora(model_dir, lora_path, lora_scale, ip_scale, ip_repo, ip_subfolder, ip_weight_name):
    """Load SDXL pipeline with IP-Adapter and LoRA."""
    print(f"\nLoading pipeline (LoRA={lora_scale}, IP={ip_scale})...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    )
    pipe = pipe.to(device)

    # Load IP-Adapter
    try:
        pipe.load_ip_adapter(ip_repo, subfolder=ip_subfolder, weight_name=ip_weight_name)
        pipe.set_ip_adapter_scale(ip_scale)
    except Exception as e:
        print(f"Warning: Could not load IP-Adapter: {e}")
        return None

    # Load LoRA weights
    try:
        state_dict = load_file(lora_path)

        # Convert PEFT format to diffusers format
        converted_state_dict = {}
        for key, value in state_dict.items():
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
            pipe.load_lora_weights(tmpdir, weight_name="temp_lora.safetensors")

        # Fuse LoRA weights
        pipe.fuse_lora(lora_scale=lora_scale)

    except Exception as e:
        print(f"Error loading LoRA: {e}")
        import traceback
        traceback.print_exc()
        return None

    return pipe


def generate_image(pipe, guidance_image, prompt, seed=42, num_steps=50, guidance_scale=7.5, denoise_strength=0.85):
    """Generate image with img2img."""
    generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        image=guidance_image,
        ip_adapter_image=guidance_image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        strength=denoise_strength,
        generator=generator,
    ).images[0]

    return result


def create_grid(images, labels, rows, cols, output_path):
    """Create a grid of images with labels."""
    if not images:
        print("No images to create grid")
        return

    # Get dimensions from first image
    img_width, img_height = images[0].size

    # Add space for labels
    label_height = 60
    cell_width = img_width
    cell_height = img_height + label_height

    # Create grid
    grid_width = cols * cell_width
    grid_height = rows * cell_height
    grid = Image.new('RGB', (grid_width, grid_height), color='white')

    from PIL import ImageDraw, ImageFont

    for idx, (img, label) in enumerate(zip(images, labels)):
        row = idx // cols
        col = idx % cols

        x = col * cell_width
        y = row * cell_height + label_height  # Leave space for label at top

        grid.paste(img, (x, y))

        # Draw label
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()

        # Center text
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (cell_width - text_width) // 2
        text_y = row * cell_height + 15

        draw.text((text_x, text_y), label, fill='black', font=font)

    grid.save(output_path)
    print(f"\nSaved grid to: {output_path}")


def main():
    print("=" * 60)
    print("Testing LoRA and IP-Adapter Weight Grid")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Remove background once
    print("\n[Step 1] Removing background...")
    white_bg_image = remove_background_birefnet(RAW_IMAGE_PATH, TEMP_WHITE_BG_PATH)

    # Step 2: Generate grid of outputs
    lora_path = os.path.join(LORA_DIR, LORA_CHECKPOINT)
    prompt = "a garment laid flat on white background, straightened and dewrinkled, professional product photo, studio catalog, centered, evenly lit, sharp edges"

    images = []
    labels = []

    total = len(LORA_SCALES) * len(IP_SCALES)
    current = 0

    for lora_scale, ip_scale in itertools.product(LORA_SCALES, IP_SCALES):
        current += 1
        print(f"\n[{current}/{total}] Generating with LoRA={lora_scale}, IP={ip_scale}...")

        # Load pipeline with these scales
        pipe = load_pipeline_with_lora(
            MODEL_DIR, lora_path, lora_scale, ip_scale,
            IP_REPO, IP_SUBFOLDER, IP_WEIGHT_NAME
        )

        if pipe is None:
            print(f"Skipping LoRA={lora_scale}, IP={ip_scale}")
            continue

        # Generate image
        output_image = generate_image(
            pipe, white_bg_image, prompt,
            seed=SEED, num_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE, denoise_strength=DENOISE_STRENGTH
        )

        # Save individual image
        output_filename = f"lora{lora_scale}_ip{ip_scale}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        output_image.save(output_path)
        print(f"Saved: {output_path}")

        images.append(output_image)
        labels.append(f"LoRA:{lora_scale} IP:{ip_scale}")

        # Clean up
        try:
            pipe.unfuse_lora()
        except:
            pass
        del pipe
        torch.cuda.empty_cache()

    # Step 3: Create grid
    print("\n[Step 3] Creating comparison grid...")
    grid_path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
    create_grid(images, labels, rows=len(LORA_SCALES), cols=len(IP_SCALES), output_path=grid_path)

    print("\n" + "=" * 60)
    print("[Done] Grid testing complete")
    print(f"Individual images: {OUTPUT_DIR}/")
    print(f"Comparison grid: {grid_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
