#!/usr/bin/env python3
"""
ControlNet Catalog Photo Parameter Testing
Tests multiple parameter combinations for wrinkle-free catalog transformation
"""

import torch
import gc
import os
import json
from pathlib import Path
from PIL import Image
from diffusers import ControlNetModel
from controlnet_aux import CannyDetector

# Memory optimization
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Try img2img variant
try:
    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
    USING_IMG2IMG = True
except ImportError:
    from diffusers import StableDiffusionXLControlNetPipeline
    USING_IMG2IMG = False

# Paths
INPUT_IMAGE = Path("u2net_output/birefnet_output_2.png").resolve()
PARAMS_FILE = Path("controlnet_params.json").resolve()
OUTPUT_DIR = Path("controlnet_catalog_output").resolve()
BASE_MODEL = "./sdxl-base"
CONTROLNET_MODEL = "./controlnet-canny-sdxl"

device = "cuda"

def round_to_multiple(value, multiple=64):
    """Round to nearest multiple"""
    return int(round(value / multiple) * multiple)

def prepare_image(image_path):
    """Load, composite RGBA to RGB on white background, and resize for SDXL"""
    img_original = Image.open(image_path)

    # Composite to white background
    if img_original.mode in ("RGBA", "LA"):
        white_bg = Image.new("RGB", img_original.size, (255, 255, 255))
        white_bg.paste(img_original, mask=img_original.split()[-1])
        img = white_bg
    else:
        img = img_original.convert("RGB")

    print(f"Input prepared: {img.mode}, {img.size}")

    # Resize to SDXL-compatible dimensions (must be multiple of 64, max 1024)
    max_size = 1024
    if img.width > max_size or img.height > max_size:
        aspect = img.width / img.height
        if aspect > 1:
            new_w = max_size
            new_h = int(max_size / aspect)
        else:
            new_w = int(max_size * aspect)
            new_h = max_size

        new_w = round_to_multiple(new_w, 64)
        new_h = round_to_multiple(new_h, 64)

        img = img.resize((new_w, new_h), Image.LANCZOS)
        print(f"Resized to SDXL-compatible: {img.size}")
    else:
        new_w = round_to_multiple(img.width, 64)
        new_h = round_to_multiple(img.height, 64)
        if (new_w, new_h) != (img.width, img.height):
            img = img.resize((new_w, new_h), Image.LANCZOS)
            print(f"Adjusted to SDXL-compatible: {img.size}")

    return img

def extract_canny(img, low=150, high=250):
    """Extract Canny edges from image"""
    detector = CannyDetector()
    canny = detector(img, low_threshold=low, high_threshold=high)

    # Ensure canny matches image size
    if canny.size != img.size:
        print(f"WARNING: Canny size mismatch! Resizing from {canny.size} to {img.size}")
        canny = canny.resize(img.size, Image.LANCZOS)

    return canny

def run_controlnet(pipe, img, canny, param_set, param_name, using_img2img):
    """Run ControlNet with specific parameters"""
    print(f"\n{'='*60}")
    print(f"Testing: {param_name}")
    print(f"Description: {param_set['desc']}")
    print(f"Parameters: strength={param_set['strength']}, scale={param_set['controlnet_conditioning_scale']}, guidance={param_set['guidance_scale']}")
    print(f"{'='*60}")

    if using_img2img:
        output = pipe(
            prompt=param_set['prompt'],
            negative_prompt=param_set['negative_prompt'],
            image=img,
            control_image=canny,
            strength=param_set['strength'],
            controlnet_conditioning_scale=param_set['controlnet_conditioning_scale'],
            guidance_scale=param_set['guidance_scale'],
            num_inference_steps=30,
        ).images[0]
    else:
        # Fallback to text2img mode
        output = pipe(
            prompt=param_set['prompt'],
            negative_prompt=param_set['negative_prompt'],
            image=canny,
            num_inference_steps=30,
            controlnet_conditioning_scale=param_set['controlnet_conditioning_scale'],
            guidance_scale=param_set['guidance_scale'],
            height=1024, width=1024,
        ).images[0]

    # Save output
    output_path = OUTPUT_DIR / f"{param_name}.png"
    output.save(output_path)

    # Save config
    config_path = OUTPUT_DIR / f"{param_name}_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Parameter Set: {param_name}\n")
        f.write(f"Description: {param_set['desc']}\n")
        f.write(f"Mode: {'img2img' if using_img2img else 'text2img'}\n\n")
        f.write(json.dumps(param_set, indent=2))

    print(f"✓ Saved: {output_path.name}")
    return output

def main():
    # Clear GPU memory first
    print("Clearing GPU memory...")
    torch.cuda.empty_cache()
    gc.collect()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load parameters
    print(f"Loading parameter sets from: {PARAMS_FILE}")
    with open(PARAMS_FILE, 'r') as f:
        param_sets = json.load(f)
    print(f"Loaded {len(param_sets)} configurations\n")

    if USING_IMG2IMG:
        print("✓ Using img2img mode - image will provide appearance reference")
    else:
        print("⚠ Using text2img mode - no appearance reference (limited results)")

    # Prepare input
    print(f"\nLoading input: {INPUT_IMAGE}")
    img = prepare_image(INPUT_IMAGE)
    print(f"Input prepared: {img.mode}, {img.size}")

    # Extract Canny
    print("Extracting Canny edges...")
    canny = extract_canny(img)

    # Save references
    img.save(OUTPUT_DIR / "00_input.png")
    canny.save(OUTPUT_DIR / "00_canny_edges.png")
    print("Saved input references\n")

    # Load models
    print("Loading SDXL ControlNet models...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_MODEL, torch_dtype=torch.float16
    )

    if USING_IMG2IMG:
        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            BASE_MODEL, controlnet=controlnet, torch_dtype=torch.float16
        )
    else:
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL, controlnet=controlnet, torch_dtype=torch.float16
        )

    # Enable memory optimizations
    print("Enabling memory optimizations...")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_attention_slicing()

    # CPU offloading to reduce VRAM usage
    try:
        pipe.enable_sequential_cpu_offload()
        print("✓ Using sequential CPU offload")
    except Exception:
        pipe.to(device)
        print("✓ Using CUDA (no offload available)")

    print("Models loaded!\n")

    # Run all parameter sets
    results = {}
    for param_name, param_set in param_sets.items():
        try:
            run_controlnet(pipe, img, canny, param_set, param_name, USING_IMG2IMG)
            results[param_name] = "SUCCESS"

            # Clear memory between runs
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"✗ ERROR: {e}")
            results[param_name] = f"FAILED: {e}"

            # Clear memory even after failures
            torch.cuda.empty_cache()
            gc.collect()

    # Summary
    print(f"\n{'='*60}")
    print("CONTROLNET PARAMETER TEST COMPLETE")
    print(f"{'='*60}")
    for name, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} {name}: {status}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nRecommended comparison order:")
    print("1. controlnet_1_balanced - Good starting point")
    print("2. controlnet_2_max_preservation - If appearance is changing too much")
    print("3. controlnet_4_strict_structure - If wrinkles remain")
    print("4. controlnet_3_more_freedom - If wrinkles persist")

if __name__ == "__main__":
    main()
