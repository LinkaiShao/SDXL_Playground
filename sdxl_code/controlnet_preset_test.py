#!/usr/bin/env python3
"""
ControlNet Parameter Preset Testing
Automatically tests all 6 parameter presets for wrinkle removal
"""

import torch
import gc
import os
from pathlib import Path
from PIL import Image
from diffusers import ControlNetModel
from controlnet_aux import CannyDetector

# Memory optimization
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Clear GPU memory first
print("Clearing GPU memory...")
torch.cuda.empty_cache()
gc.collect()

# Try to use img2img variant
try:
    from diffusers import StableDiffusionXLControlNetImg2ImgPipeline
    USING_IMG2IMG = True
    print("Using StableDiffusionXLControlNetImg2ImgPipeline (img2img mode)")
except ImportError:
    from diffusers import StableDiffusionXLControlNetPipeline
    USING_IMG2IMG = False
    print("Using StableDiffusionXLControlNetPipeline (text2img mode)")

device = "cuda"

# ================== PARAMETER PRESETS ==================
PRESETS = {
    "balanced": {
        "strength": 0.45,
        "controlnet_scale": 0.70,
        "guidance": 7.0,
        "canny_low": 150,
        "canny_high": 250,
        "desc": "Balanced preservation and transformation"
    },
    "max_preservation": {
        "strength": 0.35,
        "controlnet_scale": 0.75,
        "guidance": 6.5,
        "canny_low": 150,
        "canny_high": 250,
        "desc": "Maximum appearance preservation, minimal changes"
    },
    "more_freedom": {
        "strength": 0.55,
        "controlnet_scale": 0.65,
        "guidance": 7.5,
        "canny_low": 150,
        "canny_high": 250,
        "desc": "More freedom to remove wrinkles"
    },
    "strict_structure": {
        "strength": 0.40,
        "controlnet_scale": 0.80,
        "guidance": 7.0,
        "canny_low": 120,
        "canny_high": 220,
        "desc": "Very strict edge following, best appearance match"
    },
    "aggressive": {
        "strength": 0.60,
        "controlnet_scale": 0.60,
        "guidance": 8.0,
        "canny_low": 170,
        "canny_high": 270,
        "desc": "Aggressive wrinkle removal"
    },
    "subtle": {
        "strength": 0.30,
        "controlnet_scale": 0.75,
        "guidance": 6.0,
        "canny_low": 150,
        "canny_high": 250,
        "desc": "Very subtle changes, almost identical to input"
    }
}
# =======================================================

# Paths
INPUT_IMAGE = "u2net_output/birefnet_output_2.png"
BASE_MODEL = "./sdxl-base"
CONTROLNET_MODEL = "./controlnet-canny-sdxl"
OUTPUT_DIR = Path("controlnet_canny")

def prepare_image(image_path):
    """Composite RGBA onto white background"""
    img_original = Image.open(image_path)

    if img_original.mode in ("RGBA", "LA"):
        print("Compositing transparent image onto white background...")
        white_bg = Image.new("RGB", img_original.size, (255, 255, 255))
        white_bg.paste(img_original, mask=img_original.split()[-1])
        return white_bg
    else:
        return img_original.convert("RGB")

def round_to_multiple(value, multiple=64):
    """Round to nearest multiple"""
    return int(round(value / multiple) * multiple)

def resize_for_sdxl(img, max_size=1024):
    """Resize image to SDXL-compatible dimensions (multiple of 64)"""
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
    print(f"Extracting Canny edges (low={low}, high={high})...")
    canny = CannyDetector()(img, low_threshold=low, high_threshold=high)

    # Ensure canny matches image size
    if canny.size != img.size:
        print(f"WARNING: Canny size mismatch! Resizing from {canny.size} to {img.size}")
        canny = canny.resize(img.size, Image.LANCZOS)

    return canny

def run_preset(pipe, img, preset_name, params):
    """Run ControlNet with specific preset"""
    print(f"\n{'='*60}")
    print(f"TESTING PRESET: {preset_name}")
    print(f"Description: {params['desc']}")
    print(f"Strength: {params['strength']}, ControlNet Scale: {params['controlnet_scale']}")
    print(f"Guidance: {params['guidance']}, Canny: {params['canny_low']}/{params['canny_high']}")
    print(f"{'='*60}\n")

    # Extract Canny with preset-specific thresholds
    canny = extract_canny(img, params['canny_low'], params['canny_high'])

    # Save Canny edges for this preset
    canny.save(OUTPUT_DIR / f"02_canny_edges_{preset_name}.png")
    print(f"Saved Canny edges to 02_canny_edges_{preset_name}.png")

    # Generic prompt - NO garment description
    prompt = "professional e-commerce catalog photography, perfectly pressed wrinkle-free garment, pure white seamless background, even studio lighting, smooth fabric"
    negative_prompt = "wrinkles, creases, folds, rumpled, gray background, colored background, shadows on background, uneven lighting, texture loss"

    with torch.inference_mode():
        if USING_IMG2IMG:
            print("Running img2img with ControlNet...")

            # Verify dimensions
            assert img.width % 64 == 0 and img.height % 64 == 0, f"Image dims not multiple of 64: {img.size}"
            assert canny.width % 64 == 0 and canny.height % 64 == 0, f"Canny dims not multiple of 64: {canny.size}"
            assert img.size == canny.size, f"Size mismatch: img={img.size}, canny={canny.size}"

            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img,
                control_image=canny,
                strength=params['strength'],
                controlnet_conditioning_scale=params['controlnet_scale'],
                guidance_scale=params['guidance'],
                num_inference_steps=30,
            ).images[0]
        else:
            print("Running text2img with ControlNet...")
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny,
                num_inference_steps=30,
                controlnet_conditioning_scale=params['controlnet_scale'],
                guidance_scale=params['guidance'],
                height=img.height, width=img.width,
            ).images[0]

    # Save output
    output_filename = f"03_controlnet_output_{preset_name}.png"
    out.save(OUTPUT_DIR / output_filename)
    print(f"✓ Saved output: {output_filename}")

    # Save config
    config_path = OUTPUT_DIR / f"03_controlnet_output_{preset_name}_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Preset: {preset_name}\n")
        f.write(f"Description: {params['desc']}\n\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Saved config: {config_path.name}\n")

    return out

def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"CONTROLNET PRESET TESTING")
    print(f"Testing {len(PRESETS)} parameter presets")
    print(f"{'='*60}\n")

    # Prepare input image
    print(f"Loading input: {INPUT_IMAGE}")
    img = prepare_image(INPUT_IMAGE)
    print(f"Input prepared: {img.mode}, {img.size}")

    # Resize for SDXL
    img = resize_for_sdxl(img)
    print(f"Final input size: {img.size}")

    # Save input reference
    img.save(OUTPUT_DIR / "01_input_white_bg.png")
    print(f"Saved input to 01_input_white_bg.png\n")

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
    print("✓ Using attention slicing")

    # CPU offloading
    try:
        pipe.enable_sequential_cpu_offload()
        print("✓ Using sequential CPU offload")
    except Exception:
        pipe.to(device)
        print("✓ Using CUDA (no offload available)")

    print("\nModels loaded!\n")

    # Run all presets
    results = {}
    for preset_name, params in PRESETS.items():
        try:
            run_preset(pipe, img, preset_name, params)
            results[preset_name] = "SUCCESS"

            # Clear cache between runs
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"✗ ERROR: {e}")
            results[preset_name] = f"FAILED: {e}"

    # Summary
    print(f"\n{'='*60}")
    print("CONTROLNET PRESET TEST COMPLETE")
    print(f"{'='*60}")
    for name, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} {name}: {status}")

    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nRecommended comparison order:")
    print("1. balanced - Good starting point")
    print("2. max_preservation - If appearance is changing too much")
    print("3. strict_structure - If wrinkles remain but appearance is good")
    print("4. more_freedom - If wrinkles persist")
    print("5. aggressive - If wrinkles still not removed")
    print("6. subtle - If all others are too aggressive")

    if USING_IMG2IMG:
        print("\n✓ Used img2img mode - original image provided appearance reference")
    else:
        print("\n⚠ Used text2img mode - no appearance reference available")

if __name__ == "__main__":
    main()
