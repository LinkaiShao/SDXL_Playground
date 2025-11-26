import torch
import gc
import os
from pathlib import Path
from diffusers import ControlNetModel
from diffusers.utils import load_image
from controlnet_aux import CannyDetector

# Memory optimization (use new name to avoid deprecation warning)
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# Clear GPU memory first
print("Clearing GPU memory...")
torch.cuda.empty_cache()
gc.collect()

# Try to use img2img variant, fall back to regular if not available
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
# Set to True to test all presets, False to test only one
TEST_ALL_PRESETS = True
ACTIVE_PRESET = "balanced"  # Used if TEST_ALL_PRESETS = False

PRESETS = {
    # Ultra-low strength for maximum realism
    "ultra_realistic": {
        "strength": 0.15,
        "controlnet_scale": 0.95,
        "guidance": 4.5,
        "canny_low": 100,
        "canny_high": 200,
        "desc": "Maximum realism - minimal transformation"
    },
    "photo_realistic_1": {
        "strength": 0.20,
        "controlnet_scale": 0.90,
        "guidance": 5.0,
        "canny_low": 120,
        "canny_high": 220,
        "desc": "Strong photo realism, slight smoothing"
    },
    "photo_realistic_2": {
        "strength": 0.25,
        "controlnet_scale": 0.85,
        "guidance": 5.5,
        "canny_low": 140,
        "canny_high": 240,
        "desc": "Good photo realism with more smoothing"
    },
    "photo_realistic_3": {
        "strength": 0.30,
        "controlnet_scale": 0.80,
        "guidance": 6.0,
        "canny_low": 150,
        "canny_high": 250,
        "desc": "Balanced realism and wrinkle reduction"
    },
    # Medium strength options
    "smooth_realistic": {
        "strength": 0.35,
        "controlnet_scale": 0.75,
        "guidance": 6.0,
        "canny_low": 160,
        "canny_high": 260,
        "desc": "Smoother fabric while keeping realism"
    },
    "enhanced_smooth": {
        "strength": 0.40,
        "controlnet_scale": 0.70,
        "guidance": 6.5,
        "canny_low": 170,
        "canny_high": 270,
        "desc": "Enhanced smoothing, moderate realism"
    },
    # Higher strength for more wrinkle removal
    "strong_smooth": {
        "strength": 0.45,
        "controlnet_scale": 0.65,
        "guidance": 7.0,
        "canny_low": 180,
        "canny_high": 280,
        "desc": "Strong wrinkle removal, may lose some realism"
    },
    "aggressive_smooth": {
        "strength": 0.50,
        "controlnet_scale": 0.60,
        "guidance": 7.0,
        "canny_low": 200,
        "canny_high": 300,
        "desc": "Aggressive transformation for stubborn wrinkles"
    },
    # Low guidance variants for natural look
    "natural_low_guide": {
        "strength": 0.25,
        "controlnet_scale": 0.85,
        "guidance": 4.0,
        "canny_low": 120,
        "canny_high": 220,
        "desc": "Low guidance for more natural variation"
    },
    "natural_balanced": {
        "strength": 0.30,
        "controlnet_scale": 0.80,
        "guidance": 4.5,
        "canny_low": 140,
        "canny_high": 240,
        "desc": "Natural look with balanced smoothing"
    }
}
# =======================================================

# 1. Load base SDXL
base_model = "./sdxl-base"

# 2. Load ControlNet (Canny example)
controlnet = ControlNetModel.from_pretrained(
    "./controlnet-canny-sdxl", torch_dtype=torch.float16
)

# 3. Create pipeline with memory optimizations
if USING_IMG2IMG:
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16
    )
else:
    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_model, controlnet=controlnet, torch_dtype=torch.float16
    )

# Enable all memory optimizations
print("Enabling memory optimizations...")
pipe.enable_vae_slicing()
pipe.enable_vae_tiling()

# Disable xformers due to CUDA errors, use attention slicing instead
pipe.enable_attention_slicing()
print("✓ Using attention slicing (xformers disabled due to compatibility)")

# Alternative: try xformers if you want to test it
# try:
#     pipe.enable_xformers_memory_efficient_attention()
#     print("✓ Using xformers")
# except Exception:
#     pipe.enable_attention_slicing()
#     print("✓ Using attention slicing")

# CPU offloading (CRITICAL for VRAM)
try:
    pipe.enable_sequential_cpu_offload()
    print("✓ Using sequential CPU offload")
except Exception:
    pipe.to(device)
    print("✓ Using CUDA (no offload available)")

# Helper functions
def round_to_multiple(value, multiple=64):
    """Round to nearest multiple"""
    return int(round(value / multiple) * multiple)

def prepare_image(img_path):
    """Load and prepare image with white background"""
    from PIL import Image
    img_original = Image.open(img_path)

    # Composite RGBA onto white background
    if img_original.mode in ("RGBA", "LA"):
        print("Compositing transparent image onto white background...")
        white_bg = Image.new("RGB", img_original.size, (255, 255, 255))
        white_bg.paste(img_original, mask=img_original.split()[-1])
        img = white_bg
    else:
        img = img_original.convert("RGB")

    print(f"Input prepared: {img.mode}, {img.size}")

    # Resize to SDXL-compatible dimensions (must be multiple of 64)
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

def run_inference(pipe, img, params, preset_name, output_dir):
    """Run ControlNet inference with given parameters"""
    from PIL import Image

    print(f"\n{'='*60}")
    print(f"TESTING PRESET: {preset_name}")
    print(f"Description: {params['desc']}")
    print(f"Strength: {params['strength']}, ControlNet Scale: {params['controlnet_scale']}")
    print(f"Guidance: {params['guidance']}, Canny: {params['canny_low']}/{params['canny_high']}")
    print(f"{'='*60}\n")

    # Extract Canny edges
    print(f"Extracting Canny edges from image size: {img.size}")
    print(f"Using Canny thresholds: low={params['canny_low']}, high={params['canny_high']}")
    canny = CannyDetector()(img, low_threshold=params['canny_low'], high_threshold=params['canny_high'])

    # Ensure canny matches image size
    if canny.size != img.size:
        print(f"WARNING: Canny size mismatch! Canny: {canny.size}, Img: {img.size}")
        canny = canny.resize(img.size, Image.LANCZOS)
        print(f"Resized Canny to match: {canny.size}")

    print(f"Final sizes - Image: {img.size}, Canny: {canny.size}")

    # Save Canny edges for this preset
    canny.save(output_dir / f"02_canny_edges_{preset_name}.png")
    print(f"Saved Canny edges to {output_dir / f'02_canny_edges_{preset_name}.png'}")

    # Prompts - emphasize photorealism to avoid plastic/3D look
    prompt = "high resolution photograph, professional product photography, realistic fabric texture, natural lighting, wrinkle-free garment, smooth pressed fabric, pure white background, photorealistic, DSLR camera, sharp focus"
    negative_prompt = "wrinkles, creases, folds, rumpled, 3d render, plastic, fake, artificial, cartoon, illustration, painting, gray background, colored background, shadows, oversmoothed, texture loss, unrealistic"

    with torch.inference_mode():
        if USING_IMG2IMG:
            print("Running img2img with ControlNet...")
            print(f"DEBUG: Passing to pipe - img size: {img.size}, canny size: {canny.size}")

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
    out.save(output_dir / output_filename)
    print(f"✓ Saved final output to {output_dir / output_filename}")

    # Save config
    config_path = output_dir / f"03_controlnet_output_{preset_name}_config.txt"
    with open(config_path, 'w') as f:
        f.write(f"Preset: {preset_name}\n")
        f.write(f"Description: {params['desc']}\n\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    print(f"✓ Saved config to {config_path}\n")

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    return out

# Main execution
from PIL import Image

# Prepare input image once
img = prepare_image("u2net_output/birefnet_output_2.png")

# Create output directory
output_dir = Path("controlnet_canny")
output_dir.mkdir(parents=True, exist_ok=True)

# Save composited input once
img.save(output_dir / "01_input_white_bg.png")
print(f"Saved input (white bg) to {output_dir / '01_input_white_bg.png'}\n")

# Run inference
if TEST_ALL_PRESETS:
    print(f"\n{'='*60}")
    print(f"TESTING ALL {len(PRESETS)} PRESETS")
    print(f"{'='*60}\n")

    results = {}
    for preset_name, params in PRESETS.items():
        try:
            run_inference(pipe, img, params, preset_name, output_dir)
            results[preset_name] = "SUCCESS"
        except Exception as e:
            print(f"✗ ERROR in {preset_name}: {e}")
            results[preset_name] = f"FAILED: {e}"

    # Summary
    print(f"\n{'='*60}")
    print("ALL PRESETS COMPLETE")
    print(f"{'='*60}")
    for name, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "✗"
        print(f"{symbol} {name}: {status}")
else:
    params = PRESETS[ACTIVE_PRESET]
    print(f"\n{'='*60}")
    print(f"ACTIVE PRESET: {ACTIVE_PRESET}")
    print(f"Description: {params['desc']}")
    print(f"{'='*60}\n")
    run_inference(pipe, img, params, ACTIVE_PRESET, output_dir)

print("\n===== Check controlnet_canny/ folder for results =====")
if USING_IMG2IMG:
    print("✓ Used img2img mode - original image provided appearance reference")
else:
    print("⚠ Used text2img mode - no appearance reference available")
