#!/usr/bin/env python3
"""
SDXL Inpainting with Hyperparameter Grid Search
Tests multiple parameter combinations to find optimal settings for minimal aura
"""

from pathlib import Path
import os, sys
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import torch
from diffusers import AutoPipelineForInpainting
import json
from datetime import datetime

# ================== PATHS ==================
OOTD_ROOT = Path("models/OOTDiffusion").resolve()
IMAGE_PATH = Path("models/OOTDiffusion/run/examples/model/01008_00.jpg").resolve()
SDXL_DIR   = Path("models/sdxl-inpaint").resolve()
OUTPUT_DIR = Path("inpainting_try_output").resolve()
PARAMS_FILE = Path("inpainting_current_params.json").resolve()

REGION = "upper"  # "upper", "lower", or "dress"
BASE_PROMPT = "light beige color ribbed turtleneck sweater, minimalist, studio lighting, photorealistic, natural skin tones"
BASE_NEG = "logo, text, watermark, extra limbs, blurry, artifacts"
# ==========================================

# --- OOTD imports ---
original_dir = Path.cwd()
os.chdir(str(OOTD_ROOT))
sys.path.insert(0, str(OOTD_ROOT))
from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from run.utils_ootd import get_mask_location
os.chdir(str(original_dir))

# --- Helper Functions ---
def create_mask(mask_img: Image.Image, size, dilate_px=10, feather=5.0):
    """Create binary mask with dilation and feathering"""
    from scipy.ndimage import binary_dilation

    m = mask_img.convert("L").resize(size, Image.NEAREST)
    m = m.point(lambda v: 255 if v >= 200 else 0)

    if dilate_px > 0:
        arr = np.array(m) == 255
        arr = binary_dilation(arr, iterations=dilate_px)
        m = Image.fromarray((arr * 255).astype(np.uint8), mode="L")

    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather))

    return m

def preprocess_lighten(base_img: Image.Image, mask: Image.Image, factor=1.2):
    """Lighten masked region"""
    base_arr = np.array(base_img, dtype=np.float32)
    mask_arr = np.array(mask.convert("L"), dtype=np.float32) / 255.0
    lightened = base_arr * (1 + (factor - 1) * mask_arr[:,:,None])
    return Image.fromarray(np.clip(lightened, 0, 255).astype(np.uint8))

def simple_aura_fix(final_img: Image.Image, base_img: Image.Image, mask: Image.Image):
    """Simple targeted aura fix - only fix bright pixels near edges"""
    final_arr = np.array(final_img, dtype=np.float32)
    base_arr = np.array(base_img, dtype=np.float32)
    mask_arr = np.array(mask.convert("L"))

    # Find 5px boundary zone
    from scipy.ndimage import distance_transform_edt, binary_erosion
    mask_bin = mask_arr > 128
    inner = binary_erosion(mask_bin, iterations=5)
    boundary = mask_bin & ~inner

    # Detect aura: bright in final, darker in base
    final_bright = np.mean(final_arr, axis=2)
    base_bright = np.mean(base_arr, axis=2)
    is_aura = boundary & (final_bright > 220) & (base_bright < 180)

    # Darken only aura pixels
    if is_aura.any():
        final_arr[is_aura] = final_arr[is_aura] * 0.7

    return Image.fromarray(np.clip(final_arr, 0, 255).astype(np.uint8))

def save_debug_strip(base, mask, preprocessed, gen, final, output_path, param_name):
    """Create debug visualization strip"""
    W, H = base.size
    m_vis = mask.convert("L").point(lambda v: 255 if v > 0 else 0).convert("RGB")

    strip = Image.new("RGB", (W*5, H), (0,0,0))
    strip.paste(base, (0,0))
    strip.paste(m_vis, (W,0))
    strip.paste(preprocessed, (W*2,0))
    strip.paste(gen,  (W*3,0))
    strip.paste(final,(W*4,0))

    # Add label
    from PIL import ImageDraw
    draw = ImageDraw.Draw(strip)
    draw.text((10, 10), param_name, fill=(255, 255, 0))

    strip.save(output_path)

def run_inpainting_with_params(pipe, base, raw_mask, W, H, param_set, param_name):
    """Run inpainting with specific parameter set"""
    print(f"\n{'='*60}")
    print(f"Running: {param_name}")
    print(f"Description: {param_set['desc']}")
    print(f"{'='*60}")

    # Create mask
    mask = create_mask(raw_mask, (W, H),
                      dilate_px=param_set['mask_dilate'],
                      feather=param_set['feather'])

    # Optional preprocessing
    inpaint_base = base
    if param_set['shadow_preprocess']:
        print(f"  Preprocessing: lightening by {param_set['lighten_factor']}")
        inpaint_base = preprocess_lighten(base, mask, param_set['lighten_factor'])

    # Build prompts
    prompt = BASE_PROMPT + param_set['prompt_suffix']
    neg_prompt = BASE_NEG + param_set['neg_suffix']

    print(f"  Mask: dilate={param_set['mask_dilate']}px, feather={param_set['feather']}px")
    print(f"  Pipeline: steps={param_set['steps']}, guidance={param_set['guidance']}, strength={param_set['strength']}")

    # Generate
    gen = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=inpaint_base,
        mask_image=mask,
        num_inference_steps=param_set['steps'],
        guidance_scale=param_set['guidance'],
        strength=param_set['strength'],
        width=W, height=H,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0].convert("RGB")

    # Composite
    final = Image.composite(gen, base, mask)

    # Simple aura fix (always apply)
    final = simple_aura_fix(final, base, mask)

    # Save outputs
    output_path = OUTPUT_DIR / f"{param_name}_result.png"
    debug_path = OUTPUT_DIR / f"{param_name}_debug.png"
    config_path = OUTPUT_DIR / f"{param_name}_config.txt"

    final.save(output_path)
    save_debug_strip(base, mask, inpaint_base, gen, final, debug_path, param_name)

    # Save config
    with open(config_path, 'w') as f:
        f.write(f"Parameter Set: {param_name}\n")
        f.write(f"Description: {param_set['desc']}\n")
        f.write(f"Timestamp: {datetime.now()}\n\n")
        f.write(json.dumps(param_set, indent=2))

    print(f"  Saved: {output_path.name}")
    return final

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 0) Load parameter sets from JSON
    print(f"Loading parameter sets from: {PARAMS_FILE}")
    with open(PARAMS_FILE, 'r') as f:
        PARAM_SETS = json.load(f)
    print(f"Loaded {len(PARAM_SETS)} parameter configurations")

    # 1) Load base image
    base = Image.open(IMAGE_PATH).convert("RGB")
    W, H = base.size
    print(f"Loaded image: {W}x{H}")

    # 2) Human parsing + pose
    print("\nRunning OOTD parsing...")
    parsing_model = Parsing(gpu_id=0)
    openpose_model = OpenPose(gpu_id=0)

    small = base.resize((384, 512))
    parse_map, _ = parsing_model(small)
    kpts = openpose_model(small)

    tag = {"upper": "upper_body", "lower": "lower_body", "dress": "dresses"}[REGION]
    raw_mask, _ = get_mask_location("dc", tag, parse_map, kpts)

    # 3) Load SDXL pipeline ONCE
    print("\nLoading SDXL Inpainting pipeline...")
    pipe = AutoPipelineForInpainting.from_pretrained(
        str(SDXL_DIR), torch_dtype=torch.float16
    ).to("cuda")
    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()

    # 4) Run all parameter sets
    print(f"\nTesting {len(PARAM_SETS)} parameter configurations...")
    results = {}

    for param_name, param_set in PARAM_SETS.items():
        try:
            result = run_inpainting_with_params(pipe, base, raw_mask, W, H, param_set, param_name)
            results[param_name] = "SUCCESS"
        except Exception as e:
            print(f"  ERROR: {e}")
            results[param_name] = f"FAILED: {e}"

    # 5) Summary
    print(f"\n{'='*60}")
    print("GRID SEARCH COMPLETE")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"  {name}: {status}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("Compare all *_result.png files to find best parameters!")

if __name__ == "__main__":
    main()
