#!/home/link/venvs/ootd/bin/python3
"""
Virtual Try-On using OOTDiffusion
Combines a model image and a garment image to generate try-on results
"""
from pathlib import Path
import sys
import os
from PIL import Image

OOTD_ROOT = Path(__file__).absolute().parent / "models/OOTDiffusion"
# Change to OOTDiffusion directory for imports to work
os.chdir(str(OOTD_ROOT))
sys.path.insert(0, str(OOTD_ROOT))

from preprocess.openpose.run_openpose import OpenPose
from preprocess.humanparsing.run_parsing import Parsing
from ootd.inference_ootd_hd import OOTDiffusionHD
from run.utils_ootd import get_mask_location

# ====== CONFIG ======
GPU_ID = 0
MODEL_TYPE = "hd"  # "hd" or "dc" (hd only supports upperbody)
CATEGORY = 0  # 0:upperbody; 1:lowerbody; 2:dress
IMAGE_SCALE = 2.0
N_STEPS = 20
N_SAMPLES = 4
SEED = 42

# Paths - using first available examples
MODEL_IMAGE = OOTD_ROOT / "run/examples/model/01008_00.jpg"
GARMENT_IMAGE = OOTD_ROOT / "run/examples/garment/00055_00.jpg"
OUTPUT_DIR = Path(__file__).absolute().parent / "oot_try_output"

# ====== SETUP ======
OUTPUT_DIR.mkdir(exist_ok=True)

category_dict = ['upperbody', 'lowerbody', 'dress']
category_dict_utils = ['upper_body', 'lower_body', 'dresses']

print("Loading models...")
openpose_model = OpenPose(GPU_ID)
parsing_model = Parsing(GPU_ID)

if MODEL_TYPE == "hd":
    model = OOTDiffusionHD(GPU_ID)
else:
    raise ValueError("Only 'hd' model_type is configured")

if MODEL_TYPE == 'hd' and CATEGORY != 0:
    raise ValueError("model_type 'hd' requires category == 0 (upperbody)!")

# ====== PROCESS ======
print(f"Model image: {MODEL_IMAGE}")
print(f"Garment image: {GARMENT_IMAGE}")

cloth_img = Image.open(GARMENT_IMAGE).resize((768, 1024))
model_img = Image.open(MODEL_IMAGE).resize((768, 1024))

print("Detecting pose...")
keypoints = openpose_model(model_img.resize((384, 512)))

print("Parsing human segments...")
model_parse, _ = parsing_model(model_img.resize((384, 512)))

print("Generating mask...")
mask, mask_gray = get_mask_location(MODEL_TYPE, category_dict_utils[CATEGORY], model_parse, keypoints)
mask = mask.resize((768, 1024), Image.NEAREST)
mask_gray = mask_gray.resize((768, 1024), Image.NEAREST)

masked_vton_img = Image.composite(mask_gray, model_img, mask)
masked_vton_img.save(OUTPUT_DIR / 'mask.jpg')
print(f"Saved mask to {OUTPUT_DIR / 'mask.jpg'}")

print(f"Generating {N_SAMPLES} try-on images...")
images = model(
    model_type=MODEL_TYPE,
    category=category_dict[CATEGORY],
    image_garm=cloth_img,
    image_vton=masked_vton_img,
    mask=mask,
    image_ori=model_img,
    num_samples=N_SAMPLES,
    num_steps=N_STEPS,
    image_scale=IMAGE_SCALE,
    seed=SEED,
)

print("Saving results...")
for idx, image in enumerate(images):
    output_path = OUTPUT_DIR / f'out_{MODEL_TYPE}_{idx}.png'
    image.save(output_path)
    print(f"  Saved: {output_path}")

print(f"\nDone! All outputs saved to {OUTPUT_DIR}/")
