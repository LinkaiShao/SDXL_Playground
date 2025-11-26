#!/bin/bash
# Virtual Try-On using OOTDiffusion
# Simple wrapper to run OOTDiffusion with example images

cd "$(dirname "$0")/models/OOTDiffusion/run"

MODEL_IMG="../run/examples/model/01008_00.jpg"
GARMENT_IMG="../run/examples/garment/00055_00.jpg"

echo "Running OOTDiffusion virtual try-on..."
echo "Model: $MODEL_IMG"
echo "Garment: $GARMENT_IMG"
echo ""

python run_ootd.py \
  --model_path "$MODEL_IMG" \
  --cloth_path "$GARMENT_IMG" \
  --model_type hd \
  --scale 2.0 \
  --sample 4 \
  --step 20 \
  --seed 42

echo ""
echo "Done! Check models/OOTDiffusion/run/images_output/ for results"
