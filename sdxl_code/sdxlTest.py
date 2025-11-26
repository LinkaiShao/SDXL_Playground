import torch
from diffusers import StableDiffusionXLPipeline

file_path = "/home/link/sdxl-base/sd_xl_base_1.0.safetensors"  # your file
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

pipe = StableDiffusionXLPipeline.from_single_file(file_path, torch_dtype=dtype).to(device)

# (optional; safe if not available)
try: pipe.enable_vae_tiling()
except: pass
try: pipe.enable_xformers_memory_efficient_attention()
except: pass

img = pipe(
    prompt="studio product photo of a minimalist purple hoodie on a mannequin, detailed fabric texture",
    negative_prompt="low quality, artifacts",
    num_inference_steps=20, guidance_scale=5.5, height=1024, width=1024
).images[0]
img.save("sdxl_out.png")
print("Saved sdxl_out.png")
