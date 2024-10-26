
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import os
import torch

# Initialize the inpainting pipeline
pipeline = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")
pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

# Directory paths
degraded_images_folder = "D:\\ot"
masks_folder = "D:\\masks"
output_folder = "D:\\otpt"

# Prompt for restoration
prompt = "A restored ancient spherical pot with smooth and intact surface, resembling its original design."

# Make sure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each image-mask pair
for file_name in os.listdir(degraded_images_folder):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        # Load the degraded image and corresponding mask
        degraded_image_path = os.path.join(degraded_images_folder, file_name)
        mask_path = os.path.join(masks_folder, f"mask_{file_name}")
        
        degraded_image = Image.open(degraded_image_path).convert("RGB")
        mask_image = Image.open(mask_path).convert("L")  # Load mask as grayscale

        # Run inpainting
        restored_image = pipeline(prompt=prompt, image=degraded_image, mask_image=mask_image).images[0]

        # Save the restored image
        output_path = os.path.join(output_folder, f"restored_{file_name}")
        restored_image.save(output_path)
        print(f"Restored image saved to: {output_path}")
