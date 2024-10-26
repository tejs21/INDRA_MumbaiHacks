import subprocess
import os

# Set paths
images_folder = "./images"  # Folder containing photos
output_folder = "./meshroom_output"  # Folder to save the results

# Create output directory if it doesn’t exist
os.makedirs(output_folder, exist_ok=True)

# Run Meshroom command
subprocess.run([
    "meshroom_photogrammetry",
    "--input", images_folder,
    "--output", output_folder
])
