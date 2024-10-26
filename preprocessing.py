import cv2
import os

# Define the path to your folder containing images
input_folder = 'D:/hii7'
output_folder = 'D:\\ot'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the target size
target_size = (2048, 2048)

# Process each image in the folder
for filename in os.listdir(input_folder):
    # Check if the file is an image (you can adjust for different image extensions if needed)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        if img is not None:
            # Resize the image
            resized_img = cv2.resize(img, target_size)
            
            # Save the processed image to the output folder
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, resized_img)
            
            print(f"Processed and saved: {output_path}")
        else:
            print(f"Could not read image: {img_path}")

print("Image processing complete.")
