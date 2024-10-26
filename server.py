
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import base64


def select_jpg_files_with_prefix(directory, prefix):
    selected_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg') and filename.startswith(prefix):
            selected_files.append(filename)
    return selected_files

directory_path = 'D:/otpt'  # Replace with your directory path





# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained model and modify it to output features
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Remove last layer
model = model.to(device)
model.eval()

# Image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    """Extract features from an image using the pretrained model."""
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model(input_tensor).cpu().numpy().flatten()
        # Normalize the feature vector to unit length
        norm_features = features / np.linalg.norm(features)
        return norm_features
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Step 1: Extract and store features for each image in the folder
def build_feature_database(folder_path):
    feature_db = {}
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):
            features = extract_features(img_path)
            if features is not None:
                feature_db[img_name] = features
            else:
                print(f"Failed to extract features for {img_path}")
    return feature_db

# Step 2: Identify the most similar object for a test image
def find_similar_object(test_image_path, feature_db):
    test_features = extract_features(test_image_path)
    if test_features is None:
        print("Failed to extract features for test image.")
        return None, None
    
    best_image = None
    best_similarity = -1
    
    for img_name, features in feature_db.items():
        similarity = cosine_similarity([test_features], [features])[0][0]
        
        # Debugging output
        
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_image = img_name
    
    return best_image, best_similarity

# Paths
folder_path = 'D:/md'
test_image_path = 'ref.jpg'  # Replace with your test image path

# Run the program
feature_db = build_feature_database(folder_path)

from flask_cors import CORS
from flask import Flask, request,jsonify, send_file
import os

app = Flask(__name__)
CORS(app)
upload_folder = 'D:\\uploads1'
os.makedirs(upload_folder, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_image():
    images = {}
    if 'image' not in request.files:
        print("No file part in the request")
        return "No file part", 400
    
    file = request.files['image']
    if file.filename == '':
        print("No selected file in the request")
        return "No selected file", 400
    
    file_path = os.path.join(upload_folder, file.filename)
    file.save(file_path)
    print(f"File saved at {file_path}")
    
    # Load .obj and .mtl files as base64
    try:
        with open('D:/3d/texturedMesh.obj', 'rb') as obj_file:
            obj_data = base64.b64encode(obj_file.read()).decode('utf-8')
        with open('D:/3d/texturedMesh.mtl', 'rb') as mtl_file:
            mtl_data = base64.b64encode(mtl_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error reading .obj or .mtl files: {e}")
        obj_data = None
        mtl_data = None
    
    if feature_db:
        matched_image, similarity_score = find_similar_object(file_path, feature_db)
        if matched_image:
            matched_image_name = os.path.splitext(matched_image)[0]
            imgs = select_jpg_files_with_prefix(directory_path, matched_image_name)
            print(f"Found matched images: {imgs}")
            
            for filepath in imgs:
                filepath1 = os.path.join(directory_path, filepath)
                with open(filepath1, "rb") as img_file:
                    images[filepath] = base64.b64encode(img_file.read()).decode('utf-8')
            
            resp = {
                "M": matched_image_name,
                "i": images,
                "obj_data": obj_data,
                "mtl_data": mtl_data
            }
        else:
            print("No matching image found in the feature database.")
            resp = {"message": "No matching image found."}
    else:
        print("Feature database is empty.")
        resp = {"message": "Feature database is empty."}

      # Debug print for response data
    return jsonify(resp), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # Use 5000 or change to any available por