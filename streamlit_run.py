import streamlit as st
import requests
import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO

# Set server URL
FLASK_SERVER_URL = "http://192.168.137.242:5000/upload"  # Update with your server's URL if needed

st.title("Image Uploader and 3D Viewer")

# Step 1: Capture or upload an image
st.header("Capture or Upload an Image")

# Initialize webcam capture
cap = cv2.VideoCapture(0)

if cap.isOpened():
    ret, frame = cap.read()
    cap.release()  # Close the webcam immediately after capturing
    if ret:
        # Display the captured frame
        st.image(frame, channels="BGR", caption="Captured Image")
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    st.write("Error accessing webcam")

# Image upload option for fallback
uploaded_file = st.file_uploader("Or upload an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

if img is not None:
    # Step 2: Convert to base64 for API upload
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Step 3: Send image to Flask server
    st.write("Sending image to server...")
    response = requests.post(FLASK_SERVER_URL, files={"image": buffered.getvalue()})
    
    if response.status_code == 200:
        # Parse server response
        data = response.json()
        
        # Display matched images
        st.header("Regenrated/Restored Images")
        images = data.get("i", {})
        for img_name, img_data in images.items():
            img_bytes = base64.b64decode(img_data)
            st.image(Image.open(BytesIO(img_bytes)), caption=img_name)

        # Display 3D model data
        st.header("3D Model")
        obj_data = data.get("obj_data")
        mtl_data = data.get("mtl_data")
        if obj_data and mtl_data:
            st.write("3D model files received. Rendering is not directly supported in Streamlit, but you can download them for external viewing.")
            st.download_button(label="Download .obj file", data=base64.b64decode(obj_data), file_name="model.obj")
            st.download_button(label="Download .mtl file", data=base64.b64decode(mtl_data), file_name="model.mtl")
        else:
            st.write("3D model data not available.")
    else:
        st.write("Error from server:", response.text)

st.write("End of processing.")
