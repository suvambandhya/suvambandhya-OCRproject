import streamlit as st
import tempfile
import os
import cv2
from Custom_OCR import load_yolo, detect_objects, ocr_images

st.title("🧾 Custom OCR with YOLOv3 + Tesseract")
st.write("Upload a lab report image to extract its contents automatically.")

# Load model
CFG_PATH = "/content/drive/MyDrive/Custom_OCR_Project/models/yolov3.cfg"
WEIGHTS_PATH = "/content/drive/MyDrive/Custom_OCR_Project/models/yolov3.weights"
NAMES_PATH = "/content/drive/MyDrive/Custom_OCR_Project/models/yolov3.obj.names"


uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        image_path = temp.name

    st.image(image_path, caption="Uploaded Image", use_container_width=True)

    if st.button("Run OCR"):
        with st.spinner("Processing..."):
            cropped_images, labels = detect_objects(image_path, net, output_layers, classes)
            texts = ocr_images(cropped_images)

        st.success("✅ OCR completed!")
        for label, text in zip(labels, texts):
            st.subheader(f"Detected Field: {label}")
            st.text_area("Extracted Text", text, height=80)
