import cv2
import numpy as np
import torch
import streamlit as st
from PIL import Image
import colorizers
import io

# Load pretrained model
model = colorizers.siggraph17(pretrained=True).eval()

# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'history' not in st.session_state:
    st.session_state.history = []

# Helper to display image
def display_image_cv2(image):
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

# Helper to colorize image
def colouring_image(file, model):
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    original = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_GRAY2BGR)
    
    img = cv2.resize(img, (256, 256)) / 255.0 * 100
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    
    with torch.no_grad():
        ab = model(img_tensor).cpu().numpy()[0].transpose((1, 2, 0))

    lab = np.concatenate((img[:, :, np.newaxis], ab), axis=2)
    bgr = cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_Lab2BGR)
    bgr = np.clip(bgr * 255, 0, 255).astype(np.uint8)
    return bgr, original

# Streamlit UI
st.set_page_config(page_title="Image Colorizer", layout="wide")
st.title("🎨 Image Colorization and Post-Processing Tool")

uploaded_file = st.file_uploader("Upload a grayscale image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file:
    colorized, original = colouring_image(uploaded_file, model)
    st.session_state.processed_image = colorized.copy()
    st.session_state.original_image = original
    st.session_state.history = [colorized.copy()]

    st.subheader("Preview:")
    col1, col2 = st.columns(2)
    with col1:
        st.image(display_image_cv2(original), caption="Original Image", use_column_width=True)
    with col2:
        st.image(display_image_cv2(colorized), caption="Colorized Image", use_column_width=True)

    st.markdown("---")

    # Buttons
    colA, colB, colC, colD = st.columns(4)

    with colA:
        if st.button("🔪 Sharpen"):
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            sharpened = cv2.filter2D(st.session_state.processed_image, -1, kernel)
            st.session_state.history.append(st.session_state.processed_image.copy())
            st.session_state.processed_image = sharpened
            st.image(display_image_cv2(sharpened), caption="Sharpened Image", use_column_width=True)

    with colB:
        if st.button("💧 Blur"):
            blurred = cv2.GaussianBlur(st.session_state.processed_image, (15, 15), 0)
            st.session_state.history.append(st.session_state.processed_image.copy())
            st.session_state.processed_image = blurred
            st.image(display_image_cv2(blurred), caption="Blurred Image", use_column_width=True)

    with colC:
        if st.button("↩️ Undo"):
            if len(st.session_state.history) > 1:
                st.session_state.history.pop()
                st.session_state.processed_image = st.session_state.history[-1]
                st.image(display_image_cv2(st.session_state.processed_image), caption="Undo Applied", use_column_width=True)
            else:
                st.warning("Nothing to undo.")

    with colD:
        if st.session_state.processed_image is not None:
            buf = cv2.imencode(".jpg", st.session_state.processed_image)[1].tobytes()
            st.download_button(label="💾 Save Image", data=buf, file_name="colorized.jpg", mime="image/jpeg")
