# 🎨 Image Colorization & Post-Processing Tool

This project provides a **Streamlit-based web app** for automatic image **colorization** and **basic post-processing** (sharpening, blurring, undo, saving). It's built using a pretrained deep learning model from [Zhang et al. (2017)](https://richzhang.github.io/colorization/).

---

## 🚀 Features

- ✅ Upload grayscale image
- 🎨 Auto-colorize using `siggraph17` model
- 🧪 Sharpen or blur the result
- ↩️ Undo changes (1-step)
- 💾 Save processed image

---

## 🛠️ Technologies Used

- Python
- Streamlit
- PyTorch
- OpenCV
- PIL (Pillow)
- `colorizers` (Zhang's pretrained models)

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/haiderakt/image-colorization.git
cd image-colorization

# Install dependencies
pip install -r requirements.txt
