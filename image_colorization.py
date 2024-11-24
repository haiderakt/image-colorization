import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
import colorizers

def colorize_image(img_path, model):
    # Load the grayscale image
    img = cv2.imread(img_path, 0)
    
    # Resize and normalize the image
    img = cv2.resize(img, (256, 256))
    img = img / 255.0 * 100  # Scale L channel to [0, 100] for LAB conversion
    
    # Convert image to PyTorch tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    
    # Run the model's forward pass
    with torch.no_grad():
        colorized_ab = model(img_tensor).cpu().numpy()[0].transpose((1, 2, 0))
    
    # Combine L and AB channels
    lab_image = np.concatenate((img[:, :, np.newaxis], colorized_ab), axis=2).astype(np.float32)
    
    # Convert LAB to BGR
    colorized_img = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    colorized_img = np.clip(colorized_img * 255, 0, 255).astype(np.uint8)  # Scale and convert to uint8

    return colorized_img

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        colorized_img = colorize_image(file_path, model)
        cv2.imshow("Colorized Image", colorized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Load the model
model = colorizers.eccv16(pretrained=True).eval()

root = tk.Tk()
root.title("Image Colorizer")
root.geometry("100x100")

# Create a button to open an image
open_button = tk.Button(root, text="Select Image", command=open_image)
open_button.pack()

root.mainloop()
