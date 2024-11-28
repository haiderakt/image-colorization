import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
import colorizers

processed_image = None

def sharpening_filter():
    global processed_image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(processed_image, -1, kernel)
    cv2.imshow("Sharpened Image", sharpened_img)
    cv2.waitKey(0)
    cv2.destroyWindow("Sharpened Image")


def colorize_image(img_path, model):
    #Load the grayscale image
    img = cv2.imread(img_path, 0)
    if img is None:
        raise ValueError("Could not load the image. Please select a valid file.")

    #To return original image for display
    original = cv2.resize(img, (256, 256))

    #Resize and normalize the image
    img = cv2.resize(img, (256, 256))
    img = img / 255.0 * 100  # Scale L channel to [0, 100] for LAB conversion

    #Convert image to PyTorch tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    #Run the model's forward pass
    with torch.no_grad():
        colorized_ab = model(img_tensor).cpu().numpy()[0].transpose((1, 2, 0))

    #Combine L and AB channels
    lab_image = np.concatenate((img[:, :, np.newaxis], colorized_ab), axis=2).astype(np.float32)

    #Convert LAB to BGR
    colorized_img = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    colorized_img = np.clip(colorized_img * 255, 0, 255).astype(np.uint8)  # Scale and convert to uint8

    return colorized_img, original


def open_image():
    global processed_image
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        try:
            colorized_img, original_img = colorize_image(file_path, model)
            processed_image = colorized_img
            cv2.imshow("Colorized Image", colorized_img)
            cv2.imshow("Original Image", original_img)
        except Exception as e:
            print(f"Error: {e}")


#Load the model
model = colorizers.eccv16(pretrained=True).eval()

#Tkinter GUI
root = tk.Tk()
root.title("Image Colorization Tool")
root.geometry("400x300")

# Buttons for functionality
open_button = tk.Button(root, text="Select Image & Process", height=2, font=("Arial", 14), command=open_image)
open_button.pack(pady=20)

sharpen_button = tk.Button(root, text="Apply Sharpening Filter", height=2, font=("Arial", 14), command=sharpening_filter)
sharpen_button.pack(pady=20)

root.mainloop()