import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, Label, messagebox
from PIL import Image, ImageTk
import colorizers
from tktooltip import ToolTip

processed_image = None
original_image = None
original_photo = None
filtered_photo = None
history = []

def displaying_image(image, side):
    global original_photo, filtered_photo
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    if side == "left":
        original_photo = img
        original_label.config(image=original_photo)
    elif side == "right":
        filtered_photo = img
        filtered_label.config(image=filtered_photo)

def boost_clarity():
    global processed_image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_img = cv2.filter2D(processed_image, -1, kernel)
    history.append(processed_image.copy())
    displaying_image(sharpened_img, "right")

def blurring_filter():
    global processed_image
    blurred_img = cv2.GaussianBlur(processed_image, (15, 15), 0)
    history.append(processed_image.copy())
    displaying_image(blurred_img, "right")


def colouring_image(img_path, model):
    global processed_image, original_image
    img = cv2.imread(img_path, 0)

    # Save original image for display
    original_image = cv2.cvtColor(cv2.resize(img, (256, 256)), cv2.COLOR_GRAY2BGR)

    # Resize and normalize the image
    img = cv2.resize(img, (256, 256))
    img = img / 255.0 * 100  # Scale L channel to [0, 100] for LAB conversion

    # Convert image to PyTorch tensor and add batch and channel dimensions
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

    # Run the model's forward pass
    with torch.no_grad():
        colorized_ab = model(img_tensor).cpu().numpy()[0].transpose((1, 2, 0))

    img_with_new_axis = img[:, :, np.newaxis]
    combining_channels = np.concatenate((img_with_new_axis, colorized_ab), axis=2)
    lab_image = combining_channels.astype(np.float32)

    # Convert LAB to BGR
    colorized_img = cv2.cvtColor(lab_image, cv2.COLOR_Lab2BGR)
    colorized_img = np.clip(colorized_img * 255, 0, 255).astype(np.uint8)

    return colorized_img, original_image

def open_image():
    global processed_image, original_image

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        
        colorized_img, original_img = colouring_image(file_path, model)

        processed_image = colorized_img
        original_image = original_img

        history.append(processed_image.copy())

        displaying_image(original_image, "left")
        displaying_image(processed_image, "right")

def save_image():
    global processed_image
    if processed_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
        if file_path:
            cv2.imwrite(file_path, processed_image)
            messagebox.showinfo("Save Image", "Image saved")
    else:
        messagebox.showwarning("Save Image", "Nothing to save")

def undo():
    global processed_image
    if history:
        processed_image = history.pop()
        displaying_image(processed_image, "right")



# model = colorizers.eccv16(pretrained=True).eval()
model = colorizers.siggraph17(pretrained=True).eval()

root = tk.Tk()
root.title("Image Colorization Tool")
#root.geometry("700x400")
root.geometry("800x400")
root.configure(bg="#111142")

label_top = tk.Label(root, text="Image Colorization and Post Processing", fg="white", bg="#111142", font=("Arial", 16))
label_top.pack(pady=10)

left_frame = tk.Frame(root, width=400, height=400, bg="black")
left_frame.pack(side="left", fill="both", expand=True)

right_frame = tk.Frame(root, width=400, height=400, bg="black")
right_frame.pack(side="right", fill="both", expand=True)

original_label = Label(left_frame)
original_label.pack(padx=10, pady=10)

filtered_label = Label(right_frame)
filtered_label.pack(padx=10, pady=10)

btn_frame = tk.Frame(root, bg="#111142")
btn_frame.pack(side="bottom", pady=20)

open_button = tk.Button(btn_frame, text="Select Image & Process", height=2, width=20, bg="#872646", font=("Arial", 12), command=open_image)
open_button.grid(row=0, column=0, padx=10)
ToolTip(open_button, "Select the image you want to colorize", delay=0.5)

sharpen_button = tk.Button(btn_frame, text="Sharpen", height=2, width=12, font=("Arial", 12), command=boost_clarity)
sharpen_button.grid(row=0, column=1, padx=5)
ToolTip(sharpen_button, "Make the image sharper", delay=0.5)

blur_button = tk.Button(btn_frame, text="Blur", height=2, width=12, font=("Arial", 12), command=blurring_filter)
blur_button.grid(row=0, column=2, padx=5)
ToolTip(blur_button, "Add blurr to your photo", delay=0.5)

save_button = tk.Button(btn_frame, text="Save Image", height=2, width=12, font=("Arial", 12), command=save_image)
save_button.grid(row=0, column=3, padx=5)
ToolTip(save_button, "Save your photo", delay=0.5)

undo_button = tk.Button(btn_frame, text="Undo", height=2, width=12, font=("Arial", 12), command=undo)
undo_button.grid(row=0, column=4, padx=5)
ToolTip(undo_button, "Removes all filters", delay=0.5)

root.mainloop()