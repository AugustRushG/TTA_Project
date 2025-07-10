import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Center Crop Image Example")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file.')
    return parser.parse_args()

def main(args):
    # Load the image
    img = Image.open(args.image_path).convert("RGB")

    # Define the center crop transform
    center_crop = transforms.CenterCrop(size=(224, 224))
    center_crop = transforms.Compose([
        transforms.Resize((288, 512)),  # Resize to a larger size first
    ])

    # Apply the transform
    cropped_img = center_crop(img)

    # Plot original and cropped side by side
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Center Cropped")
    plt.imshow(cropped_img)
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    args = parse_args()
    main(args)