import cv2
from .model import ScoreClassifier
import torch
from PIL import Image
from torchvision import transforms
import os
import numpy as np


def main():
    # read an image and show the box for testing
    folder_path = '/home/august/github/TTA_Project/data/train_videos/scoreboard_data/frames/26WPF_AUS_M11_G_von_Einem_AUS_v_Yuen_King_Shing_HKG_game1_frames'
    # read how many images are in the folder
    num_images = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
    print(f"Number of images in folder: {num_images}")
    # select random image from the folder
    random_image_index = np.random.randint(1, num_images + 1)
    print(f"Selected image index: {random_image_index}")
    image_path = os.path.join(folder_path, f"{random_image_index:06d}.jpg")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Failed to load image")

    # Open interactive ROI selector
    bbox = cv2.selectROI("Select Scoreboard Box", img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    x, y, w, h = bbox
    box_coordinates = [int(x), int(y), int(x + w), int(y + h)]

    print("Selected box:", box_coordinates)

    # Crop and show result
    cropped_img = img[y:y+h, x:x+w]
    cv2.imshow("Cropped Box", cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    model = ScoreClassifier(num_classes=14, backbone="resnet34")
    # load the trained model weights
    model.load_state_dict(torch.load("/home/august/github/TTA_Project/src/best_score_classifier.pt", map_location="cpu"))

    # convert cropped image from cv2 to PIL and apply the same transforms as in training
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3-channel grayscale
        transforms.ToTensor(),  # converts PIL -> Tensor in [0,1]
    ])
    cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    input_tensor = transform(cropped_pil).unsqueeze(0)  # add batch dimension


    output = model(input_tensor)
    print(f"Model output logits: {output}")
    output = torch.softmax(output, dim=1)  # convert logits to probabilities
    print(f"Model output probabilities: {output}")
    predicted_score = output.argmax(dim=1).item()
    print(f"Predicted score: {predicted_score} with confidence {output.max().item():.4f}")


if __name__ == "__main__":
    main()