import cv2
from .model import ScoreClassifier
import torch
from PIL import Image
from torchvision import transforms


def main():
    # read an image and show the box for testing
    image_path = '/home/august/github/TTA_Project/data/25WPF_JPN_M1_G_Shikai_JPN_v_Savinov_AUS_game1_frames/007913.jpg'

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

    model = ScoreClassifier(num_classes=12, backbone="resnet34")
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
    predicted_score = output.argmax(dim=1).item()
    print(f"Predicted score: {predicted_score}")


if __name__ == "__main__":
    main()