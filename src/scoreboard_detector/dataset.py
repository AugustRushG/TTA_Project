from torch.utils.data import Dataset
import os
import cv2
import torch
from PIL import Image

class ScoreboardDataset(Dataset):
    def __init__(self, data, frame_dir, transform=None):
        self.data = data
        self.frame_dir = frame_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load_image_and_crop(frame_path, box_coordinates):
        img = cv2.imread(frame_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {frame_path}")

        x_min, y_min, x_max, y_max = box_coordinates
        h, w = img.shape[:2]

        # clip to image bounds
        x_min = max(0, min(w - 1, int(x_min)))
        x_max = max(0, min(w,     int(x_max)))
        y_min = max(0, min(h - 1, int(y_min)))
        y_max = max(0, min(h,     int(y_max)))

        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"Invalid crop box {box_coordinates} for image {frame_path} (w={w}, h={h})")

        crop = img[y_min:y_max, x_min:x_max]
        # BGR -> RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop

    def __getitem__(self, idx):
        item = self.data[idx]
        video_name = item["video_name"]
        frame_index = item["frame_index"]
        score = item["score"]
        box_coordinates = item["box_coordinates"]

        frame_path = os.path.join(
            self.frame_dir,
            f"{video_name}_frames",
            f"{frame_index:06d}.jpg"
        )

        crop_rgb = self.load_image_and_crop(frame_path, box_coordinates)

        # numpy (H,W,C) -> PIL
        img = Image.fromarray(crop_rgb)

        if self.transform is not None:
            img = self.transform(img)
        else:
            # default: PIL -> Tensor [0,1] CHW
            img = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0

        return img, score


if __name__ == "__main__":
    import json
    from torchvision import transforms
    import matplotlib.pyplot as plt

    with open("/home/august/github/TTA_Project/data/converted_scoreboard_data.json", "r") as f:
        data = json.load(f)

    # IMPORTANT: remove horizontal flip for digits; keep small rotations
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3-channel grayscale
        transforms.RandomRotation(3),
        transforms.ToTensor(),  # converts PIL -> Tensor in [0,1]
    ])

    dataset = ScoreboardDataset(data, "/home/august/github/TTA_Project/data", transform=transform)
    print(f"Dataset size: {len(dataset)}")
    # count scores
    score_counts = {}
    for item in data:
        score = item["score"]
        score_counts[score] = score_counts.get(score, 0) + 1
    print("Score distribution:", score_counts)

    random_idx = torch.randint(0, len(dataset), (1,)).item()
    img, score = dataset.__getitem__(random_idx)
    print(f"Image shape: {tuple(img.shape)}, dtype: {img.dtype}, Score: {score}")

    img_np = img.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    plt.imshow(img_np)
    plt.title(f"Score: {score}")
    plt.axis("off")
    plt.show()
