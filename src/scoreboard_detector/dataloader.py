import torch
from torch.utils.data import DataLoader




if __name__ == "__main__":
    # simple test
    from .dataset import ScoreboardDataset
    from torchvision import transforms
    import json
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3-channel grayscale
        transforms.RandomRotation(3),
        transforms.ToTensor(),  # converts PIL -> Tensor in [0,1]
    ])
    with open("/home/august/github/TTA_Project/data/converted_scoreboard_data.json", "r") as f:
        data = json.load(f)
    dataset = ScoreboardDataset(data, "/home/august/github/TTA_Project/data", transform=transform)
