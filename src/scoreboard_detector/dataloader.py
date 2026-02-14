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
    train_loader, val_loader = make_loaders(dataset, batch_size=5, num_workers=4)
    # print one to check
    imgs, labels = next(iter(train_loader))
    print(f"Batch of images shape: {imgs.shape}, dtype: {imgs.dtype}")
    print(f"Batch of labels shape: {labels.shape}, dtype: {labels.dtype}")