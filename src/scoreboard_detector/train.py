import torch
import torch.nn.functional as F
from .model import ScoreClassifier, SmallCNN
from .dataset import ScoreboardDataset
import json
from torchvision import transforms
import random
from collections import defaultdict
from collections import Counter

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)

        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        loss_sum += loss.item() * labels.size(0)

    return loss_sum / total, correct / total


def train(model, train_loader, val_loader, device, epochs=10, lr=1e-3, weight_decay=1e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = torch.as_tensor(labels, dtype=torch.long, device=device)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(imgs)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            pred = logits.argmax(dim=1)
            running_correct += (pred == labels).sum().item()
            running_total += labels.numel()

        train_loss = running_loss / running_total
        train_acc = running_correct / running_total

        val_loss, val_acc = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_score_classifier.pt")

        print(
            f"Epoch {ep:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"best {best_val_acc:.4f}"
        )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("/home/august/github/TTA_Project/data/converted_scoreboard_data.json", "r") as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)

    # Group samples by class
    class_groups = defaultdict(list)
    for item in data:
        class_groups[item["score"]].append(item)

    train_data = []
    val_data = []

    split_ratio = 0.9

    for score, items in class_groups.items():
        random.shuffle(items)

        split_idx = int(len(items) * split_ratio)

        train_data.extend(items[:split_idx])
        val_data.extend(items[split_idx:])

    # Optional: shuffle again so batches are mixed
    random.shuffle(train_data)
    random.shuffle(val_data)

    print(f"Total: {len(data)}, Train: {len(train_data)}, Val: {len(val_data)}")
    print("Train distribution:", Counter([d["score"] for d in train_data]))
    print("Val distribution:", Counter([d["score"] for d in val_data]))


    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3-channel grayscale
        # transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # add slight blur
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(3),
        transforms.ToTensor(),  # converts PIL -> Tensor in [0,1]
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3-channel grayscale
        transforms.ToTensor(),  # converts PIL -> Tensor in [0,1]
    ])



    train_dataset = ScoreboardDataset(train_data, "/home/august/github/TTA_Project/data", transform=train_transform)
    val_dataset = ScoreboardDataset(val_data, "/home/august/github/TTA_Project/data", transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    # print one to check
    imgs, labels = next(iter(train_loader))
    print(f"Batch of images shape: {imgs.shape}, dtype: {imgs.dtype}")
    print(f"Batch of labels shape: {labels.shape}, dtype: {labels.dtype}")

    model = ScoreClassifier(num_classes=12, backbone="resnet34")
    train(model, train_loader, val_loader, device, epochs=100, lr=1e-3, weight_decay=1e-4)
    evaluate(model, val_loader, device)


if __name__ == "__main__":
    main()