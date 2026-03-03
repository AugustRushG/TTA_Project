import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from .model import ScoreClassifier, SmallCNN
from .dataset import ScoreboardDataset
import json
from torchvision import transforms
import random
from collections import defaultdict
from collections import Counter
import json

@torch.no_grad()
def evaluate(model, loader, device, num_classes=14):
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0

    # confusion matrix: rows = true, cols = predicted
    conf_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)

        logits = model(imgs)
        loss = F.cross_entropy(logits, labels)

        preds = logits.argmax(dim=1)

        # accumulate stats
        correct += (preds == labels).sum().item()
        total += labels.numel()
        loss_sum += loss.item() * labels.size(0)

        # update confusion matrix
        for t, p in zip(labels.view(-1), preds.view(-1)):
            conf_mat[t.long(), p.long()] += 1

    avg_loss = loss_sum / total
    acc = correct / total

    print(f"\nEvaluation Results")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Accuracy: {acc:.4f}\n")

    print("Confusion Matrix (rows=true, cols=pred):")
    print(conf_mat)

    # per-class accuracy
    print("\nPer-class accuracy:")
    for i in range(num_classes):
        class_total = conf_mat[i].sum().item()
        if class_total > 0:
            class_acc = conf_mat[i, i].item() / class_total
            print(f"Class {i:2d}: {class_acc:.4f} ({conf_mat[i, i].item()}/{class_total})")
        else:
            print(f"Class {i:2d}: No samples")

    return avg_loss, acc, conf_mat


def train(model, train_loader, val_loader, device, epochs=10, lr=1e-3, weight_decay=1e-4):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

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

            with autocast(device.type):
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

        val_loss, val_acc, val_confusion_matrix = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_score_classifier.pt")

        print(
            f"Epoch {ep:03d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
            f"best {best_val_acc:.4f}"
        )


def split_data(data):
    # Group by class
    class_groups = defaultdict(list)
    for item in data:
        class_groups[item["score"]].append(item)

    train_data, val_data, test_data = [], [], []

    train_ratio = 0.8
    val_ratio   = 0.1
    test_ratio  = 0.1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    for score, items in class_groups.items():
        random.shuffle(items)
        n = len(items)

        # base split sizes
        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        n_test  = n - n_train - n_val  # remainder goes to test

        # ---- handle tiny classes more safely ----
        # If we have at least 3 samples, try to ensure val and test get >=1
        if n >= 3:
            if n_val == 0:
                n_val = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_test -= 1

            if n_test == 0:
                n_test = 1
                if n_train > 1:
                    n_train -= 1
                else:
                    n_val -= 1

        # Final safety clamp (avoid negative)
        n_train = max(n_train, 0)
        n_val   = max(n_val, 0)
        n_test  = max(n_test, 0)

        # Recompute if rounding messed up
        # ensure total matches n
        while n_train + n_val + n_test > n:
            n_train = max(0, n_train - 1)
        while n_train + n_val + n_test < n:
            n_test += 1

        train_data.extend(items[:n_train])
        val_data.extend(items[n_train:n_train + n_val])
        test_data.extend(items[n_train + n_val:])

    # Shuffle splits so batches are mixed
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    print(f"Total: {len(data)}")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    print("Train distribution:", Counter([d["score"] for d in train_data]))
    print("Val distribution:",   Counter([d["score"] for d in val_data]))
    print("Test distribution:",  Counter([d["score"] for d in test_data]))

    return train_data, val_data, test_data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open("/home/august/github/TTA_Project/data/train_videos/scoreboard_data/converted_scoreboard_data.json", "r") as f:
        data = json.load(f)

    random.seed(42)
    random.shuffle(data)

    train_data, val_data, test_data = split_data(data)


    train_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3-channel grayscale
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),  # add slight blur
        transforms.RandomHorizontalFlip(p=0.5),  # random horizontal flip
        transforms.RandomVerticalFlip(p=0.5),    # random vertical flip
        transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.95, 1.05)),  # small random affine
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(3),
        transforms.ToTensor(),  # converts PIL -> Tensor in [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std =[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.Grayscale(num_output_channels=3),  # convert to 3-channel grayscale
        transforms.ToTensor(),  # converts PIL -> Tensor in [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std =[0.229, 0.224, 0.225])
    ])



    train_dataset = ScoreboardDataset(train_data, "/home/august/github/TTA_Project/data/train_videos/scoreboard_data/frames", transform=train_transform)
    val_dataset = ScoreboardDataset(val_data, "/home/august/github/TTA_Project/data/train_videos/scoreboard_data/frames", transform=val_transform)
    test_dataset = ScoreboardDataset(test_data, "/home/august/github/TTA_Project/data/train_videos/scoreboard_data/frames", transform=val_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    # print one to check
    imgs, labels = next(iter(train_loader))
    print(f"Batch of images shape: {imgs.shape}, dtype: {imgs.dtype}")
    print(f"Batch of labels shape: {labels.shape}, dtype: {labels.dtype}")

    model = ScoreClassifier(num_classes=14, backbone="resnet34", freeze_backbone=False)
    train(model, train_loader, val_loader, device, epochs=100, lr=1e-4, weight_decay=1e-4)
    print("Training done, loading best model for test evaluation")
    model.load_state_dict(torch.load("best_score_classifier.pt", map_location=device))
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()