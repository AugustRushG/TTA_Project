import torch
import torch.nn as nn
from torchvision import models

class ScoreClassifier(nn.Module):
    def __init__(self, num_classes=100, backbone="resnet18", pretrained=True, freeze_backbone=False):
        super().__init__()

        if backbone == "resnet18":
            m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            in_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            print(f"Using ResNet18 backbone, pretrained={pretrained}, feature dim={in_dim}")
        elif backbone == "resnet34":
            m = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            in_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            print(f"Using ResNet34 backbone, pretrained={pretrained}, feature dim={in_dim}")
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen, only training head layers")

        # multiple fully connected layers with dropout
        self.head = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        feat = self.backbone(x)      # [B, in_dim]
        logits = self.head(feat)     # [B, num_classes]
        return logits
    


class SmallCNN(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)
