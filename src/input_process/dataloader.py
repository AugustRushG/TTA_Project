from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
import torch
import cv2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class FrameClipDataset(Dataset):
    def __init__(self, frame_dir, window_size=100, stride=50, transform=None):


        self.frame_paths = sorted([
            os.path.join(frame_dir, f)
            for f in os.listdir(frame_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        self.window_size = window_size
        self.stride = stride
        self.transform = transform
        self.dummy_shape = (224,224)

        self.num_clips = max(0, (len(self.frame_paths) - window_size) // stride + 1)
    
    def _extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break
            if self.resize_to:
                frame = cv2.resize(frame, self.resize_to)
            out_path = os.path.join(self.frame_dir, f"{idx:05d}.jpg")
            cv2.imwrite(out_path, frame)
            idx += 1
        cap.release()

    def __len__(self):
        return self.num_clips

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.window_size
        clip_paths = self.frame_paths[start:end]
        
        frames = []
        for path in clip_paths:
            if os.path.exists(path):
                frame = read_image(path).float() / 255.
            else:
                frame = torch.zeros(*self.dummy_shape)
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)

        clip_tensor = torch.stack(frames)  # (T, C, H, W)
        return {'frames': clip_tensor, 'start_idx': start, 'end_idx': end - 1}


if __name__ == "__main__":
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.CenterCrop(size=(224, 224)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    dataset = FrameClipDataset("/home/s224705071/github/TTA_Project/data/Game_4_frames", window_size=100, stride=50, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    for data in dataloader:
        clips = data['frames']
        start_idx = data['start_idx']
        end_idx = data['end_idx']
        print(f"Clip shape: {clips.shape}, Start index: {start_idx}, End index: {end_idx}")
    