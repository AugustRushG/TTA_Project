import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torchvision.transforms as transforms
import torch
from glob import glob  # At the top of your script
from torchvision.io import read_image


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def extract_frames(video_path, resize_to=None):
    frame_dir = os.path.splitext(video_path)[0] + "_frames"
    if os.path.exists(frame_dir):
        print(f"Frame directory {frame_dir} already exists, skipping extraction.")
        return frame_dir
    else:
        print(f"Creating frame directory {frame_dir}...")
        os.makedirs(frame_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if resize_to:
            frame = cv2.resize(frame, resize_to)
        out_path = os.path.join(frame_dir, f"{idx:06d}.jpg")
        cv2.imwrite(out_path, frame)
        idx += 1
    cap.release()

    print(f"Extracted {idx} frames to {frame_dir}")
    return frame_dir



class VideoLoader:
    def __init__(self, video_path, window_size=100, stride=50, rgb=True, pad_mode='zero', transform=None, frame_dir=None, resize_to=None):
        self.video_path = video_path
        self.window_size = window_size
        self.stride = stride
        self.rgb = rgb
        self.pad_mode = pad_mode
        self.transform = transform
        self.resize_to = resize_to

        # Frame extraction directory
        self.frame_dir = frame_dir or os.path.splitext(video_path)[0] + "_frames"
        os.makedirs(self.frame_dir, exist_ok=True)

        # Extract frames if not already extracted
        if len(os.listdir(self.frame_dir)) == 0:
            print(f"Extracting frames from {video_path} to {self.frame_dir}...")
            self._extract_frames()
        else:
            print(f"Frames already extracted in {self.frame_dir}, skipping extraction.")

        # Load sorted frame paths
        self.frame_paths = sorted(glob(os.path.join(self.frame_dir, "*.jpg")))
        self.total_frames = len(self.frame_paths)

        # Get fps from video
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

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

    def _preprocess(self, frame):
        if self.rgb:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.transform:
            frame = self.transform(frame)
        return frame

    def __iter__(self):
        return self._sliding_window()

    
    def _sliding_window(self):
        all_frames = []

        for path in self.frame_paths:
            frame = read_image(path).float() / 255.0  # Read as tensor and normalize to [0, 1]
            if self.transform:
                frame = self.transform(frame)  # Optionally apply more transforms
            all_frames.append(frame)

        # Padding
        if self.pad_mode == 'zero':
            pad_frame = torch.zeros_like(all_frames[0])
        elif self.pad_mode == 'edge':
            pad_frame = all_frames[0]
        else:
            raise ValueError("pad_mode must be 'zero' or 'edge'")

        padded = [pad_frame] * (self.window_size // 2) + all_frames

        for start in range(0, len(all_frames), self.stride):
            end = start + self.window_size
            if end > len(padded):
                break
            clip = padded[start:end]
            yield torch.stack(clip), start - self.window_size // 2, start + self.window_size // 2 - 1

    def get_fps(self):
        return self.fps

    def get_total_frames(self):
        return self.total_frames


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.CenterCrop(size=(224, 224)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    loader = VideoLoader("../data/Game_4.mp4", window_size=100, stride=50, transform=transform, resize_to=(398, 224))

    output_dir = "saved_frames"
    os.makedirs(output_dir, exist_ok=True)


    for clip, start_idx, end_idx in loader:
        print(f"Window {start_idx}-{end_idx}, shape: {clip.shape}")
        
        # Save the middle frame
        mid_frame = clip[len(clip) // 2]  # shape: (H, W, C)
        print(f"Middle frame shape: {mid_frame.shape}")
        # frame_id = start_idx + len(clip) // 2
        # save_path = os.path.join(output_dir, f"frame_{frame_id:05d}.png")

        # plt.imsave(save_path, mid_frame)
        # print(f"Saved frame to {save_path}")

        break  # Remove break to save all sliding window middle frames

