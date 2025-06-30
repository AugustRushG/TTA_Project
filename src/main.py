from input_process.video_loader import VideoLoader
from event_detection import EventDetectionModel, load_event_detection_model
from ball_tracking import build_ball_tracking_model, load_ball_tracking_model
import torch
import torchvision.transforms as transforms
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Process video frames for event detection and ball tracking.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--window_size', type=int, default=100, help='Size of the sliding window for frame processing.')
    parser.add_argument('--stride', type=int, default=50, help='Stride for the sliding window.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to run the models on.')
    return parser.parse_args()

def main(args):
    # Initialize video loader
    transform = transforms.Compose([
        transforms.CenterCrop(size=(224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    video_loader = VideoLoader(args.video_path, window_size=args.window_size, stride=args.stride, transform=transform)
    print(f'Video loader initialized successfully with {video_loader.get_total_frames()} frames.')

    # Load event detection model configuration
    event_model_config = 'event_detection/model_configs/e2e_res18_hgsm.json'
    with open(event_model_config, 'r') as f:
        event_model_config = json.load(f)

    # Initialize event detection model
    event_model = EventDetectionModel(event_model_config, device=args.device)
    load_event_detection_model(event_model, 'event_detection/checkpoints/E2E_RES18_TTA.pt')

    print(f'Event Detection Model initialized successfully')

    # Initialize ball tracking model
    ball_tracking_model = build_ball_tracking_model(args=type('', (), {'img_size': (224,224), 'num_frames': 5, 'device': args.device})())
    ball_tracking_model = load_ball_tracking_model(ball_tracking_model, 'ball_tracking/checkpoints/TOTNet_TTA_(5)_(224,398)_best.pth', args.device)

    print(f'Ball Tracking Model initialized successfully')


    # Process video frames
    for frames, start_idx, end_idx in video_loader:
        print(frames.shape, start_idx, end_idx)
        event_predictions = event_model.predict(frames)
        # Further processing can be done here
        print(f"Processed frames from {start_idx} to {end_idx}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
