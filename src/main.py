from input_process import FrameClipDataset, extract_frames
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
    frame_dir = extract_frames(args.video_path, resize_to=(398, 224))
    dataset = FrameClipDataset(frame_dir, window_size=args.window_size, stride=args.stride, transform=transform)
    video_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f'Video Loader initialized with {len(dataset)} clips')

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


    for data in video_loader:
        clips = data['frames']
        start_idx = data['start_idx']
        end_idx = data['end_idx']
        clips = clips.to(args.device, dtype=torch.float32)
        # Process clips with event detection model
        pred_results, pred_scores = event_model.predict(clips)
        print(f"Clip shape: {clips.shape}, Start index: {start_idx}, End index: {end_idx}")
        pred_results = pred_results[0]
        for i, pred in enumerate(pred_results):
            if pred != 0:
                print(f"Event detected at clip {i + start_idx} with score {pred_scores[i]:.4f}")
if __name__ == "__main__":
    args = parse_args()
    main(args)
