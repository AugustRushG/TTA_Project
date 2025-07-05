import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from input_process import FrameClipDataset, extract_frames
from event_detection import EventDetectionModel, load_event_detection_model
from ball_tracking import build_ball_tracking_model, load_ball_tracking_model
from table_detector import TableDetector
import torch
import torchvision.transforms as transforms
import json
import argparse
import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

CLASS_CONVERSION = {
    0: 'empty',
    1: 'far_table_bounce',
    2: 'far_table_forehand',
    3: 'far_table_backhand',
    4: 'far_table_serve',
    5: 'close_table_bounce',
    6: 'close_table_forehand',
    7: 'close_table_backhand',
    8: 'close_table_serve',
}

def parse_args():
    parser = argparse.ArgumentParser(description="Process video frames for event detection and ball tracking.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--window_size', type=int, default=100, help='Size of the sliding window for frame processing.')
    parser.add_argument('--stride', type=int, default=50, help='Stride for the sliding window.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to run the models on.')
    return parser.parse_args()


def draw_bounces_on_table(bounces, table_size=(1525, 2740), save_path=None):
    """
    Draw bounces on a table view.

    Args:
        bounces (dict): {frame_id: {"event_type": str, "mapped_ball_location": {"x":, "y":}}}
        table_size (tuple): (width, height) of the table in pixels
        save_path (str): optional path to save figure
    """
    W, H = table_size

    fig, ax = plt.subplots(figsize=(6, 10))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # origin at top-left
    ax.set_aspect('equal')
    ax.set_title("Bounce Events on Table")

    # Draw table outline
    ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, linewidth=2, edgecolor='black'))

    # Draw bounces
    for frame_id, info in bounces.items():
        if "mapped_ball_location" not in info:
            continue
        x = info["mapped_ball_location"]["x"]
        y = info["mapped_ball_location"]["y"]
        event_type = info.get("event_type", "")

        ax.plot(x, y, 'ro')
        ax.text(x+5, y, f"{frame_id}", fontsize=6, color='blue')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


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
    load_event_detection_model(event_model, 'event_detection/checkpoints/E2E_RES18_HGSM_TTA.pt')


    print(f'Event Detection Model initialized successfully')

    # Initialize ball tracking model
    ball_tracking_model = build_ball_tracking_model(args=type('', (), {'img_size': (224,224), 'num_frames': 5, 'device': args.device})())
    ball_tracking_model = load_ball_tracking_model(ball_tracking_model, 'ball_tracking/checkpoints/TOTNet_TTA_(5)_(224,398)_best.pth', args.device)
    ball_tracking_model.to(args.device)
    ball_tracking_model.eval()

    print(f'Ball Tracking Model initialized successfully')
    
    # Generate predictions
    pred_events = {}
    threshold = 0.1  # or whatever you choose

    # generating event predictions
    for data in video_loader:
        clips = data['frames']
        start_idx = data['start_idx']
        end_idx = data['end_idx']
        clips = clips.to(args.device, dtype=torch.float32)
        # Process clips with event detection model
        pred_results, pred_scores = event_model.predict(clips)
   
        print(f"Clip shape: {clips.shape}, Start index: {start_idx}, End index: {end_idx}")
        pred_results = pred_results[0] # batch = 1, so we can take the first element
        pred_scores = pred_scores[0]  # batch = 1, so we can take the first element

        for i, (pred_event, pred_score_classes) in enumerate(zip(pred_results, pred_scores)):
            current_id = (i + start_idx).item()
           
            # filter: set scores < threshold to 0
            filtered_scores = pred_score_classes * (pred_score_classes >= threshold)

            filtered_scores[0] = -1
            # find the highest class & score after filtering
            best_class = filtered_scores.argmax()
            best_score = filtered_scores[best_class]

            # skip if the highest remaining class is 0 (empty) or score is 0
            if best_score == 0:
                continue

            # add to dict
            pred_events[current_id] = {
                'event_type': CLASS_CONVERSION.get(best_class.item(), 'unknown'),
                'score': float(best_score)
            }
    
    pred_events = event_model.nms_on_dict(pred_events, nms_window=3)  # Apply NMS to the predictions
    
    # # generate pred_events as json file 
    # with open('predicted_events.json', 'w') as f:
    #     json.dump(pred_events, f, indent=4)


    
    frame_indices = [
        dataset.num_frames // 2 - random.randint(0, 100),
        dataset.num_frames // 2 - random.randint(0, 100),
        dataset.num_frames // 2 + random.randint(0, 100),
        dataset.num_frames // 2 + random.randint(0, 100),
    ]

    cropped_img_paths = []

    for i, idx in enumerate(frame_indices):
        filename = f"{idx:06d}.jpg"
        img_path = os.path.join(frame_dir, filename)

        img = Image.open(img_path).convert("RGB")
        center_crop = transforms.CenterCrop(size=(224, 224))
        cropped_img = center_crop(img)

        os.makedirs('./result', exist_ok=True)
        cropped_img_path = os.path.join('./result', f'cropped_{i}.jpg')
        cropped_img.save(cropped_img_path)
        cropped_img_paths.append(cropped_img_path)
    
    table_detector = TableDetector(image_paths=cropped_img_paths, topdown_width=1525, topdown_height=2740)

    table_corners = table_detector.detect_average_corners()
    table_detector.warp_table(save_path=os.path.join('./result', 'warped_table.jpg'))

    print(f"Detected table corners: {table_corners}")

                
    #generate ball locations for each event
    for frame_idx, event_info in pred_events.items():
        event_type = event_info['event_type']
        score = event_info['score']

        if event_type in ['far_table_bounce', 'close_table_bounce']:
            ball_location_frames = dataset.get_surrounding_frames(frame_idx, radius=2)

            ball_location_frames = ball_location_frames.to(args.device, dtype=torch.float32)
            ball_location_frames = ball_location_frames.unsqueeze(0)
            ball_location_result, _ = ball_tracking_model(ball_location_frames)
            extracted_coord = ball_tracking_model.extract_coords(ball_location_result)[0]
            x_pred, y_pred = extracted_coord[1], extracted_coord[0]
            # to numpy 
            x_pred = x_pred.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            pred_events[frame_idx]['ball_location'] = {
                'x': float(x_pred),
                'y': float(y_pred)
            }

            # map the coordinates to the top-down view
            mapped_x, mapped_y = table_detector.transform_ball(x_pred, y_pred)
            pred_events[frame_idx]['mapped_ball_location'] = {
                'x': float(mapped_x),
                'y': float(mapped_y)
            }

    # generate pred_events as json file 
    with open('predicted_events.json', 'w') as f:
        json.dump(pred_events, f, indent=4)




    




            

            
            
if __name__ == "__main__":
    args = parse_args()
    # main(args)
    # read json file
    with open('predicted_events.json', 'r') as f:
        pred_events = json.load(f)
    draw_bounces_on_table(pred_events, save_path='bounces_on_table.jpg')


