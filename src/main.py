import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

from input_process import FrameClipDataset, extract_frames
from event_detection import create_model, nms_on_dict
from ball_tracking import BallTrackingModel
from table_detector import TableDetector
from scoreboard_detector import ScoreboardChangeDetector
from utils.visualization import draw_bounces_on_split_table, draw_bounces_on_table
from analysis import AnalysisUtils
import torch
import torchvision.transforms as transforms
import json
import argparse
import random
from tqdm import tqdm
from PIL import Image
import numpy as np


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

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def parse_args():
    parser = argparse.ArgumentParser(description="Process video frames for event detection and ball tracking.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--window_size', type=int, default=100, help='Size of the sliding window for frame processing.')
    parser.add_argument('--stride', type=int, default=50, help='Stride for the sliding window.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'], help='Device to run the models on.')
    return parser.parse_args()





def main(args):
    # Initialize video loader
    event_transform = transforms.Compose([
        transforms.Resize((224, 398)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    ball_transform = transforms.Compose([
        transforms.Resize((288, 512)),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    frame_dir, fps_rate = extract_frames(args.video_path) # dont resize 

    # scoreboard_detector = ScoreboardChangeDetector(frames_folder=frame_dir, video_fps=fps_rate)
    # changes = scoreboard_detector.detect_changes()
    # print(f"In total {len(changes)} scoreboard changes detected at frames:")
    # with open("score_timeline.json", "w") as f:
    #     json.dump(changes, f, indent=4)

    game_name = os.path.basename(args.video_path).split('.')[0]
    print(f'Extracted frames for {game_name} into {frame_dir}')

    dataset = FrameClipDataset(frame_dir, window_size=args.window_size, stride=args.stride, event_transform=event_transform, ball_transform=ball_transform)
    video_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f'Video Loader initialized with {len(dataset)} clips')

    # Load event detection model configuration
    event_model = create_model(model_type='astrm', model_config_path='event_detection/model_configs/ASTRM.json',
                               device=args.device, model_checkpoint_path='event_detection/checkpoints/ASTRM_TTAV3.pth')

    print(f'Event Detection Model initialized successfully')

    # Initialize ball tracking model
    ball_tracking_model = BallTrackingModel(
        model_choice='TOTNet',
        model_args=type('', (), {'img_size': (288, 512), 'num_frames': 5, 'device': args.device})(),
        checkpoint_path='ball_tracking/checkpoints/TOTNet_TTA_(5)_(288,512)_30epochs_best.pth'
    )

    # ball_tracking_model = BallTrackingModel(
    #     model_choice='TOTNet',
    #     model_args=type('', (), {'img_size': (288, 512), 'num_frames': 5, 'device': args.device})(),
    #     checkpoint_path='ball_tracking/checkpoints/TOTNet_TTA_(5)_(288,512)_bidirect_30epochs_best.pth'
    # )
    TOTNet_OF = BallTrackingModel(
        model_choice='TOTNet_OF',
        model_args=type('', (), {'img_size': (288, 512), 'num_frames': 5, 'device': args.device})(),
        checkpoint_path='ball_tracking/checkpoints/TOTNet_OF_TTA_(5)_(288,512)_30epochs_best.pth'
    )

    print(f'Ball Tracking Model initialized successfully')
    
    # Generate predictions
    pred_events = {}
    threshold = 0.05  # threshold for event detection scores, can be adjusted

    # generating event predictions
    for data in tqdm(video_loader, desc="Processing clips"):
        clips = data['frames']
        start_idx = data['start_idx']
        end_idx = data['end_idx']
        clips = clips.to(args.device, dtype=torch.float32)
        # Process clips with event detection model
        pred_results, pred_scores = event_model.predict(clips, device=args.device)  # pred_results: [B, T], pred_scores: [B, T, C]

        # print(f"Clip shape: {clips.shape}, Start index: {start_idx}, End index: {end_idx}")
        pred_results = pred_results[0] # batch = 1, so we can take the first element
        pred_scores = pred_scores[0]  # batch = 1, so we can take the first element

        for i, (pred_event, pred_score_classes) in enumerate(zip(pred_results, pred_scores)):
            current_id = (i + start_idx).item()
            current_time = current_id / fps_rate
            current_time_in_mins = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"

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
                'time': current_time,
                'time_in_mins': current_time_in_mins,
                'event_type': CLASS_CONVERSION.get(best_class.item(), 'unknown'),
                'score': float(best_score)
            }

    event_windows = {'close_table_serve': 20, 'far_table_serve': 20}
    pred_events = nms_on_dict(pred_events, event_windows=event_windows)  # Apply NMS to the predictions

    # generate pred_events as json file 
    with open(f'predicted_events_{game_name}.json', 'w') as f:
        json.dump(pred_events, f, indent=4)

    frame_indices = [
        dataset.num_frames // 2 - random.randint(0, 100),
        # dataset.num_frames // 2 - random.randint(0, 100),
        # dataset.num_frames // 2 + random.randint(0, 100),
        dataset.num_frames // 2 + random.randint(0, 100),
    ]

    converted_img_paths = []

    for i, idx in enumerate(frame_indices):
        filename = f"{idx:06d}.jpg"
        img_path = os.path.join(frame_dir, filename)

        img = Image.open(img_path).convert("RGB")
        # img_trans = transforms.CenterCrop(size=(224,224))
        img_trans = transforms.Resize((288, 512))
        converted_img = img_trans(img)

        os.makedirs('./result', exist_ok=True)
        converted_img_path = os.path.join('./result', f'converted_{i}.jpg')
        converted_img.save(converted_img_path)
        converted_img_paths.append(converted_img_path)

    table_detector = TableDetector(image_path=converted_img_paths[0], topdown_width=1525, topdown_height=2740)
    # corners_top = np.array([(204, 96), (306, 96), (197, 129), (309, 126)], dtype=np.float32)
    # corners_bottom = np.array([(197, 129), (309, 126), (191, 173), (313, 172)], dtype=np.float32)
    # table_detector.detect_corners_and_compute()
    table_detector.set_corners_manual(
        corners_top=np.array([(204, 96), (306, 96), (197, 129), (309, 126)], dtype=np.float32),
        corners_bottom=np.array([(197, 129), (309, 126), (191, 173), (313, 172)], dtype=np.float32)
    )

    # plt.show(converted_img)

                
    #generate ball locations for each event
    for frame_idx, event_info in tqdm(pred_events.items(), desc="Ball tracking"):

        event_type = event_info['event_type']
        score = event_info['score']

        if event_type in ['far_table_bounce', 'close_table_bounce']:
            ball_location_frames = dataset.get_surrounding_frames(frame_idx, radius=2)

            ball_location_frames = ball_location_frames.to(args.device, dtype=torch.float32)
            ball_location_frames = ball_location_frames.unsqueeze(0)
            # print(f"Processing ball location for frame {frame_idx} with event type {event_type} and score {score}")
            # print(f"Ball location frames shape: {ball_location_frames.shape}")
            ball_location_result, confidence = ball_tracking_model.predict(ball_location_frames)
            extracted_coord = ball_tracking_model.extract_coordinates_2d(ball_location_result, H=288, W=512)[0]
            x_pred, y_pred = extracted_coord[0], extracted_coord[1]

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

            pred_events[frame_idx]['ball_coord_confidence'] = float(confidence)

            # pred_events[frame_idx]['draw_ball_location'] = {
            #     'x': float(mapped_x),
            #     'y': float(mapped_y)
            # }

            if confidence < 0.5:
                print(f"Low confidence ({confidence}) for frame {frame_idx}, trying Optical Flow model")
                ball_location_result_OF, confidence_OF = TOTNet_OF.predict(ball_location_frames)
                extracted_coord_OF = TOTNet_OF.extract_coordinates_2d(ball_location_result_OF, H=288, W=512)[0]
                x_pred_OF, y_pred_OF = extracted_coord_OF[0], extracted_coord_OF[1]
                # to numpy
                x_pred_OF = x_pred_OF.cpu().numpy()
                y_pred_OF = y_pred_OF.cpu().numpy()
                print(f"Using Optical Flow model for frame {frame_idx} with confidence {confidence_OF}")
                pred_events[frame_idx]['ball_coord_confidence_of'] = float(confidence_OF)
                pred_events[frame_idx]['ball_location_of'] = {
                    'x': float(x_pred_OF),
                    'y': float(y_pred_OF)
                }
                mapped_x_OF, mapped_y_OF = table_detector.transform_ball(x_pred_OF, y_pred_OF)
                pred_events[frame_idx]['mapped_ball_location_of'] = {
                    'x': float(mapped_x_OF),
                    'y': float(mapped_y_OF)
                }
                if mapped_x <0 or mapped_y < 0:
                    if mapped_x_OF > 0 and mapped_y_OF > 0:
                        pred_events[frame_idx]['draw_ball_location'] = {
                            'x': float(mapped_x_OF),
                            'y': float(mapped_y_OF)
                        }
                    else:
                        pred_events[frame_idx]['draw_ball_location'] = {
                            'x': float(mapped_x),
                            'y': float(mapped_y)
                        }
                else:
                    pred_events[frame_idx]['draw_ball_location'] = {
                        'x': float(mapped_x),
                        'y': float(mapped_y)
                    }

    # generate pred_events as json file 
    with open(f'predicted_events_{game_name}.json', 'w') as f:
        json.dump(pred_events, f, indent=4)


    return pred_events

    
def calculate_points_score_pred(scoreboard_changes, pred_events):
    """ Calculate points based on predicted events and scoreboard changes """
    close_points = 0
    far_points = 0
    if not pred_events:
        return

    # sort keys numerically (handles string keys like "188")
    keys = sorted(pred_events.keys(), key=lambda x: int(x) if isinstance(x, str) else x)

    for rec in scoreboard_changes.values():
        frame_idx = rec['frame']
        closest_event = None

        for i, k in enumerate(keys):
            cur = pred_events[k]
            cur_frame = cur['frame_index']  # often == int(k)

            next_k = keys[i+1] if i+1 < len(keys) else None
            next_frame = pred_events[next_k]['frame_index'] if next_k is not None else None

            if cur_frame <= frame_idx and (next_k is None or next_frame > frame_idx):
                closest_event = k
                event_type = cur['event_type']
                print(f"Frame {frame_idx} scoreboard change matched to event at frame {cur_frame} (key={k}), type: {event_type}, time {cur['time_in_mins']}")
                if event_type.startswith('close_table'):
                    far_points += 1
                    print("far table point")
                elif event_type.startswith('far_table'):
                    close_points += 1
                    print("close table point")
                break
    
    print(f"Predicted Score - Close Table: {close_points}, Far Table: {far_points}")


        

   



            

            
            
if __name__ == "__main__":
    args = parse_args()
    pred_events = main(args)
    # read json file
    with open(f'predicted_events_{os.path.basename(args.video_path).split(".")[0]}.json', 'r') as f:
        pred_events = json.load(f)
    with open('/home/august/github/TTA_Project/src/score_timeline.json', 'r') as f:
        scoreboard_changes = json.load(f)
    calculate_points_score_pred(scoreboard_changes, pred_events)
    draw_bounces_on_table(pred_events, save_path='bounces_on_table.jpg')
    draw_bounces_on_split_table(pred_events, save_path='bounces_on_table_split.jpg')
    # # visualize the result
    # AnalysisUtils.count_bounces(pred_events)
    # AnalysisUtils.count_serves(pred_events)
    # AnalysisUtils.calculate_points(pred_events)
    # # save again
    # AnalysisUtils.save_json(pred_events, f'predicted_events_{os.path.basename(args.video_path).split(".")[0]}.json')
