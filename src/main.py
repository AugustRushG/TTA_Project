import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import argparse
import copy
import json
import random
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")  # or Qt5Agg
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# Import notebook helpers - these encapsulate the same logic as the notebook cells
import notebook_helpers as nh
from notebook_helpers import (
    reorganize_rallies_with_bounces_under_hits,
    update_split_draw_coordinates,
    load_pipeline,
    prepare_video_inputs,
    detect_scoreboard_changes,
    prepare_rallies,
    add_ball_tracking_to_rallies,
    build_final_output_paths,
    export_final_artifacts,
    summarize_rallies_counts,
    read_json,
    save_json,
    select_roi_jupyter,
    get_roi_result,
    click_6_points_jupyter,
    show_cropped_image,
    save_table_points_preview,
)

# Also import base components for direct usage when needed
from input_process import FrameClipDataset, extract_frames
from event_detection import create_model, nms_on_dict
from ball_tracking import BallTrackingModel
from table_detector import TableDetector
from scoreboard_detector import ResNetScoreboardChangeDetector
from utils.visualization import draw_bounces_on_split_table
from convert_grouped_rallies_to_output import convert_grouped_rallies
from json_to_xml import write_xml_data



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
    parser.add_argument('--summary-output', type=str, help='Path to the final rally summary JSON file.')
    parser.add_argument('--json-output', type=str, help='Path to the XML-ready JSON file.')
    parser.add_argument('--xml-output', type=str, help='Path to the final XML file.')
    parser.add_argument('--game-label', type=str, default='Game 1', help='Game label text for XML-ready output.')
    parser.add_argument('--close-player', type=str, default='Close Player', help='Close side player display name.')
    parser.add_argument('--far-player', type=str, default='Far Player', help='Far side player display name.')
    parser.add_argument('--include-bounces', action='store_true', help='Include nested bounce entries in XML-ready output.')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Base output directory for results.')
    
    # Optional: Model paths (if not using default locations)
    parser.add_argument('--event-model-folder', type=str, default=None, help='Path to event detection model folder.')
    parser.add_argument('--ball-tracking-model-dir', type=str, default=None, help='Path to ball tracking model directory.')
    
    return parser.parse_args()



def main(args):
    # Runtime configuration matching notebook defaults
    window_size = args.window_size
    stride = args.stride
    device = args.device
    
    # Model paths - use args or defaults
    event_detection_model_folder = args.event_model_folder or 'event_detection/checkpoints'
    ball_tracking_model_dir_path = args.ball_tracking_model_dir or 'ball_tracking/checkpoints'
    
    # Event detection threshold and windows
    EVENT_DETECTION_THRESHOLD = 0.05
    EVENT_WINDOWS = {
        "close_table_serve": 20,
        "far_table_serve": 20,
    }

    # Print GPU information
    if device == 'cuda' and torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif device == 'mps' and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS device")
    else:
        print("Using CPU")

    # Load models and transforms using notebook helper
    (
        event_detection_model,
        ball_tracking_model,
        event_transform,
        ball_transform,
        _,
        _,
    ) = load_pipeline(
        device=device,
        event_detection_model_folder=event_detection_model_folder,
        ball_tracking_model_dir_path=ball_tracking_model_dir_path,
    )

    # Prepare video inputs (extract frames)
    video_inputs = prepare_video_inputs(
        video_path=args.video_path,
        device=device,
        extract_frames_fn=extract_frames,
        read_image_fn=Image.open,
    )

    frame_dir = video_inputs["frame_dir"]
    fps_rate = video_inputs["fps_rate"]
    game_name = video_inputs["game_name"]
    sample_frame = video_inputs["sample_frame"]

    print(f'Extracted frames for {game_name} into {frame_dir}')

    # Create output directory early (before interactive selections)
    RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = Path(args.output_dir or f'outputs/{game_name}/{RUN_ID}')
    VISUALS_DIR = OUTPUT_DIR / "visuals"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {OUTPUT_DIR}")

    # ===== Interactive ROI Selection for Scoreboard =====
    print("\n" + "="*60)
    print("STEP 1: Select Close Scoreboard ROI")
    print("="*60)
    close_roi_coords = select_roi_jupyter(
        sample_frame,
        title="Select Close Scoreboard ROI",
    )
    existing_close_coord = get_roi_result(close_roi_coords)
    print(f"Close ROI: {existing_close_coord}")
    show_cropped_image(
        np.array(sample_frame),
        existing_close_coord,
        save_path=str(VISUALS_DIR / "scoreboard_close_roi.png"),
    )

    print("\n" + "="*60)
    print("STEP 2: Select Far Scoreboard ROI")
    print("="*60)
    far_roi_coords = select_roi_jupyter(
        sample_frame,
        title="Select Far Scoreboard ROI",
    )
    existing_far_coord = get_roi_result(far_roi_coords)
    print(f"Far ROI: {existing_far_coord}")
    show_cropped_image(
        np.array(sample_frame),
        existing_far_coord,
        save_path=str(VISUALS_DIR / "scoreboard_far_roi.png"),
    )

    # Detect scoreboard changes using selected ROIs
    scoreboard_result = detect_scoreboard_changes(
        frame_dir=frame_dir,
        fps_rate=fps_rate,
        device=device,
        existing_close_coord=existing_close_coord,
        existing_far_coord=existing_far_coord,
    )
    changes = scoreboard_result["changes"]
    final_score_close = scoreboard_result["final_score_close"]
    final_score_far = scoreboard_result["final_score_far"]

    print(f"In total {len(changes)} scoreboard changes detected at frames:")
    print(f"Final score of the game - Close Table: {final_score_close}, Far Table: {final_score_far}")

    # Prepare rallies using notebook helper
    rally_result = prepare_rallies(
        frame_dir=frame_dir,
        window_size=window_size,
        stride=stride,
        event_transform=event_transform,
        ball_transform=ball_transform,
        event_detection_model=event_detection_model,
        device=device,
        fps_rate=fps_rate,
        class_conversion=CLASS_CONVERSION,
        scoreboard_changes=changes,
        threshold=EVENT_DETECTION_THRESHOLD,
        event_windows=EVENT_WINDOWS,
    )

    dataset = rally_result["dataset"]
    video_loader = rally_result["video_loader"]
    pred_events = rally_result["pred_events"]
    grouped_rallies = rally_result["grouped_rallies"]

    print(f'Video Loader initialized with {len(dataset)} clips')

    # ===== Interactive Table Points Selection =====
    print("\n" + "="*60)
    print("STEP 3: Select 6 Table Reference Points")
    print("="*60)
    sample_image_path = video_inputs["sample_image_path"]
    img = Image.open(sample_image_path).convert("RGB")
    img_np = np.array(img)
    
    # Resize for display
    img_trans = transforms.Resize((512, 512))
    converted_img = img_trans(img)
    converted_img_np = np.array(converted_img)
    
    table_points_state = click_6_points_jupyter(
        converted_img_np,
        title="Click 6 points: TL, TR, mid-left, mid-right, BL, BR",
    )
    table_points = table_points_state["points"]
    print(f"Selected table points: {table_points}")
    
    save_table_points_preview(
        converted_img_np,
        table_points,
        str(VISUALS_DIR / "table_points_selection.png"),
    )
    
    # Initialize table detector with selected points
    table_detector = TableDetector(image_path=sample_image_path, topdown_width=1525, topdown_height=2740)
    table_detector.compute_homographies(corners6=table_points)

    print(f'Table Detector initialized successfully')

    # Add ball tracking to rallies using notebook helper
    grouped_rallies = add_ball_tracking_to_rallies(
        grouped_rallies=grouped_rallies,
        dataset=dataset,
        ball_tracking_model=ball_tracking_model,
        table_detector=table_detector,
        device=device,
    )
    print(f"Ball tracking complete for {len(grouped_rallies)} rallies")

    # Table dimensions
    TABLE_W, TABLE_H = 1525.0, 2740.0
    SPLIT_W, SPLIT_L = 153.0, 137.0

    # Summarize rallies
    summaries, invalid_bounce_total, bounce_total = summarize_rallies_counts(
        grouped_rallies, SPLIT_W, SPLIT_L, TABLE_W, TABLE_H
    )
    for s in summaries:
        print(f"Rally {s['rally_id']}: bounces(close/far)={s['close_bounce']}/{s['far_bounce']}, "
              f"forehand={s['total_forehand']}, backhand={s['total_backhand']}, invalid_bounces={s['invalid_bounces']}")
    print(f"Invalid bounces: {invalid_bounce_total}/{bounce_total}")

    # Build output paths
    OUTPUT_PATHS = build_final_output_paths(game_name, output_dir=str(OUTPUT_DIR))
    
    # Use provided paths or defaults
    summary_output_path = args.summary_output or OUTPUT_PATHS.get("summary_path", f'final_rally_summary_{game_name}.json')
    json_output_path = args.json_output or OUTPUT_PATHS.get("xml_ready_json_path", f'xml_ready_{game_name}.json')
    xml_output_path = args.xml_output or OUTPUT_PATHS.get("xml_path", f'xml_ready_{game_name}.xml')

    # Prepare output with reorganized rallies
    summary_grouped = copy.deepcopy(grouped_rallies)
    summary_grouped = reorganize_rallies_with_bounces_under_hits(summary_grouped)
    summary_grouped, _ = update_split_draw_coordinates(
        grouped_rallies=summary_grouped,
        table_w=TABLE_W,
        table_h=TABLE_H,
        split_w=SPLIT_W,
        split_l=SPLIT_L,
    )

    # Save summary
    save_json(summary_output_path, summary_grouped)

    # Convert to XML-ready format
    output_payload = convert_grouped_rallies(
        grouped_rallies=grouped_rallies,
        fps=fps_rate,
        game=args.game_label,
        close_player=args.close_player,
        far_player=args.far_player,
        include_bounces=args.include_bounces,
    )

    save_json(json_output_path, output_payload)
    write_xml_data(output_payload, xml_output_path)

    print(f"Saved final rally summary: {summary_output_path}")
    print(f"Saved XML-ready JSON: {json_output_path}")
    print(f"Saved XML: {xml_output_path}")
    print(f"Run output directory: {OUTPUT_DIR}")

    return grouped_rallies

    
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



# filter_based_on_score_changes is now imported from notebook_helpers
# The logic is encapsulated in prepare_rallies() function
    
            
   



            

            
            
if __name__ == "__main__":
    args = parse_args()
    main(args)
    # # visualize the result
    # AnalysisUtils.count_bounces(pred_events)
    # AnalysisUtils.count_serves(pred_events)
    # AnalysisUtils.calculate_points(pred_events)
    # # save again
    # AnalysisUtils.save_json(pred_events, f'predicted_events_{os.path.basename(args.video_path).split(".")[0]}.json')
