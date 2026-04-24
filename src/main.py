import os
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import argparse
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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

from input_process import FrameClipDataset, extract_frames
from event_detection import nms_on_dict
from ball_tracking import BallTrackingModel
import ball_tracking.transform as bt_transform
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


def parse_args():
    parser = argparse.ArgumentParser(description="Process video frames for event detection and ball tracking.")
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda', 'mps'])
    parser.add_argument('--game-label', type=str, default='Game 1')
    parser.add_argument('--close-player', type=str, default='Close Player')
    parser.add_argument('--far-player', type=str, default='Far Player')
    parser.add_argument('--include-bounces', action='store_true')
    parser.add_argument('--output-dir', type=str, default='outputs')
    return parser.parse_args()


def main(args):
    window_size = args.window_size
    stride = args.stride
    device = args.device

    # ── Model paths via HuggingFace (mirrors notebook Cell 2) ──────────────
    from huggingface_hub import snapshot_download
    ball_tracking_dir   = snapshot_download("AugustRushG123/TOTNet")
    event_detection_dir = snapshot_download("AugustRushG123/MFS_Model")

    event_detection_model_folder = os.path.join(
        event_detection_dir,
        "E2E800MFS_SL_TTAV3(1.0)_FP16_8GPU_50epochs_lr1e-3_bs10_seed42",
    )
    ball_tracking_model_dir_path = os.path.join(
        ball_tracking_dir,
        "TOTNet_TTA_(5)_Bidirect_(512,512)_BallMask_50epochs_WBCE[1,2,3,3]_bs_ch32",
    )

    EVENT_DETECTION_THRESHOLD = 0.01
    EVENT_WINDOWS = {
        "close_table_serve":  150,
        "far_table_serve":    150,
        "far_table_bounce":   3,
        "close_table_bounce": 3,
    }

    print(f"Device: {device}")
    print(f"Video path: {args.video_path}")
    print(f"Event model folder: {event_detection_model_folder}")
    print(f"Ball tracking model folder: {ball_tracking_model_dir_path}")
    print(f"Event detection threshold: {EVENT_DETECTION_THRESHOLD}")
    print(f"Serve NMS windows: {EVENT_WINDOWS}")

    if device == 'cuda' and torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif device == 'mps' and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS device")
    else:
        print("Using CPU")

    # ── Load models (mirrors notebook Cell 4) ──────────────────────────────
    (
        event_detection_model,
        ball_tracking_model,
        event_transform,
        ball_transform,
        MEAN,
        STD,
    ) = load_pipeline(
        device=device,
        event_detection_model_folder=event_detection_model_folder,
        ball_tracking_model_dir_path=ball_tracking_model_dir_path,
    )

    # ── Video preparation (mirrors notebook Cells 6-7) ─────────────────────
    video_inputs = prepare_video_inputs(
        video_path=args.video_path,
        device=device,
        extract_frames_fn=extract_frames,
        read_image_fn=nh.read_image,
    )

    frame_dir        = video_inputs["frame_dir"]
    fps_rate         = video_inputs["fps_rate"]
    game_name        = video_inputs["game_name"]
    image_files      = video_inputs["image_files"]
    sample_image_path = video_inputs["sample_image_path"]
    sample_frame     = video_inputs["sample_frame"]

    print(f'Extracted frames for {game_name} into {frame_dir}')

    RUN_ID      = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR  = Path(args.output_dir) / game_name / RUN_ID
    VISUALS_DIR = OUTPUT_DIR / "visuals"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Run output directory: {OUTPUT_DIR}")
    print(f"Saved drawings folder: {VISUALS_DIR}")

    # ── Scoreboard ROI selection (mirrors notebook Cells 9-13) ─────────────
    print("\n" + "="*60)
    print("STEP 1: Select Close Scoreboard ROI")
    print("="*60)
    close_roi_coords = select_roi_jupyter(sample_frame, title="Select Close Scoreboard ROI")
    existing_close_coord = get_roi_result(close_roi_coords)
    print(existing_close_coord)
    show_cropped_image(
        sample_frame,
        existing_close_coord,
        save_path=str(VISUALS_DIR / "scoreboard_close_roi.png"),
    )

    print("\n" + "="*60)
    print("STEP 2: Select Far Scoreboard ROI")
    print("="*60)
    far_roi_coords = select_roi_jupyter(sample_frame, title="Select Far Scoreboard ROI")
    existing_far_coord = get_roi_result(far_roi_coords)
    print(existing_far_coord)
    show_cropped_image(
        sample_frame,
        existing_far_coord,
        save_path=str(VISUALS_DIR / "scoreboard_far_roi.png"),
    )

    # ── Scoreboard change detection (mirrors notebook Cell 14) ─────────────
    scoreboard_result = detect_scoreboard_changes(
        frame_dir=frame_dir,
        fps_rate=fps_rate,
        device=device,
        existing_close_coord=existing_close_coord,
        existing_far_coord=existing_far_coord,
    )
    changes          = scoreboard_result["changes"]
    final_score_close = scoreboard_result["final_score_close"]
    final_score_far   = scoreboard_result["final_score_far"]

    print(f"In total {len(changes)} scoreboard changes detected")
    print(f"Final score - Close: {final_score_close}, Far: {final_score_far}")

    # ── Rally preparation (mirrors notebook Cell 15) ────────────────────────
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

    dataset        = rally_result["dataset"]
    video_loader   = rally_result["video_loader"]
    pred_events    = rally_result["pred_events"]
    grouped_rallies = rally_result["grouped_rallies"]

    # Debug bounce count (mirrors notebook Cell 16)
    bounce_count = sum(
        1 for e in pred_events.values()
        if e.get("event_type", "").endswith("bounce")
    )
    print(f"Total predicted bounces: {bounce_count}")

    # ── Table point selection (mirrors notebook Cells 18-19) ───────────────
    print("\n" + "="*60)
    print("STEP 3: Select 6 Table Reference Points")
    print("="*60)
    CenterCropResizeFrame = bt_transform.CenterCropResizeFrame
    img_trans    = CenterCropResizeFrame(size=(512, 512), crop_ratio=(1, 1))
    img          = Image.open(sample_image_path).convert("RGB")
    frame_np     = np.array(img)
    converted_img = img_trans(frame_np)

    state = click_6_points_jupyter(converted_img)
    table_points = state["points"]
    print(f"Selected table points: {table_points}")

    save_table_points_preview(
        converted_img,
        table_points,
        str(VISUALS_DIR / "table_points_selection.png"),
    )

    table_detector = TableDetector(image_path=sample_image_path, topdown_width=1525, topdown_height=2740)
    table_detector.compute_homographies(corners6=table_points)
    table_detector.warp_table(save_path=str(VISUALS_DIR / "table_topdown.png"))

    print("Table Detector initialized successfully")

    # ── Ball tracking (mirrors notebook Cell 20) ────────────────────────────
    grouped_rallies = add_ball_tracking_to_rallies(
        grouped_rallies=grouped_rallies,
        dataset=dataset,
        ball_tracking_model=ball_tracking_model,
        table_detector=table_detector,
        device=device,
    )
    print(f"Ball tracking complete for {len(grouped_rallies)} rallies (kept in memory).")

    # ── Bounce visualisation (mirrors notebook Cell 22) ─────────────────────
    BOUNCE_DRAWING_PATH = VISUALS_DIR / "bounces_split.png"
    draw_bounces_on_split_table(grouped_rallies, save_path=str(BOUNCE_DRAWING_PATH))
    print(f"Saved bounce drawing: {BOUNCE_DRAWING_PATH}")

    # ── Rally summary (mirrors notebook Cell 23) ────────────────────────────
    SPLIT_W, SPLIT_L = 153.0, 137.0
    TABLE_W, TABLE_H = 1525.0, 2740.0

    summaries, invalid_bounce_total, bounce_total = summarize_rallies_counts(
        grouped_rallies, SPLIT_W, SPLIT_L, TABLE_W, TABLE_H
    )
    for s in summaries:
        print(f"Rally {s['rally_id']}: bounces(close/far)={s['close_bounce']}/{s['far_bounce']}, "
              f"forehand={s['total_forehand']}, backhand={s['total_backhand']}, invalid_bounces={s['invalid_bounces']}")
    print(f"Invalid bounces: {invalid_bounce_total}/{bounce_total}")
    print("Rally event counts kept in memory only.")

    # ── Export configuration (mirrors notebook Cell 24) ─────────────────────
    GAME_LABEL             = args.game_label
    CLOSE_PLAYER           = args.close_player
    FAR_PLAYER             = args.far_player
    FPS                    = fps_rate
    INCLUDE_BOUNCES_IN_OUTPUT = args.include_bounces

    print("Export configuration ready.")
    print(f"Game label: {GAME_LABEL}")
    print(f"Close player: {CLOSE_PLAYER}")
    print(f"Far player: {FAR_PLAYER}")

    # ── Export artifacts (mirrors notebook Cell 26) ──────────────────────────
    OUTPUT_PATHS       = build_final_output_paths(game_name, output_dir=str(OUTPUT_DIR))
    SCORE_TIMELINE_PATH = OUTPUT_DIR / f"{game_name}_score_timeline.json"

    legacy_paths = [
        f"grouped_rallies_{game_name}.json",
        f"grouped_rallies_with_nesting_{game_name}.json",
        f"grouped_rallies_with_nesting_{game_name}_enriched.json",
        f"rally_event_counts_{game_name}.json",
        "bounces_on_table.jpg",
        "bounces_on_table_split.png",
    ]

    export_result = export_final_artifacts(
        grouped_rallies=grouped_rallies,
        game_name=game_name,
        game_label=GAME_LABEL,
        close_player=CLOSE_PLAYER,
        far_player=FAR_PLAYER,
        fps=FPS,
        output_dir=str(OUTPUT_DIR),
        summary_path=OUTPUT_PATHS["summary_path"],
        xml_ready_json_path=OUTPUT_PATHS["xml_ready_json_path"],
        xml_path=OUTPUT_PATHS["xml_path"],
        cleanup_paths=legacy_paths,
        nest_bounces_under_hits=False,
        table_w=TABLE_W,
        table_h=TABLE_H,
        split_w=SPLIT_W,
        split_l=SPLIT_L,
        include_bounces_in_output=INCLUDE_BOUNCES_IN_OUTPUT,
    )

    save_json(str(SCORE_TIMELINE_PATH), changes)

    print(f"Run output directory: {OUTPUT_DIR}")
    print(f"Saved drawings folder: {VISUALS_DIR}")
    print(f"Saved final rally summary: {export_result['summary_path']}")
    print(f"Saved XML-ready JSON: {export_result['xml_ready_json_path']}")
    print(f"Saved XML: {export_result['xml_path']}")
    print(f"Saved score timeline: {SCORE_TIMELINE_PATH}")
    print(f"Updated bounce split coords: {export_result['updated_bounces']}")
    print(f"Total output instances: {export_result['instance_count']}")
    if export_result['removed_files']:
        print("Removed temporary files:")
        for path in export_result['removed_files']:
            print(f"  - {path}")

    # Final summary (mirrors notebook Cell 27)
    print("\nFinal artifacts created in this run folder:")
    print(f"  - {OUTPUT_PATHS['summary_path']}")
    print(f"  - {OUTPUT_PATHS['xml_ready_json_path']}")
    print(f"  - {OUTPUT_PATHS['xml_path']}")
    print(f"  - {SCORE_TIMELINE_PATH}")
    print(f"Run directory: {OUTPUT_DIR}")

    return grouped_rallies


def calculate_points_score_pred(scoreboard_changes, pred_events):
    """Calculate points based on predicted events and scoreboard changes."""
    close_points = 0
    far_points = 0
    if not pred_events:
        return

    keys = sorted(pred_events.keys(), key=lambda x: int(x) if isinstance(x, str) else x)

    for rec in scoreboard_changes.values():
        frame_idx = rec['frame']

        for i, k in enumerate(keys):
            cur        = pred_events[k]
            cur_frame  = cur['frame_index']
            next_k     = keys[i + 1] if i + 1 < len(keys) else None
            next_frame = pred_events[next_k]['frame_index'] if next_k is not None else None

            if cur_frame <= frame_idx and (next_k is None or next_frame > frame_idx):
                event_type = cur['event_type']
                print(f"Frame {frame_idx} matched to event at frame {cur_frame} "
                      f"(key={k}), type: {event_type}, time {cur['time_in_mins']}")
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
    main(args)