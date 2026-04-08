import base64
import copy
import html
import json
import os
import shutil
import sys
import threading
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
from matplotlib.widgets import RectangleSelector
from PIL import Image
from tqdm import tqdm

from json_to_xml import write_xml_data


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_json(path: str) -> Any:
    return load_json(path)


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def split_xy_right_bottom(
    full_loc: Dict[str, float],
    table_w: float = 1525.0,
    table_h: float = 2740.0,
    mirror_far_x: bool = True,
    split_width: float = 153.0,
    split_length: float = 137.0,
) -> Dict[str, Any]:
    x = float(full_loc["x"])
    y = float(full_loc["y"])
    half_h = table_h / 2.0

    if y < half_h:
        table_half = "far"
        x_half = (table_w - x) if mirror_far_x else x
        y_half = y
    else:
        table_half = "close"
        x_half = x
        y_half = table_h - y

    return {
        "x": float(x_half),
        "y": float(y_half),
        "table_half": table_half,
    }


def compute_x2_y2_from_draw_location(
    draw_loc: Dict[str, float],
    table_w: float = 1525.0,
    table_h: float = 2740.0,
    mirror_far_x: bool = True,
    split_width: float = 153.0,
    split_length: float = 137.0,
) -> Tuple[float, float]:
    x = float(draw_loc["x"])
    y = float(draw_loc["y"])
    half_h = table_h / 2.0

    if y < half_h:
        x_player = (table_w - x) if mirror_far_x else x
        y_player = y
    else:
        x_player = x
        y_player = table_h - y

    x2 = ((table_w - x_player) / table_w) * split_width
    y2 = (y_player / half_h) * split_length
    return float(x2), float(y2)


def _attach_x2_y2(
    item: Dict[str, Any],
    draw: Dict[str, float],
    table_w: float,
    table_h: float,
    mirror_far_x: bool,
    split_width: float,
    split_length: float,
) -> None:
    x2, y2 = compute_x2_y2_from_draw_location(
        draw_loc=draw,
        table_w=table_w,
        table_h=table_h,
        mirror_far_x=mirror_far_x,
        split_width=split_width,
        split_length=split_length,
    )
    item["x2"] = x2
    item["y2"] = y2


def update_split_coordinates_in_enriched_file(
    enriched_path: str,
    table_w: float = 1525.0,
    table_h: float = 2740.0,
    mirror_far_x: bool = True,
    split_width: float = 153.0,
    split_length: float = 137.0,
) -> int:
    grouped = load_json(enriched_path)
    updated = 0

    for rally in grouped:
        for event in rally.get("events", []):
            for bounce in event.get("bounces", []):
                draw = bounce.get("draw_ball_location")
                if isinstance(draw, dict) and isinstance(draw.get("x"), (int, float)) and isinstance(draw.get("y"), (int, float)):
                    bounce["draw_ball_location_split"] = split_xy_right_bottom(
                        draw,
                        table_w=table_w,
                        table_h=table_h,
                        mirror_far_x=mirror_far_x,
                        split_width=split_width,
                        split_length=split_length,
                    )
                    _attach_x2_y2(
                        bounce,
                        draw,
                        table_w=table_w,
                        table_h=table_h,
                        mirror_far_x=mirror_far_x,
                        split_width=split_width,
                        split_length=split_length,
                    )
                    updated += 1

            draw_event = event.get("draw_ball_location")
            if str(event.get("event_type", "")).endswith("_bounce") and isinstance(draw_event, dict):
                if isinstance(draw_event.get("x"), (int, float)) and isinstance(draw_event.get("y"), (int, float)):
                    event["draw_ball_location_split"] = split_xy_right_bottom(
                        draw_event,
                        table_w=table_w,
                        table_h=table_h,
                        mirror_far_x=mirror_far_x,
                        split_width=split_width,
                        split_length=split_length,
                    )
                    _attach_x2_y2(
                        event,
                        draw_event,
                        table_w=table_w,
                        table_h=table_h,
                        mirror_far_x=mirror_far_x,
                        split_width=split_width,
                        split_length=split_length,
                    )
                    updated += 1

    save_json(enriched_path, grouped)
    return updated


def update_split_draw_coordinates(
    grouped_rallies: List[Dict[str, Any]],
    table_w: float = 1525.0,
    table_h: float = 2740.0,
    split_w: float = 153.0,
    split_l: float = 137.0,
    mirror_far_x: bool = True,
) -> Tuple[List[Dict[str, Any]], int]:
    updated = 0

    for rally in grouped_rallies:
        for event in rally.get("events", []):
            if not isinstance(event, dict):
                continue

            for bounce in event.get("bounces", []):
                draw = bounce.get("draw_ball_location") if isinstance(bounce, dict) else None
                if isinstance(draw, dict) and isinstance(draw.get("x"), (int, float)) and isinstance(draw.get("y"), (int, float)):
                    bounce["draw_ball_location_split"] = split_xy_right_bottom(
                        draw,
                        table_w=table_w,
                        table_h=table_h,
                        mirror_far_x=mirror_far_x,
                        split_width=split_w,
                        split_length=split_l,
                    )
                    _attach_x2_y2(
                        bounce,
                        draw,
                        table_w=table_w,
                        table_h=table_h,
                        mirror_far_x=mirror_far_x,
                        split_width=split_w,
                        split_length=split_l,
                    )
                    updated += 1

            draw_event = event.get("draw_ball_location")
            if str(event.get("event_type", "")).endswith("_bounce") and isinstance(draw_event, dict):
                if isinstance(draw_event.get("x"), (int, float)) and isinstance(draw_event.get("y"), (int, float)):
                    event["draw_ball_location_split"] = split_xy_right_bottom(
                        draw_event,
                        table_w=table_w,
                        table_h=table_h,
                        mirror_far_x=mirror_far_x,
                        split_width=split_w,
                        split_length=split_l,
                    )
                    _attach_x2_y2(
                        event,
                        draw_event,
                        table_w=table_w,
                        table_h=table_h,
                        mirror_far_x=mirror_far_x,
                        split_width=split_w,
                        split_length=split_l,
                    )
                    updated += 1

    return grouped_rallies, updated


def _event_to_player(event_type: str, close_player: str, far_player: str) -> str:
    if isinstance(event_type, str) and event_type.startswith("close_table"):
        return close_player
    if isinstance(event_type, str) and event_type.startswith("far_table"):
        return far_player
    return "NA"


def _event_to_shot(event_type: str) -> str:
    if not isinstance(event_type, str):
        return "NA"
    if event_type.endswith("serve"):
        return "Serve"
    if event_type.endswith("forehand"):
        return "Forehand"
    if event_type.endswith("backhand"):
        return "Backhand"
    if event_type.endswith("bounce"):
        return "Bounce"
    return "NA"


def _infer_point_won(rally: Dict[str, Any], close_player: str, far_player: str) -> str:
    old_score = rally.get("old_score", {}) or {}
    new_score = rally.get("new_score", {}) or {}

    old_close = old_score.get("close")
    old_far = old_score.get("far")
    new_close = new_score.get("close")
    new_far = new_score.get("far")

    if isinstance(old_close, (int, float)) and isinstance(new_close, (int, float)) and new_close > old_close:
        return f"{close_player} Win"
    if isinstance(old_far, (int, float)) and isinstance(new_far, (int, float)) and new_far > old_far:
        return f"{far_player} Win"
    return "NA"


def _find_xy_from_event_or_bounce(event: Dict[str, Any]) -> Dict[str, str]:
    if isinstance(event.get("x2"), (int, float)) and isinstance(event.get("y2"), (int, float)):
        return {
            "x2": str(event.get("x2")),
            "y2": str(event.get("y2")),
        }

    if isinstance(event.get("draw_ball_location_split"), dict):
        xy = event["draw_ball_location_split"]
        return {
            "x2": str(xy.get("x", "NA")) if xy.get("x") is not None else "NA",
            "y2": str(xy.get("y", "NA")) if xy.get("y") is not None else "NA",
        }

    bounces = event.get("bounces", [])
    if isinstance(bounces, list):
        for bounce in bounces:
            if isinstance(bounce, dict) and isinstance(bounce.get("x2"), (int, float)) and isinstance(bounce.get("y2"), (int, float)):
                return {
                    "x2": str(bounce.get("x2")),
                    "y2": str(bounce.get("y2")),
                }
            if isinstance(bounce, dict) and isinstance(bounce.get("draw_ball_location_split"), dict):
                xy = bounce["draw_ball_location_split"]
                return {
                    "x2": str(xy.get("x", "NA")) if xy.get("x") is not None else "NA",
                    "y2": str(xy.get("y", "NA")) if xy.get("y") is not None else "NA",
                }

    return {"x2": "NA", "y2": "NA"}


def _rally_start_end(rally: Dict[str, Any]) -> Dict[str, str]:
    start = rally.get("serve_event_time")
    end = rally.get("score_change_time")

    if not isinstance(start, (int, float)):
        start = min([
            float(e.get("time")) for e in rally.get("events", [])
            if isinstance(e, dict) and isinstance(e.get("time"), (int, float))
        ] or [0.0])

    if not isinstance(end, (int, float)):
        end = max([
            float(e.get("time")) for e in rally.get("events", [])
            if isinstance(e, dict) and isinstance(e.get("time"), (int, float))
        ] or [start])

    if end < start:
        end = start

    return {"start": f"{start:.2f}", "end": f"{end:.2f}"}





def _build_common_labels(rally: Dict[str, Any], point_number: int, game_label: str, close_player: str, far_player: str) -> List[Dict[str, str]]:
    old_score = rally.get("old_score", {}) or {}
    point_won = _infer_point_won(rally, close_player, far_player)

    return [
        {"group": f"{close_player} Score", "text": str(old_score.get("close", "NA"))},
        {"group": f"{far_player} Score", "text": str(old_score.get("far", "NA"))},
        {"group": "Game", "text": game_label},
        {"group": "Point Number", "text": str(point_number)},
        {"group": "Hand", "text": "NA"},
        {"group": "Serve Type", "text": "NA"},
        {"group": "Point Won", "text": point_won},
    ]



def _build_single_shot_label(event, rally_shot_number):
    shot_name = _event_to_shot(event.get("event_type", ""))
    xy = _find_xy_from_event_or_bounce(event)

    label = ([
        {"group": "Rally Shot Number", "text": str(rally_shot_number)},
        {"group": "Rally Shot", "text": shot_name},
        {"group": "Rally Stroke", "text": shot_name},
        {"group": "x2", "text": xy["x2"]},
        {"group": "y2", "text": xy["y2"]},
        {"group": "Error Type", "text": "NA"},
        {"group": "Error", "text": "NA"},
        {"group": "Error Area", "text": "NA"},
    ])
    
    return label


    

    

def _build_rally_shot_labels(rally: Dict[str, Any]) -> List[Dict[str, str]]:
    labels: List[Dict[str, str]] = []
    shot_no = 1

    for event in rally.get("events", []):
        if not isinstance(event, dict):
            continue
        shot_name = _event_to_shot(event.get("event_type", ""))
        xy = _find_xy_from_event_or_bounce(event)

        labels.extend([
            {"group": "Rally Shot Number", "text": str(shot_no)},
            {"group": "Rally Shot", "text": shot_name},
            {"group": "Rally Stroke", "text": shot_name},
            {"group": "x2", "text": xy["x2"]},
            {"group": "y2", "text": xy["y2"]},
            {"group": "Error Type", "text": "NA"},
            {"group": "Error", "text": "NA"},
            {"group": "Error Area", "text": "NA"},
        ])
        shot_no += 1

    if shot_no == 1:
        labels.extend([
            {"group": "Rally Shot Number", "text": "NA"},
            {"group": "Rally Shot", "text": "NA"},
            {"group": "Rally Stroke", "text": "NA"},
            {"group": "x2", "text": "NA"},
            {"group": "y2", "text": "NA"},
            {"group": "Error Type", "text": "NA"},
            {"group": "Error", "text": "NA"},
            {"group": "Error Area", "text": "NA"},
        ])

    return labels



def _add_seconds(time_val: float, seconds: int = 1) -> float:
    whole = int(time_val) + seconds
    centis = time_val - int(time_val)
    return round(whole + centis, 2)

def convert_enriched_to_output(
    enriched_data: List[Dict[str, Any]],
    game_label: str,
    close_player: str,
    far_player: str,
) -> Dict[str, Any]:
    instances: List[Dict[str, Any]] = []
    instance_id = 1

    for idx, rally in enumerate(enriched_data, start=1):
        if not isinstance(rally, dict):
            continue

        point_number = int(rally.get("rally_id", idx))
        times = _rally_start_end(rally)
        events = [e for e in rally.get("events", []) if isinstance(e, dict)]

        serve_event = next((e for e in events if str(e.get("event_type", "")).endswith("serve")), None)
        serve_player = _event_to_player(serve_event.get("event_type", "") if serve_event else "", close_player, far_player)
        serve_code = f"{serve_player} Serve" if serve_player != "NA" else "NA Serve"

        common_labels = _build_common_labels(rally, point_number, game_label, close_player, far_player)
        shot_labels = _build_rally_shot_labels(rally)

        instances.append({
            "ID": str(instance_id),
            "start": times["start"],
            "end": times["end"],
            "code": serve_code,
            "label": common_labels + shot_labels,
        })
        instance_id += 1

        instances.append({
            "ID": str(instance_id),
            "start": times["start"],
            "end": times["end"],
            "code": "Point",
            "label": common_labels + shot_labels,
        })
        instance_id += 1

        instances.append({
            "ID": str(instance_id),
            "start": times["start"],
            "end": times["end"],
            "code": "All",
            "label": [
                {"group": f"{close_player} Score", "text": "NA"},
                {"group": f"{far_player} Score", "text": "NA"},
                {"group": "Game", "text": "NA"},
                {"group": "Point Number", "text": "NA"},
                {"group": "Hand", "text": "NA"},
                {"group": "Serve Type", "text": "NA"},
                {"group": "Point Won", "text": "NA"},
                {"group": "Rally Shot Number", "text": "NA"},
                {"group": "Rally Shot", "text": "NA"},
                {"group": "Rally Stroke", "text": "NA"},
                {"group": "x2", "text": "NA"},
                {"group": "y2", "text": "NA"},
                {"group": "Error Type", "text": "NA"},
                {"group": "Error", "text": "NA"},
                {"group": "Error Area", "text": "NA"},
            ],
        })
        instance_id += 1


        # --- Per-shot instances for a single rally
        rally_shot_number = 1
        for event in events:
            event_type = str(event.get("event_type", ""))
            if not event_type:
                continue
            
            # Keep shot contacts, including serve.
            if not (
                event_type.endswith("serve")
                or event_type.endswith("forehand")
                or event_type.endswith("backhand")
            ):
                continue

            player = _event_to_player(event_type, close_player, far_player)
            if player == "NA":
                continue

            # Get shot-level timing if available, else fall back to rally times.
            raw_time = event.get("time")
            if isinstance(raw_time, (int, float)):
                shot_start_num = float(raw_time)
            else:
                shot_start_num = float(times["start"])

            shot_end_num = _add_seconds(shot_start_num)
            shot_start = f"{shot_start_num:.2f}"
            shot_end = f"{shot_end_num:.2f}"

            

            shot_code = f"{player} Shot"

            # Build shot-specific labels (same structure as serve but for this event).
            shot_instance_labels = _build_common_labels(rally, point_number, game_label, close_player, far_player)

            
            shot_instance_labels += _build_single_shot_label(event, rally_shot_number)
           
            instances.append({
                "ID": str(instance_id),
                "start": shot_start,
                "end":   shot_end,
                "code":  shot_code,
                "label": shot_instance_labels,
            })

            rally_shot_number += 1
            instance_id += 1

    return {"file": {"ALL_INSTANCES": {"instance": instances}}}


def convert_grouped_to_output(
    grouped_rallies: List[Dict[str, Any]],
    fps: float,
    game: str,
    close_player: str,
    far_player: str,
    include_bounces: bool = True,
) -> Dict[str, Any]:
    return convert_enriched_to_output(
        enriched_data=grouped_rallies,
        game_label=game,
        close_player=close_player,
        far_player=far_player,
    )


def build_final_output_paths(game_name: str, output_dir: str = ".") -> Dict[str, str]:
    base_dir = Path(output_dir)
    return {
        "summary_path": str(base_dir / f"final_rally_summary_{game_name}.json"),
        "xml_ready_json_path": str(base_dir / f"xml_ready_{game_name}.json"),
        "xml_path": str(base_dir / f"xml_ready_{game_name}.xml"),
    }


def remove_files(paths: Optional[List[str]]) -> List[str]:
    removed: List[str] = []
    if not paths:
        return removed

    for path in paths:
        if not path:
            continue
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)

    return removed


def export_final_artifacts(
    grouped_rallies: List[Dict[str, Any]],
    game_name: str,
    game_label: str,
    close_player: str,
    far_player: str,
    fps: float = 30.0,
    output_dir: str = ".",
    summary_path: Optional[str] = None,
    xml_ready_json_path: Optional[str] = None,
    xml_path: Optional[str] = None,
    nest_bounces_under_hits: bool = True,
    include_bounces_in_output: bool = True,
    table_w: float = 1525.0,
    table_h: float = 2740.0,
    split_w: float = 153.0,
    split_l: float = 137.0,
    cleanup_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    paths = build_final_output_paths(game_name=game_name, output_dir=output_dir)
    summary_path = summary_path or paths["summary_path"]
    xml_ready_json_path = xml_ready_json_path or paths["xml_ready_json_path"]
    xml_path = xml_path or paths["xml_path"]

    export_source = copy.deepcopy(grouped_rallies)
    if nest_bounces_under_hits:
        export_source = reorganize_rallies_with_bounces_under_hits(export_source)

    enriched_grouped, updated_bounces = update_split_draw_coordinates(
        grouped_rallies=export_source,
        table_w=table_w,
        table_h=table_h,
        split_w=split_w,
        split_l=split_l,
    )

    # Persist summary after bounce nesting and split-coordinate enrichment
    # so final_rally_summary contains grouped bounces and x2/y2.
    save_json(summary_path, enriched_grouped)

    output_payload = convert_grouped_to_output(
        grouped_rallies=enriched_grouped,
        fps=fps,
        game=game_label,
        close_player=close_player,
        far_player=far_player,
        include_bounces=include_bounces_in_output,
    )

    save_json(xml_ready_json_path, output_payload)
    write_xml_data(output_payload, xml_path)
    removed_files = remove_files(cleanup_paths)

    return {
        "summary_path": summary_path,
        "xml_ready_json_path": xml_ready_json_path,
        "xml_path": xml_path,
        "updated_bounces": updated_bounces,
        "instance_count": len(output_payload["file"]["ALL_INSTANCES"]["instance"]),
        "removed_files": removed_files,
    }


def save_new_output_file(payload: Dict[str, Any], prefix: str = "grouped_to_output_game2_from_enriched") -> str:
    filename = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    if os.path.basename(filename).lower() == "output.json":
        raise ValueError("Refusing to overwrite output.json (golden standard).")
    save_json(filename, payload)
    return filename


def read_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image


def show_cropped_image(img, coord, save_path: Optional[str] = None):
    x1, y1, x2, y2 = coord
    cropped = img[y1:y2, x1:x2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    original_rgb = img[:, :, ::-1].copy()
    cv2.rectangle(original_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original — ROI box highlighted")
    axes[0].axis("off")

    axes[1].imshow(cropped[:, :, ::-1])
    axes[1].set_title(f"Cropped ROI  ({x2-x1}w × {y2-y1}h px)")
    axes[1].axis("off")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"Saved ROI preview to {save_path}")
    plt.show()


def save_table_points_preview(frame, points, save_path: str, title: str = "Selected table points"):
    if frame is None or len(points) != 6:
        raise ValueError("Expected a frame and 6 selected table points.")

    img = frame[:, :, ::-1].copy() if (len(frame.shape) == 3 and frame.shape[2] == 3) else frame.copy()
    point_labels = ["TL", "TR", "mid-left", "mid-right", "BL", "BR"]
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

    for (x, y), label, color in zip(points, point_labels, colors):
        ax.plot(x, y, "o", color=color, markersize=10, zorder=5)
        ax.text(
            x + 8,
            y - 8,
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            zorder=6,
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
        )

    tl, tr, ml, mr, bl, br = points
    for start, end in [(tl, tr), (bl, br), (tl, bl), (tr, br), (ml, mr)]:
        ax.plot([start[0], end[0]], [start[1], end[1]], "--", color="yellow", linewidth=1.5, alpha=0.8, zorder=4)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved table point preview to {save_path}")


def _is_colab_environment() -> bool:
    return "google.colab" in sys.modules or bool(os.environ.get("COLAB_RELEASE_TAG"))


def _show_selection_reference(frame, title: str, existing=None, points=None, point_labels=None, colors=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(frame)
    ax.set_title(title)

    height, width = frame.shape[:2]
    step = max(50, min(height, width) // 8)
    ax.set_xticks(np.arange(0, width + 1, step))
    ax.set_yticks(np.arange(0, height + 1, step))
    ax.grid(color="white", alpha=0.35, linewidth=0.8)

    if existing is not None:
        x1, y1, x2, y2 = map(int, existing)
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
        ax.add_patch(rect)

    if points:
        if point_labels is None:
            point_labels = [str(idx + 1) for idx in range(len(points))]
        if colors is None:
            colors = ["red"] * len(points)

        for (x, y), label, color in zip(points, point_labels, colors):
            ax.plot(x, y, "o", color=color, markersize=10, zorder=5)
            ax.text(
                x + 8,
                y - 8,
                label,
                color=color,
                fontsize=10,
                fontweight="bold",
                zorder=6,
                bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
            )

    plt.tight_layout()
    plt.show()


def _frame_to_data_url(frame: np.ndarray) -> str:
    image = Image.fromarray(frame.astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _wait_for_colab_selection(html_content: str, callback_name: str, timeout: int = 3600):
    from IPython.display import HTML, display

    output = __import__("google.colab.output", fromlist=["output"])

    result: Dict[str, Any] = {"value": None}
    selection_ready = threading.Event()

    def _callback(payload):
        result["value"] = payload
        selection_ready.set()

    output.register_callback(callback_name, _callback)
    display(HTML(html_content))

    if not selection_ready.wait(timeout=timeout):
        raise TimeoutError("Timed out waiting for Colab interactive selection.")

    return result["value"]


def _select_roi_colab_interactive(frame, title="Select ROI", existing=None):
        callback_name = f"notebook_helpers_roi_{uuid.uuid4().hex}"
        container_id = f"roi_container_{uuid.uuid4().hex}"
        canvas_id = f"roi_canvas_{uuid.uuid4().hex}"
        data_url = _frame_to_data_url(frame)
        existing_json = json.dumps(
                None if existing is None else {
                        "x1": int(existing[0]),
                        "y1": int(existing[1]),
                        "x2": int(existing[2]),
                        "y2": int(existing[3]),
                }
        )

        html_content = f"""
<div id=\"{container_id}\" style=\"font-family: sans-serif;\">
    <div style=\"font-weight: 600; margin-bottom: 8px;\">{html.escape(title)}</div>
    <div style=\"margin-bottom: 8px;\">Drag on the image to draw the ROI, then click Confirm ROI.</div>
    <canvas id=\"{canvas_id}\" style=\"max-width: 100%; border: 1px solid #ccc; cursor: crosshair;\"></canvas>
    <div style=\"margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;\">
        <button id=\"{canvas_id}_confirm\">Confirm ROI</button>
        <button id=\"{canvas_id}_clear\">Clear</button>
        <button id=\"{canvas_id}_existing\">Use Existing</button>
        <button id=\"{canvas_id}_cancel\">Cancel</button>
    </div>
    <div id=\"{canvas_id}_status\" style=\"margin-top: 8px; color: #444;\"></div>
</div>
<script>
(() => {{
    const callbackName = {json.dumps(callback_name)};
    const imageSrc = {json.dumps(data_url)};
    const existing = {existing_json};
    const root = document.getElementById({json.dumps(container_id)});
    const canvas = document.getElementById({json.dumps(canvas_id)});
    const ctx = canvas.getContext('2d');
    const status = document.getElementById({json.dumps(canvas_id + '_status')});
    const image = new Image();
    let dragging = false;
    let start = null;
    let currentRect = existing ? {{...existing}} : null;

    function disableButtons() {{
        root.querySelectorAll('button').forEach((button) => {{
            button.disabled = true;
        }});
    }}

    function sendSelection(payload) {{
        disableButtons();
        google.colab.kernel.invokeFunction(callbackName, [payload], {{}});
    }}

    function getMousePosition(event) {{
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {{
            x: Math.max(0, Math.min(canvas.width - 1, Math.round((event.clientX - rect.left) * scaleX))),
            y: Math.max(0, Math.min(canvas.height - 1, Math.round((event.clientY - rect.top) * scaleY))),
        }};
    }}

    function normalizeRect(rect) {{
        const x1 = Math.min(rect.x1, rect.x2);
        const y1 = Math.min(rect.y1, rect.y2);
        const x2 = Math.max(rect.x1, rect.x2);
        const y2 = Math.max(rect.y1, rect.y2);
        return {{x1, y1, x2, y2}};
    }}

    function draw() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);
        if (!currentRect) {{
            status.textContent = 'No ROI selected yet.';
            return;
        }}
        const rect = normalizeRect(currentRect);
        ctx.strokeStyle = '#00b300';
        ctx.lineWidth = 3;
        ctx.strokeRect(rect.x1, rect.y1, rect.x2 - rect.x1, rect.y2 - rect.y1);
        status.textContent = `ROI: (${{rect.x1}}, ${{rect.y1}}) -> (${{rect.x2}}, ${{rect.y2}})`;
    }}

    image.onload = () => {{
        canvas.width = image.width;
        canvas.height = image.height;
        draw();
    }};
    image.src = imageSrc;

    canvas.addEventListener('mousedown', (event) => {{
        const point = getMousePosition(event);
        start = point;
        currentRect = {{x1: point.x, y1: point.y, x2: point.x, y2: point.y}};
        dragging = true;
        draw();
    }});

    canvas.addEventListener('mousemove', (event) => {{
        if (!dragging || !start) return;
        const point = getMousePosition(event);
        currentRect = {{x1: start.x, y1: start.y, x2: point.x, y2: point.y}};
        draw();
    }});

    window.addEventListener('mouseup', () => {{
        dragging = false;
    }});

    document.getElementById({json.dumps(canvas_id + '_confirm')}).onclick = () => {{
        if (!currentRect) {{
            status.textContent = 'Draw an ROI first.';
            return;
        }}
        const rect = normalizeRect(currentRect);
        if (rect.x1 === rect.x2 || rect.y1 === rect.y2) {{
            status.textContent = 'ROI must have non-zero width and height.';
            return;
        }}
        sendSelection({{...rect, done: true}});
    }};

    document.getElementById({json.dumps(canvas_id + '_clear')}).onclick = () => {{
        currentRect = null;
        draw();
    }};

    document.getElementById({json.dumps(canvas_id + '_existing')}).onclick = () => {{
        if (!existing) {{
            status.textContent = 'No existing ROI available.';
            return;
        }}
        currentRect = {{...existing}};
        draw();
    }};

    document.getElementById({json.dumps(canvas_id + '_cancel')}).onclick = () => {{
        sendSelection({{x1: null, y1: null, x2: null, y2: null, done: false}});
    }};
}})();
</script>
"""

        return _wait_for_colab_selection(html_content, callback_name)


def _click_6_points_colab_interactive(frame, title="Click 6 points: TL, TR, mid-left, mid-right, BL, BR"):
        callback_name = f"notebook_helpers_points_{uuid.uuid4().hex}"
        container_id = f"points_container_{uuid.uuid4().hex}"
        canvas_id = f"points_canvas_{uuid.uuid4().hex}"
        data_url = _frame_to_data_url(frame)
        point_labels = ["TL", "TR", "mid-left", "mid-right", "BL", "BR"]
        point_labels_json = json.dumps(point_labels)
        colors_json = json.dumps(["red", "blue", "green", "orange", "purple", "cyan"])

        html_content = f"""
<div id=\"{container_id}\" style=\"font-family: sans-serif;\">
    <div style=\"font-weight: 600; margin-bottom: 8px;\">{html.escape(title)}</div>
    <div style=\"margin-bottom: 8px;\">Click the six table points in order, then click Confirm Points.</div>
    <canvas id=\"{canvas_id}\" style=\"max-width: 100%; border: 1px solid #ccc; cursor: crosshair;\"></canvas>
    <div style=\"margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;\">
        <button id=\"{canvas_id}_confirm\">Confirm Points</button>
        <button id=\"{canvas_id}_undo\">Undo</button>
        <button id=\"{canvas_id}_reset\">Reset</button>
        <button id=\"{canvas_id}_cancel\">Cancel</button>
    </div>
    <div id=\"{canvas_id}_status\" style=\"margin-top: 8px; color: #444;\"></div>
</div>
<script>
(() => {{
    const callbackName = {json.dumps(callback_name)};
    const imageSrc = {json.dumps(data_url)};
    const labels = {point_labels_json};
    const colors = {colors_json};
    const root = document.getElementById({json.dumps(container_id)});
    const canvas = document.getElementById({json.dumps(canvas_id)});
    const ctx = canvas.getContext('2d');
    const status = document.getElementById({json.dumps(canvas_id + '_status')});
    const image = new Image();
    const points = [];

    function disableButtons() {{
        root.querySelectorAll('button').forEach((button) => {{
            button.disabled = true;
        }});
    }}

    function sendSelection(payload) {{
        disableButtons();
        google.colab.kernel.invokeFunction(callbackName, [payload], {{}});
    }}

    function getMousePosition(event) {{
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {{
            x: Math.max(0, Math.min(canvas.width - 1, Math.round((event.clientX - rect.left) * scaleX))),
            y: Math.max(0, Math.min(canvas.height - 1, Math.round((event.clientY - rect.top) * scaleY))),
        }};
    }}

    function draw() {{
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(image, 0, 0);
        points.forEach((point, index) => {{
            ctx.fillStyle = colors[index];
            ctx.beginPath();
            ctx.arc(point.x, point.y, 6, 0, Math.PI * 2);
            ctx.fill();
            ctx.fillStyle = colors[index];
            ctx.font = '16px sans-serif';
            ctx.fillText(labels[index], point.x + 8, point.y - 8);
        }});
        if (points.length > 0) {{
            status.textContent = `Selected ${{points.length}} / 6 points. Next: ${{labels[Math.min(points.length, 5)]}}`;
        }} else {{
            status.textContent = `Selected 0 / 6 points. Next: ${{labels[0]}}`;
        }}
        if (points.length === 6) {{
            status.textContent = 'All 6 points selected. Click Confirm Points.';
        }}
    }}

    image.onload = () => {{
        canvas.width = image.width;
        canvas.height = image.height;
        draw();
    }};
    image.src = imageSrc;

    canvas.addEventListener('click', (event) => {{
        if (points.length >= 6) return;
        const point = getMousePosition(event);
        points.push(point);
        draw();
    }});

    document.getElementById({json.dumps(canvas_id + '_undo')}).onclick = () => {{
        if (points.length > 0) {{
            points.pop();
            draw();
        }}
    }};

    document.getElementById({json.dumps(canvas_id + '_reset')}).onclick = () => {{
        points.length = 0;
        draw();
    }};

    document.getElementById({json.dumps(canvas_id + '_confirm')}).onclick = () => {{
        if (points.length !== 6) {{
            status.textContent = 'Select all 6 points before confirming.';
            return;
        }}
        sendSelection({{points, done: true}});
    }};

    document.getElementById({json.dumps(canvas_id + '_cancel')}).onclick = () => {{
        sendSelection({{points: [], done: false}});
    }};
}})();
</script>
"""

        return _wait_for_colab_selection(html_content, callback_name)


def _prompt_for_int_values(prompt: str, expected_count: int, allow_blank: bool = False):
    while True:
        raw = input(prompt).strip()
        if not raw:
            if allow_blank:
                return None
            print("Input cannot be blank.")
            continue

        parts = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
        if len(parts) != expected_count:
            print(f"Expected {expected_count} comma-separated values.")
            continue

        try:
            return [int(round(float(part))) for part in parts]
        except ValueError:
            print("Please enter numeric values only.")


def _select_roi_colab(frame, title="Select ROI", existing=None, allow_cancel=True):
    try:
        selection = _select_roi_colab_interactive(frame, title=title, existing=existing)
        if selection is not None:
            return selection
    except Exception as exc:
        raise RuntimeError(
            "Colab interactive ROI selection is unavailable. Ensure you are running in the Colab browser frontend "
            "with JavaScript enabled, then rerun the cell. Manual coordinate fallback has been disabled."
        ) from exc

    raise RuntimeError("Colab interactive ROI selection did not return a result.")


def _click_6_points_colab(frame, title="Click 6 points: TL, TR, mid-left, mid-right, BL, BR"):
    try:
        selection = _click_6_points_colab_interactive(frame, title=title)
        if selection is not None:
            points = [tuple(map(int, point)) for point in selection.get("points", [])]
            return {"points": points, "done": bool(selection.get("done", False))}
    except Exception as exc:
        raise RuntimeError(
            "Colab interactive table-point selection is unavailable. Ensure you are running in the Colab browser frontend "
            "with JavaScript enabled, then rerun the cell. Manual coordinate fallback has been disabled."
        ) from exc

    raise RuntimeError("Colab interactive table-point selection did not return a result.")


def select_roi_jupyter(frame, title="Select ROI", existing=None, allow_cancel=True):
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame provided to ROI selector.")

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        img = frame[:, :, ::-1].copy()
    else:
        img = frame.copy()

    roi_coords = {"x1": None, "y1": None, "x2": None, "y2": None, "done": False}

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f"{title}\nClick and drag to select ROI, then close the window")
    ax.set_navigate(False)

    if existing is not None:
        x1, y1, x2, y2 = map(int, existing)
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
            label="Existing ROI",
        )
        ax.add_patch(rect)
        ax.legend(loc="upper left")

    def on_select(eclick, erelease):
        roi_coords["x1"] = int(min(eclick.xdata, erelease.xdata))
        roi_coords["y1"] = int(min(eclick.ydata, erelease.ydata))
        roi_coords["x2"] = int(max(eclick.xdata, erelease.xdata))
        roi_coords["y2"] = int(max(eclick.ydata, erelease.ydata))
        roi_coords["done"] = True

    selector = RectangleSelector(
        ax,
        on_select,
        useblit=False,
        button=[1],
        minspanx=5,
        minspany=5,
        spancoords="pixels",
        interactive=True,
    )

    plt.tight_layout()
    plt.show()
    fig._roi_selector = selector
    return roi_coords


def get_roi_result(roi_coords, allow_cancel=True):
    x1, y1, x2, y2 = roi_coords["x1"], roi_coords["y1"], roi_coords["x2"], roi_coords["y2"]

    if not roi_coords["done"] or x1 is None:
        if allow_cancel:
            return None
        raise RuntimeError("ROI selection cancelled / invalid.")

    return (x1, y1, x2, y2)


def click_6_points_jupyter(frame, title="Click 6 points: TL, TR, mid-left, mid-right, BL, BR"):
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame provided.")

    img = frame[:, :, ::-1].copy() if (len(frame.shape) == 3 and frame.shape[2] == 3) else frame.copy()

    point_labels = ["TL", "TR", "mid-left", "mid-right", "BL", "BR"]
    colors = ["red", "blue", "green", "orange", "purple", "cyan"]
    state = {"points": [], "done": False}

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img)
    ax.set_title(f"[1/6] Click: {point_labels[0]}", fontsize=13)
    ax.set_navigate(False)

    status_text = ax.text(
        10,
        20,
        f"Next: {point_labels[0]}  (0/6)",
        color="white",
        fontsize=11,
        fontweight="bold",
        bbox=dict(facecolor="black", alpha=0.6, boxstyle="round"),
    )

    def on_click(event):
        if event.inaxes != ax or event.button != 1:
            return
        if state["done"]:
            return

        x, y = int(event.xdata), int(event.ydata)
        idx = len(state["points"])
        label = point_labels[idx]
        color = colors[idx]

        state["points"].append((x, y))
        ax.plot(x, y, "o", color=color, markersize=10, zorder=5)
        ax.text(
            x + 8,
            y - 8,
            label,
            color=color,
            fontsize=10,
            fontweight="bold",
            zorder=6,
            bbox=dict(facecolor="black", alpha=0.5, boxstyle="round,pad=0.2"),
        )

        if len(state["points"]) == 6:
            state["done"] = True
            ax.set_title("✅ All 6 points collected! Run next cell.", fontsize=13, color="green")
            status_text.set_text("Done! All 6 points collected.")

            pts = state["points"]
            tl, tr, ml, mr, bl, br = pts
            for start, end in [(tl, tr), (bl, br), (tl, bl), (tr, br), (ml, mr)]:
                ax.plot([start[0], end[0]], [start[1], end[1]], "--", color="yellow", linewidth=1.5, alpha=0.8, zorder=4)
        else:
            next_label = point_labels[len(state["points"])]
            ax.set_title(f"[{len(state['points'])+1}/6] Click: {next_label}", fontsize=13)
            status_text.set_text(f"Next: {next_label}  ({len(state['points'])}/6)")

        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()
    return state


def reorganize_rallies_with_bounces_under_hits(grouped_rallies):
    def _linearize_events(events):
        linear = []
        for event in events:
            if not isinstance(event, dict):
                continue

            event_copy = dict(event)
            nested_bounces = event_copy.pop("bounces", None)
            linear.append(event_copy)

            if isinstance(nested_bounces, list):
                linear.extend(_linearize_events(nested_bounces))
        return linear

    def _bounce_matches_previous(bounce_event_type, previous_event_type):
        if previous_event_type.startswith("far"):
            return bounce_event_type.startswith("close")
        elif previous_event_type.startswith("close"):
            return bounce_event_type.startswith("far")
        return True  # if previous doesn't start with far/close, no restriction

    for rally in grouped_rallies:
        events = rally["events"]
        linear_events = _linearize_events(events)
        reorganized_events = []
        previous_non_bounce_event = None

        for event in linear_events:
            event_type = str(event.get("event_type", ""))

            if "bounce" in event_type:
                if previous_non_bounce_event is not None:
                    previous_event_type = str(previous_non_bounce_event.get("event_type", ""))
                    if not _bounce_matches_previous(event_type, previous_event_type):
                        continue  # drop the bounce if it doesn't match
                    if "bounces" not in previous_non_bounce_event or not isinstance(previous_non_bounce_event.get("bounces"), list):
                        previous_non_bounce_event["bounces"] = []
                    previous_non_bounce_event["bounces"].append(event)
                else:
                    event.setdefault("bounces", [])
                    reorganized_events.append(event)
            else:
                event.setdefault("bounces", [])
                reorganized_events.append(event)
                previous_non_bounce_event = event

        rally["events"] = reorganized_events

    return grouped_rallies





def filter_based_on_score_changes(pred_events, scoreboard_changes, fps_rate: float):
    """
    scoreboard_changes: a list of dicts, each dict has keys: 'frame', 'new' (which contains the new scores for close and far)
    use scores to filter out those fake serves that does not change the score
    So the logic is, we have the time when the score changes,
    we look for the closest serve event (either close_table_serve or far_table_serve) 
    that happens before the score change, we group it as a segment of the rally 
    once all the events are grouped into rallies, we will remove all other serves or events that are not in 
    """

    grouped_rallies = []
    rally_counter = 1

    for score_change in scoreboard_changes:
        frame_idx = score_change['frame']
        old_score_close = score_change['old']['close']['score']
        old_score_far = score_change['old']['far']['score']
        new_score_close = score_change['new']['close']['score']
        new_score_far = score_change['new']['far']['score']
        closest_event = None

        # find the closest serve event before the score change
        for event_frame, event_info in pred_events.items():
            if event_frame <= frame_idx and event_info['event_type'] in ['close_table_serve', 'far_table_serve']:
                closest_event = event_frame
            elif event_frame > frame_idx:
                break
        
        if closest_event is not None:
            print(f"Score change at frame {frame_idx} matched to serve event at frame {closest_event}, type: {pred_events[closest_event]['event_type']}")
            # group events into rallies based on the closest serve event
            rally_events = []
            for event_frame, event_info in pred_events.items():
                if event_frame >= closest_event and event_frame <= frame_idx:
                    rally_events.append(event_info)
            grouped_rallies.append({
                'rally_id': rally_counter,
                'old_score': {
                    'close': old_score_close,
                    'far': old_score_far
                },
                'new_score': {
                    'close': new_score_close,
                    'far': new_score_far
                },
                'score_change_frame': frame_idx,
                'score_change_time': frame_idx / fps_rate,
                'serve_event_frame': closest_event,
                'serve_event_time': closest_event / fps_rate,
                'duration': frame_idx / fps_rate - closest_event / fps_rate,
                'events': rally_events
            })
            rally_counter += 1
    
    return grouped_rallies


def _count_events_list(event_list, counts: dict):
    for e in event_list:
        if isinstance(e, dict):
            et = e.get('event_type')
            if et:
                counts[et] = counts.get(et, 0) + 1

            bounces = e.get('bounces', [])
            if isinstance(bounces, list):
                _count_events_list(bounces, counts)

            nested_events = e.get('events', [])
            if isinstance(nested_events, list):
                _count_events_list(nested_events, counts)
        elif isinstance(e, list):
            _count_events_list(e, counts)


def _iter_bounces(events):
    for e in events:
        if not isinstance(e, dict):
            continue

        et = str(e.get('event_type', ''))
        if 'bounce' in et:
            yield e

        bounces = e.get('bounces', [])
        if isinstance(bounces, list):
            for b in bounces:
                if isinstance(b, dict) and 'bounce' in str(b.get('event_type', '')):
                    yield b


def _is_invalid_bounce(b, split_w: float, split_l: float, table_w: float, table_h: float):
    if isinstance(b.get('x2'), (int, float)) and isinstance(b.get('y2'), (int, float)):
        x2, y2 = float(b['x2']), float(b['y2'])
        return (x2 < 0) or (x2 > split_w) or (y2 < 0) or (y2 > split_l)

    draw = b.get('draw_ball_location')
    if isinstance(draw, dict) and isinstance(draw.get('x'), (int, float)) and isinstance(draw.get('y'), (int, float)):
        x, y = float(draw['x']), float(draw['y'])
        return (x < 0) or (x > table_w) or (y < 0) or (y > table_h)

    return True


def summarize_rallies_counts(grouped_rallies, split_w: float = 153.0, split_l: float = 137.0, table_w: float = 1525.0, table_h: float = 2740.0):
    summaries = []
    invalid_bounce_total = 0
    bounce_total = 0

    for r in grouped_rallies:
        counts = {}
        events = r.get('events', [])
        _count_events_list(events, counts)

        rally_bounces = list(_iter_bounces(events))
        rally_invalid_bounces = sum(1 for b in rally_bounces if _is_invalid_bounce(b, split_w, split_l, table_w, table_h))

        close_bounce = sum(1 for b in rally_bounces if str(b.get('event_type', '')).startswith('close_table'))
        far_bounce = sum(1 for b in rally_bounces if str(b.get('event_type', '')).startswith('far_table'))

        close_fore = counts.get('close_table_forehand', 0)
        far_fore = counts.get('far_table_forehand', 0)
        close_back = counts.get('close_table_backhand', 0)
        far_back = counts.get('far_table_backhand', 0)

        bounce_total += len(rally_bounces)
        invalid_bounce_total += rally_invalid_bounces

        summary = {
            'rally_id': r.get('rally_id'),
            'serve_event_frame': r.get('serve_event_frame'),
            'score_change_frame': r.get('score_change_frame'),
            'duration_s': r.get('duration'),
            'close_bounce': int(close_bounce),
            'far_bounce': int(far_bounce),
            'close_forehand': int(close_fore),
            'far_forehand': int(far_fore),
            'close_backhand': int(close_back),
            'far_backhand': int(far_back),
            'total_forehand': int(close_fore + far_fore),
            'total_backhand': int(close_back + far_back),
            'total_bounces': int(close_bounce + far_bounce),
            'invalid_bounces': int(rally_invalid_bounces),
        }
        summaries.append(summary)

    return summaries, invalid_bounce_total, bounce_total


def extract_frames_with_retry(video_path: str, extract_frames_fn):
    frame_dir, fps_rate = extract_frames_fn(video_path)

    frame_images = [
        name for name in os.listdir(frame_dir)
        if os.path.isfile(os.path.join(frame_dir, name)) and name.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(frame_images) == 0:
        print(f"Frame directory exists but is empty: {frame_dir}. Re-extracting frames...")
        shutil.rmtree(frame_dir, ignore_errors=True)
        frame_dir, fps_rate = extract_frames_fn(video_path)

    return frame_dir, fps_rate


def prepare_video_inputs(
    video_path: str,
    device: str,
    extract_frames_fn,
    read_image_fn=None,
    rng=None,
):
    if read_image_fn is None:
        read_image_fn = read_image
    if rng is None:
        rng = np.random

    frame_dir, fps_rate = extract_frames_with_retry(video_path, extract_frames_fn)
    game_name = os.path.basename(video_path).split(".")[0]
    print(f"Extracted frames for {game_name} into {frame_dir}")

    if device == "cuda" and torch.cuda.is_available():
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif device == "mps" and torch.backends.mps.is_available():
        print("Using Apple Silicon MPS device")
    else:
        print("Using CPU")

    image_files = sorted(
        name for name in os.listdir(frame_dir)
        if os.path.isfile(os.path.join(frame_dir, name)) and name.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    num_images = len(image_files)
    print(f"Number of images in folder: {num_images}")

    if num_images == 0:
        raise RuntimeError(
            f"No frame images found in {frame_dir}. Run the extract-frames cell first or check video_path/frame_dir."
        )

    random_image_name = rng.choice(image_files)
    print(f"Selected image for region selection: {random_image_name}")

    sample_image_path = os.path.join(frame_dir, random_image_name)
    sample_frame = read_image_fn(sample_image_path)

    return {
        "frame_dir": frame_dir,
        "fps_rate": fps_rate,
        "game_name": game_name,
        "image_files": image_files,
        "sample_image_path": sample_image_path,
        "sample_frame": sample_frame,
    }


def load_pipeline(
    device: str,
    event_detection_model_folder: str,
    ball_tracking_model_dir_path: str,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
):
    from ball_tracking import BallTrackingModel
    from ball_tracking.transform import CenterCropResizeFrame
    from event_detection import get_model
    from event_detection.utils import load_model_compiled
    import torchvision.transforms as transforms

    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    model_config_path = os.path.join(event_detection_model_folder, "model_config.json")
    with open(model_config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"Model config loaded from {model_config_path}")

    train_config_path = os.path.join(event_detection_model_folder, "train_config.json")
    with open(train_config_path, "r", encoding="utf-8") as f:
        train_config = json.load(f)
    event_detection_model_selection = train_config.get("model_choice")

    event_detection_model = get_model(event_detection_model_selection, config)
    checkpoint_path = os.path.join(event_detection_model_folder, "best_model.pth")
    event_detection_model = load_model_compiled(event_detection_model, checkpoint_path, device)
    event_detection_model.eval()
    print("Event Detection Model initialized successfully")

    model_params_path = os.path.join(ball_tracking_model_dir_path, "model_params.json")
    with open(model_params_path, "r", encoding="utf-8") as f:
        model_params = json.load(f)

    image_size = model_params.get("image_size")
    num_frames = model_params.get("num_frames")
    model_choice = model_params.get("model_choice")
    model_channels = model_params.get("num_channels")
    checkpoint_path = os.path.join(ball_tracking_model_dir_path, "model_best.pth")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    ball_tracking_model_loader = BallTrackingModel(
        num_frames=num_frames,
        image_size=image_size,
        model_choice=model_choice,
        totnet_channels=model_channels,
    )
    ball_tracking_model = ball_tracking_model_loader.load_model().to(device)
    ball_tracking_model.load_state_dict(checkpoint["state_dict"])
    ball_tracking_model = ball_tracking_model.to(device)
    print("Ball Tracking Model initialized successfully")

    event_transform = transforms.Compose([
        transforms.Resize((224, 398)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.Normalize(mean=mean, std=std),
    ])
    ball_transform = transforms.Compose([
        CenterCropResizeFrame(size=(512, 512), crop_ratio=(1, 1)),
        transforms.Normalize(mean=mean, std=std),
    ])

    return (
        event_detection_model,
        ball_tracking_model,
        event_transform,
        ball_transform,
        mean,
        std,
    )


def detect_scoreboard_changes(
    frame_dir: str,
    fps_rate: float,
    device: str,
    existing_close_coord,
    existing_far_coord,
    model_path: str = "best_score_classifier.pt",
    conf_threshold: float = 0.9,
    stride: int = 5,
):
    from scoreboard_detector import ResNetScoreboardChangeDetector

    scoreboard_detector = ResNetScoreboardChangeDetector(
        frames_folder=frame_dir,
        video_fps=fps_rate,
        model_path=model_path,
        device=device,
        existing_close_coord=existing_close_coord,
        existing_far_coord=existing_far_coord,
    )
    changes, meta = scoreboard_detector.detect_changes(conf_threshold=conf_threshold, stride=stride)

    if not changes:
        raise RuntimeError("No scoreboard changes detected.")

    final_score_close = changes[-1]["new"]["close"]["score"]
    final_score_far = changes[-1]["new"]["far"]["score"]
    print(f"In total {len(changes)} scoreboard changes detected at frames:")
    print(f"Final score of the game - Close Table: {final_score_close}, Far Table: {final_score_far}")

    return {
        "changes": changes,
        "meta": meta,
        "final_score_close": final_score_close,
        "final_score_far": final_score_far,
    }


def prepare_rallies(
    frame_dir: str,
    window_size: int,
    stride: int,
    event_transform,
    ball_transform,
    event_detection_model,
    device: str,
    fps_rate: float,
    class_conversion: Dict[int, str],
    scoreboard_changes,
    threshold: float = 0.01,
    event_windows: Optional[Dict[str, int]] = None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 4,
):
    from input_process import FrameClipDataset

    dataset = FrameClipDataset(
        frame_dir,
        window_size=window_size,
        stride=stride,
        event_transform=event_transform,
        ball_transform=ball_transform,
    )
    video_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    print(f"Video Loader initialized with {len(dataset)} clips")

    pred_events = generate_event_predictions(
        video_loader=video_loader,
        event_detection_model=event_detection_model,
        device=device,
        fps_rate=fps_rate,
        class_conversion=class_conversion,
        threshold=threshold,
        event_windows=event_windows,
    )
    grouped_rallies = filter_based_on_score_changes(pred_events, scoreboard_changes, fps_rate)
    print(f"Prepared {len(grouped_rallies)} rallies in memory.")

    return {
        "dataset": dataset,
        "video_loader": video_loader,
        "pred_events": pred_events,
        "grouped_rallies": grouped_rallies,
    }


def generate_event_predictions(
    video_loader,
    event_detection_model,
    device: str,
    fps_rate: float,
    class_conversion: Dict[int, str],
    threshold: float = 0.01,
    event_windows: Optional[Dict[str, int]] = None,
    nms_on_dict_fn=None,
):
    pred_events = {}

    for data in tqdm(video_loader, desc="Processing clips"):
        clips = data['frames']
        start_idx = data['start_idx']
        clips = clips.to(device, dtype=torch.float32)

        pred_results, pred_scores = event_detection_model.predict(clips, device=device)
        pred_results = pred_results[0]
        pred_scores = pred_scores[0]

        for i, (_, pred_score_classes) in enumerate(zip(pred_results, pred_scores)):
            current_id = (i + start_idx).item()
            current_time = current_id / fps_rate
            current_time_in_mins = f"{int(current_time // 60):02d}:{int(current_time % 60):02d}"

            filtered_scores = pred_score_classes * (pred_score_classes >= threshold)
            filtered_scores[0] = -1
            best_class = filtered_scores.argmax()
            best_score = filtered_scores[best_class]

            if best_score == 0:
                continue

            pred_events[current_id] = {
                'time': current_time,
                'time_in_mins': current_time_in_mins,
                'event_type': class_conversion.get(best_class.item(), 'unknown'),
                'score': float(best_score),
            }

    if event_windows is None:
        event_windows = {'close_table_serve': 150, 'far_table_serve': 150}

    if nms_on_dict_fn is None:
        from event_detection.utils import nms_on_dict as nms_on_dict_fn

    pred_events = nms_on_dict_fn(pred_events, event_windows=event_windows)

    return pred_events


def add_ball_tracking_to_rallies(grouped_rallies, dataset, ball_tracking_model, table_detector, device: str):
    for rally in tqdm(grouped_rallies, desc="Ball tracking"):
        events = rally.get('events', [])
        for event in events:
            if not isinstance(event, dict):
                continue

            event_type = event.get('event_type')
            frame_idx = event.get('frame_index')

            if event_type not in ['far_table_bounce', 'close_table_bounce'] or frame_idx is None:
                continue

            ball_location_frames = dataset.get_surrounding_frames(frame_idx, radius=2, bidirectional=True)
            ball_location_frames = ball_location_frames.to(device, dtype=torch.float32).unsqueeze(0)
            extracted_coord, confidence = ball_tracking_model.predict(ball_location_frames)
            x_pred, y_pred = extracted_coord[0]

            x_pred = x_pred.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            event['ball_location'] = {
                'x': float(x_pred),
                'y': float(y_pred),
            }

            mapped_x, mapped_y = table_detector.transform_ball(x_pred, y_pred, blend_band_px=20)
            event['mapped_ball_location'] = {
                'x': float(mapped_x),
                'y': float(mapped_y),
            }
            event['draw_ball_location'] = {
                'x': float(mapped_x),
                'y': float(mapped_y),
            }
            event['ball_coord_confidence'] = float(confidence)

    return grouped_rallies