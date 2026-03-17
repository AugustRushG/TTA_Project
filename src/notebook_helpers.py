import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import RectangleSelector

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


def show_cropped_image(img, coord):
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
    plt.show()


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

    for rally in grouped_rallies:
        events = rally["events"]
        linear_events = _linearize_events(events)
        reorganized_events = []
        previous_non_bounce_event = None

        for event in linear_events:
            event_type = str(event.get("event_type", ""))

            if "bounce" in event_type:
                if previous_non_bounce_event is not None:
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
