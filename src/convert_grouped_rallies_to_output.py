import argparse
import json
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert grouped_rallies JSON into output.json structure (file/ALL_INSTANCES/instance)."
    )
    parser.add_argument("--input", required=True, help="Path to grouped_rallies JSON file")
    parser.add_argument("--output", default="output_from_grouped.json", help="Path to output JSON file")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS used for frame-to-time fallback")
    parser.add_argument("--game", default="Game 1", help="Game label text")
    parser.add_argument("--close-player", default="Close Player", help="Display name for close side player")
    parser.add_argument("--far-player", default="Far Player", help="Display name for far side player")
    parser.add_argument(
        "--include-bounces",
        action="store_true",
        help="Include nested bounce entries as separate instances",
    )
    return parser.parse_args()


def side_from_event_type(event_type: str) -> str:
    if event_type.startswith("close_table"):
        return "close"
    if event_type.startswith("far_table"):
        return "far"
    return ""


def shot_from_event_type(event_type: str) -> str:
    if event_type.endswith("serve"):
        return "Serve"
    if event_type.endswith("forehand"):
        return "Forehand"
    if event_type.endswith("backhand"):
        return "Backhand"
    if event_type.endswith("bounce"):
        return "Bounce"
    return event_type


def point_winner(old_score: Dict[str, Any], new_score: Dict[str, Any]) -> str:
    old_close = old_score.get("close")
    old_far = old_score.get("far")
    new_close = new_score.get("close")
    new_far = new_score.get("far")

    if isinstance(old_close, (int, float)) and isinstance(new_close, (int, float)) and new_close > old_close:
        return "close"
    if isinstance(old_far, (int, float)) and isinstance(new_far, (int, float)) and new_far > old_far:
        return "far"
    return ""


def normalize_time(event: Dict[str, Any], fps: float) -> float:
    if isinstance(event.get("time"), (int, float)):
        return float(event["time"])
    frame_idx = event.get("frame_index")
    if isinstance(frame_idx, (int, float)) and fps > 0:
        return float(frame_idx) / fps
    return 0.0


def infer_end_time(
    current_time: float,
    next_time: Optional[float],
    score_change_frame: Optional[int],
    fps: float,
) -> float:
    if next_time is not None and next_time > current_time:
        return next_time
    if isinstance(score_change_frame, int) and fps > 0:
        score_time = score_change_frame / fps
        if score_time > current_time:
            return score_time
    return current_time + (1.0 / fps if fps > 0 else 0.033)


def build_labels(
    rally: Dict[str, Any],
    point_number: int,
    shot_number: int,
    event_type: str,
    frame_index: Any,
    side_name: str,
    winner_name: str,
    game: str,
    close_player: str,
    far_player: str,
    old_score: Dict[str, Any],
    score_change_frame: Any,
    confidence: Any,
    location: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    shot_name = shot_from_event_type(event_type)
    winner_text = f"{winner_name} Win" if winner_name else "NA"

    error_text = "NA"
    if winner_name and side_name and winner_name != side_name:
        error_text = side_name

    serve_type_text = shot_name if event_type.endswith("serve") else "NA"

    x_text = "NA"
    y_text = "NA"
    if location:
        x = location.get("x")
        y = location.get("y")
        if x is not None:
            x_text = str(x)
        if y is not None:
            y_text = str(y)

    labels: List[Dict[str, str]] = [
        {"group": f'{close_player} Score', "text": str(old_score.get("close", "NA"))},
        {"group": f'{far_player} Score', "text": str(old_score.get("far", "NA"))},
        {"group": "Game", "text": game},
        {"group": "Point Number", "text": str(point_number)},
        {"group": "Hand", "text": "NA"},
        {"group": "Serve Type", "text": serve_type_text},
        {"group": "Rally Shot Number", "text": str(shot_number)},
        {"group": "x2", "text": x_text},
        {"group": "y2", "text": y_text},
        {"group": "Error Type", "text": "NA"},
        {"group": "Error", "text": error_text},
        {"group": "Error Area", "text": "NA"},
        {"group": "Rally Shot", "text": shot_name},
        {"group": "Rally Stroke", "text": shot_name},
        {"group": "Point Won", "text": winner_text},
    ]

    return labels


def event_to_instance(
    event: Dict[str, Any],
    rally: Dict[str, Any],
    point_number: int,
    shot_number: int,
    next_time: Optional[float],
    fps: float,
    game: str,
    close_player: str,
    far_player: str,
    instance_id: int,
) -> Dict[str, Any]:
    event_type = str(event.get("event_type", "unknown"))
    side = side_from_event_type(event_type)
    side_name = close_player if side == "close" else far_player if side == "far" else ""

    old_score = rally.get("old_score", {}) or {}
    new_score = rally.get("new_score", {}) or {}
    winner_side = point_winner(old_score, new_score)
    winner_name = close_player if winner_side == "close" else far_player if winner_side == "far" else ""

    start = normalize_time(event, fps)
    end = infer_end_time(start, next_time, rally.get("score_change_frame"), fps)

    shot_name = shot_from_event_type(event_type)
    code = f"{side_name} {shot_name}".strip() if side_name else shot_name

    labels = build_labels(
        rally=rally,
        point_number=point_number,
        shot_number=shot_number,
        event_type=event_type,
        frame_index=event.get("frame_index", ""),
        side_name=side_name,
        winner_name=winner_name,
        game=game,
        close_player=close_player,
        far_player=far_player,
        old_score=old_score,
        score_change_frame=rally.get("score_change_frame"),
        confidence=event.get("score"),
        location=event.get("ball_location") if isinstance(event.get("ball_location"), dict) else None,
    )

    return {
        "ID": str(instance_id),
        "start": f"{start:.2f}",
        "end": f"{end:.2f}",
        "code": code,
        "label": labels,
    }


def convert_grouped_rallies(
    grouped_rallies: List[Dict[str, Any]],
    fps: float,
    game: str,
    close_player: str,
    far_player: str,
    include_bounces: bool,
) -> Dict[str, Any]:
    instances: List[Dict[str, Any]] = []
    instance_id = 1

    for rally_idx, rally in enumerate(grouped_rallies, start=1):
        point_number = int(rally.get("rally_id", rally_idx))
        events = rally.get("events", []) or []

        shot_number = 1
        for event_idx, event in enumerate(events):
            if not isinstance(event, dict):
                continue

            next_time = None
            if event_idx + 1 < len(events):
                next_event = events[event_idx + 1]
                if isinstance(next_event, dict):
                    next_time = normalize_time(next_event, fps)

            instances.append(
                event_to_instance(
                    event=event,
                    rally=rally,
                    point_number=point_number,
                    shot_number=shot_number,
                    next_time=next_time,
                    fps=fps,
                    game=game,
                    close_player=close_player,
                    far_player=far_player,
                    instance_id=instance_id,
                )
            )
            instance_id += 1
            shot_number += 1

            if not include_bounces:
                continue

            bounces = event.get("bounces", []) or []
            for bounce in bounces:
                if not isinstance(bounce, dict) or not bounce.get("event_type"):
                    continue

                bounce_time = normalize_time(bounce, fps)
                bounce_end = bounce_time + (1.0 / fps if fps > 0 else 0.033)

                # Reuse event mapping logic with a tiny override for timing.
                bounce_instance = event_to_instance(
                    event=bounce,
                    rally=rally,
                    point_number=point_number,
                    shot_number=shot_number,
                    next_time=None,
                    fps=fps,
                    game=game,
                    close_player=close_player,
                    far_player=far_player,
                    instance_id=instance_id,
                )
                bounce_instance["start"] = f"{bounce_time:.2f}"
                bounce_instance["end"] = f"{bounce_end:.2f}"

                instances.append(bounce_instance)
                instance_id += 1
                shot_number += 1

    return {"file": {"ALL_INSTANCES": {"instance": instances}}}


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        grouped_rallies = json.load(f)

    output_data = convert_grouped_rallies(
        grouped_rallies=grouped_rallies,
        fps=args.fps,
        game=args.game,
        close_player=args.close_player,
        far_player=args.far_player,
        include_bounces=args.include_bounces,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    print(f"Converted {args.input} -> {args.output}")
    print(f"Total instances: {len(output_data['file']['ALL_INSTANCES']['instance'])}")


if __name__ == "__main__":
    main()
