import argparse
import json
from typing import Any, Dict, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich grouped_rallies JSON with rally frame-time fields and bounce draw coordinates "
            "for full-table and split-table views."
        )
    )
    parser.add_argument("--input", required=True, help="Path to grouped_rallies JSON file")
    parser.add_argument("--output", required=True, help="Path to write enriched JSON file")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS for frame to time conversion")
    parser.add_argument("--table-width", type=float, default=1525.0, help="Full table width in mapped coordinates")
    parser.add_argument("--table-height", type=float, default=2740.0, help="Full table height in mapped coordinates")
    parser.add_argument(
        "--mirror-far-x",
        action="store_true",
        help="Mirror x on far half for split-table coordinates (same as draw_bounces_on_split_table default)",
    )
    return parser.parse_args()


def frame_to_time(frame_idx: Any, fps: float) -> Tuple[Optional[float], Optional[str]]:
    if not isinstance(frame_idx, (int, float)) or fps <= 0:
        return None, None
    sec = float(frame_idx) / float(fps)
    mins = int(sec // 60)
    rem = int(sec % 60)
    return sec, f"{mins:02d}:{rem:02d}"


def _valid_xy(loc: Any) -> bool:
    return isinstance(loc, dict) and isinstance(loc.get("x"), (int, float)) and isinstance(loc.get("y"), (int, float))


def pick_full_draw_location(item: Dict[str, Any]) -> Optional[Dict[str, float]]:
    for key in ("draw_ball_location", "mapped_ball_location", "mapped_ball_location_of"):
        candidate = item.get(key)
        if _valid_xy(candidate):
            return {"x": float(candidate["x"]), "y": float(candidate["y"])}
    return None


def to_split_xy(full_loc: Dict[str, float], table_w: float, table_h: float, mirror_far_x: bool) -> Dict[str, Any]:
    x = float(full_loc["x"])
    y = float(full_loc["y"])

    half_h = table_h / 2.0
    y_bl_full = table_h - y

    if y < half_h:
        table_half = "far"
        y_half = y
        x_half = (table_w - x) if mirror_far_x else x
    else:
        table_half = "close"
        y_half = y_bl_full
        x_half = x

    return {
        "x": float(x_half),
        "y": float(y_half),
        "table_half": table_half,
    }


def enrich_bounce(bounce: Dict[str, Any], table_w: float, table_h: float, mirror_far_x: bool) -> bool:
    full_loc = pick_full_draw_location(bounce)
    if not full_loc:
        return False

    bounce["draw_ball_location"] = full_loc
    bounce["draw_ball_location_split"] = to_split_xy(full_loc, table_w, table_h, mirror_far_x)
    return True


def enrich_grouped_rallies(
    grouped: Any,
    fps: float,
    table_w: float,
    table_h: float,
    mirror_far_x: bool,
) -> Tuple[Any, int]:
    if not isinstance(grouped, list):
        raise ValueError("Input JSON must be a list of rallies.")

    enriched_bounces = 0

    for rally in grouped:
        if not isinstance(rally, dict):
            continue

        score_change_frame = rally.get("score_change_frame")
        serve_event_frame = rally.get("serve_event_frame")

        score_sec, score_mmss = frame_to_time(score_change_frame, fps)
        serve_sec, serve_mmss = frame_to_time(serve_event_frame, fps)

        if score_sec is not None:
            rally["score_change_time"] = score_sec
            rally["score_change_time_in_mins"] = score_mmss

        if serve_sec is not None:
            rally["serve_event_time"] = serve_sec
            rally["serve_event_time_in_mins"] = serve_mmss

        events = rally.get("events", [])
        if not isinstance(events, list):
            continue

        for event in events:
            if not isinstance(event, dict):
                continue

            bounces = event.get("bounces", [])
            if isinstance(bounces, list):
                for bounce in bounces:
                    if isinstance(bounce, dict) and bounce.get("event_type", "").endswith("_bounce"):
                        if enrich_bounce(bounce, table_w, table_h, mirror_far_x):
                            enriched_bounces += 1

            if event.get("event_type", "").endswith("_bounce"):
                if enrich_bounce(event, table_w, table_h, mirror_far_x):
                    enriched_bounces += 1

    return grouped, enriched_bounces


def main() -> None:
    args = parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        grouped = json.load(f)

    enriched, bounce_count = enrich_grouped_rallies(
        grouped=grouped,
        fps=args.fps,
        table_w=args.table_width,
        table_h=args.table_height,
        mirror_far_x=args.mirror_far_x,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=4, ensure_ascii=False)

    print(f"Enriched file written: {args.output}")
    print(f"Bounces with draw coordinates updated: {bounce_count}")


if __name__ == "__main__":
    main()
