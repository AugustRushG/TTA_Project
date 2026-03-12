import json
import csv


# =========================================================
# USER CONFIG
# =========================================================
JSON_PATH = "final_rally_summary_26WPF_AUS_W10_G_Yang_Qian_AUS v_Tian_TPE_game2.json"
CSV_PATH = "output_rally_events.csv"

TIMELINE_VALUE = "Game 2"
GAME_VALUE = "26WPF_AUS_W10"

PLAYER_CLOSE_NAME = "yang"
PLAYER_FAR_NAME = "tian"


# =========================================================
# DYNAMIC COLUMN NAMES
# =========================================================
CLOSE_SCORE_COL = f"{PLAYER_CLOSE_NAME}.score"
FAR_SCORE_COL = f"{PLAYER_FAR_NAME}.score"


# =========================================================
# EXACT OUTPUT COLUMNS
# =========================================================
COLUMNS = [
    "id",
    "timeline",
    "start.time",
    "duration",
    "game",
    "point.number",
    CLOSE_SCORE_COL,
    FAR_SCORE_COL,
    "row",
    "rally.shot.number",
    "rally.shot",
    "rally.stroke",
    "rally.success",
    "x1",
    "y1",
    "x2",
    "y2",
    "shot.origin",
    "shot.location",
    "serve",
    "return",
    "serve.type",
    "serve.from",
    "serve.outcome",
    "first.attack",
    "key.shot",
    "key.shot.type",
    "winner",
    "error",
    "error.area",
    "error.type",
    "match.event",
    "hand",
    "handedness",
    "point.won",
    "point.lost",
    "w.l",
]


# =========================================================
# HELPERS
# =========================================================
def map_scores(score_dict):
    return score_dict.get("close", ""), score_dict.get("far", "")


def infer_side(event_type):
    et = (event_type or "").lower()
    if "close" in et:
        return "close"
    if "far" in et:
        return "far"
    return ""


def infer_player_name(side):
    if side == "close":
        return PLAYER_CLOSE_NAME
    if side == "far":
        return PLAYER_FAR_NAME
    return ""


def infer_stroke(event_type):
    et = (event_type or "").lower()

    if "serve" in et:
        return "serve"
    if "forehand" in et:
        return "forehand"
    if "backhand" in et:
        return "backhand"
    if "push" in et:
        return "push"
    if "loop" in et:
        return "loop"
    if "flick" in et:
        return "flick"
    if "smash" in et:
        return "smash"
    if "block" in et:
        return "block"
    if "chop" in et:
        return "chop"
    if "lob" in et:
        return "lob"

    return event_type


def build_row_label(event_type):
    side = infer_side(event_type)
    player_name = infer_player_name(side)
    stroke = infer_stroke(event_type)

    if player_name and stroke:
        return f"{player_name} {stroke}"
    if player_name:
        return player_name
    return event_type or ""


def infer_serve(event_type):
    if event_type == 'close_table_serve':
        return PLAYER_CLOSE_NAME, PLAYER_FAR_NAME
    if event_type == 'far_table_serve':
        return PLAYER_FAR_NAME, PLAYER_CLOSE_NAME
    return 'NA', 'NA'



def infer_winner_and_scores(old_score, new_score):
    old_close, old_far = map_scores(old_score)
    new_close, new_far = map_scores(new_score)

    result = {
        CLOSE_SCORE_COL: old_close if old_close != "" else "",
        FAR_SCORE_COL: old_far if old_far != "" else "",
        "winner": "",
        "point.won": "",
        "point.lost": "",
        "w.l": "",
    }

    try:
        if old_close != "" and new_close != "" and new_close > old_close:
            result["winner"] = PLAYER_CLOSE_NAME
            result["point.won"] = PLAYER_CLOSE_NAME
            result["point.lost"] = PLAYER_FAR_NAME
            result["w.l"] = "W"
        elif old_far != "" and new_far != "" and new_far > old_far:
            result["winner"] = PLAYER_FAR_NAME
            result["point.won"] = PLAYER_FAR_NAME
            result["point.lost"] = PLAYER_CLOSE_NAME
            result["w.l"] = "L"
    except TypeError:
        pass

    return result


def build_row(event_id, point_number, shot_number, rally, event):
    row = {col: "" for col in COLUMNS}

    event_type = event.get("event_type", "")
    side = infer_side(event_type)
    player_name = infer_player_name(side)
    stroke = infer_stroke(event_type)

    old_score = rally.get("old_score", {})
    new_score = rally.get("new_score", {})
    score_info = infer_winner_and_scores(old_score, new_score)

    ball_location = event.get("ball_location", {}) or {}
    x = ball_location.get("x", "")
    y = ball_location.get("y", "")

    row["id"] = event_id
    row["timeline"] = TIMELINE_VALUE
    row["start.time"] = event.get("time", "")
    row["duration"] = 1
    row["game"] = GAME_VALUE
    row["point.number"] = point_number

    row[CLOSE_SCORE_COL] = score_info[CLOSE_SCORE_COL]
    row[FAR_SCORE_COL] = score_info[FAR_SCORE_COL]

    # row is player + shot label
    row["row"] = build_row_label(event_type)

    row["rally.shot.number"] = shot_number
    row["rally.shot"] = event_type
    row["rally.stroke"] = stroke
    row["rally.success"] = 1 if event.get("score", "") != "" else ""

    row["x1"] = ""
    row["y1"] = ""
    row["x2"] = ""
    row["y2"] = ""

    row["shot.origin"] = player_name
    row["shot.location"] = f"{x},{y}" if x != "" and y != "" else ""

    row["serve"], row["return"] = infer_serve(event_type)
    row["serve.type"] = stroke if row["serve"] == 1 else ""
    row["serve.from"] = player_name if row["serve"] == 1 else ""
    row["serve.outcome"] = ""

    row["first.attack"] = ""
    row["key.shot"] = ""
    row["key.shot.type"] = ""

    row["winner"] = score_info["winner"]
    row["error"] = ""
    row["error.area"] = ""
    row["error.type"] = ""

    row["match.event"] = event_type
    row["hand"] = ""
    row["handedness"] = ""

    row["point.won"] = score_info["point.won"]
    row["point.lost"] = score_info["point.lost"]
    row["w.l"] = score_info["w.l"]

    return row


# =========================================================
# MAIN
# =========================================================
def json_to_csv(json_path, csv_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    event_id = 1

    for rally_idx, rally in enumerate(data, start=1):
        point_number = rally.get("rally_id", rally_idx)
        events = rally.get("events", [])

        for shot_number, event in enumerate(events, start=1):
            rows.append(
                build_row(
                    event_id=event_id,
                    point_number=point_number,
                    shot_number=shot_number,
                    rally=rally,
                    event=event,
                )
            )
            event_id += 1

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to: {csv_path}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    json_to_csv(JSON_PATH, CSV_PATH)