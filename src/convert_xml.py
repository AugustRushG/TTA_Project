import json
import csv
import os


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
JSON_PATH = "/home/august/github/TTA_Project/src/final_rally_summary_26WPF_AUS_M11_G_von_Einem_AUS_v_Yuen_King_Shing_HKG_game1.json"
CSV_PATH = "final_rally_summary_custom_columns.csv"

# change these if close/far should map differently
CLOSE_PLAYER = "yang"        # score shown as old_score/new_score["close"]
FAR_PLAYER = "alexandre"     # score shown as old_score/new_score["far"]

GAME_NAME = os.path.splitext(os.path.basename(JSON_PATH))[0]


# --------------------------------------------------
# EXACT OUTPUT COLUMNS
# --------------------------------------------------
COLUMNS = [
    "timeline",
    "start.time",
    "duration",
    "game",
    "point.number",
    "yang.score",
    "alexandre.score",
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


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def get_score_mapping(score_dict):
    """
    Map JSON score dict {'close': x, 'far': y}
    to yang.score / alexandre.score based on config.
    """
    close_score = score_dict.get("close")
    far_score = score_dict.get("far")

    if CLOSE_PLAYER == "yang":
        yang_score = close_score
        alex_score = far_score
    else:
        yang_score = far_score
        alex_score = close_score

    return yang_score, alex_score


def event_to_row(rally, event, shot_idx):
    old_score = rally.get("old_score", {})
    new_score = rally.get("new_score", {})

    # usually the point result after the rally ends is more useful
    yang_score, alex_score = get_score_mapping(new_score)

    frame_index = event.get("frame_index")
    time_sec = event.get("time")
    time_min = event.get("time_in_mins")
    event_type = event.get("event_type", "")

    # ball location if available
    ball_loc = event.get("ball_location", {}) or {}
    mapped_ball_loc = event.get("mapped_ball_location", {}) or {}

    x = ball_loc.get("x", "")
    y = ball_loc.get("y", "")

    # if your XML/CSV later needs mapped coordinates instead, replace x/y with mapped values
    mapped_x = mapped_ball_loc.get("x", "")
    mapped_y = mapped_ball_loc.get("y", "")

    row = {col: "" for col in COLUMNS}

    # fill known values
    row["timeline"] = time_min if time_min is not None else ""
    row["start.time"] = time_sec if time_sec is not None else ""
    row["duration"] = ""
    row["game"] = GAME_NAME
    row["point.number"] = rally.get("rally_id", "")
    row["yang.score"] = yang_score if yang_score is not None else ""
    row["alexandre.score"] = alex_score if alex_score is not None else ""
    row["row"] = shot_idx + 1
    row["rally.shot.number"] = shot_idx + 1
    row["rally.shot"] = event_type
    row["rally.stroke"] = event_type
    row["rally.success"] = 1 if event.get("score") is not None else ""

    # bounding box columns are not in your JSON, so keep blank
    row["x1"] = ""
    row["y1"] = ""
    row["x2"] = ""
    row["y2"] = ""

    # choose one of these:
    # row["shot.location"] = f"{mapped_x},{mapped_y}" if mapped_x != "" and mapped_y != "" else ""
    row["shot.location"] = f"{x},{y}" if x != "" and y != "" else ""

    # infer some semantic columns from event_type
    lower_type = event_type.lower()

    row["serve"] = 1 if "serve" in lower_type else 0
    row["return"] = 1 if "return" in lower_type else 0
    row["match.event"] = event_type

    if "serve" in lower_type:
        row["serve.type"] = event_type
        row["serve.from"] = "far" if "far" in lower_type else ("close" if "close" in lower_type else "")
        row["shot.origin"] = row["serve.from"]
    else:
        row["shot.origin"] = "far" if "far" in lower_type else ("close" if "close" in lower_type else "")

    # winner / point result inference from score change
    old_yang, old_alex = get_score_mapping(old_score)
    new_yang, new_alex = get_score_mapping(new_score)

    if old_yang != new_yang or old_alex != new_alex:
        if new_yang is not None and old_yang is not None and new_yang > old_yang:
            row["point.won"] = "yang"
            row["point.lost"] = "alexandre"
            row["winner"] = "yang"
            row["w.l"] = "W"
        elif new_alex is not None and old_alex is not None and new_alex > old_alex:
            row["point.won"] = "alexandre"
            row["point.lost"] = "yang"
            row["winner"] = "alexandre"
            row["w.l"] = "L" if CLOSE_PLAYER == "yang" else "W"

    return row


def json_to_custom_csv(json_path, csv_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for rally in data:
        events = rally.get("events", [])

        if not events:
            empty_row = {col: "" for col in COLUMNS}
            empty_row["game"] = GAME_NAME
            empty_row["point.number"] = rally.get("rally_id", "")
            rows.append(empty_row)
            continue

        for shot_idx, event in enumerate(events):
            rows.append(event_to_row(rally, event, shot_idx))

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved to: {csv_path}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    json_to_custom_csv(JSON_PATH, CSV_PATH)