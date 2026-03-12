import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os


def iter_bounce_locations(events, loc_key):
    for event in events:
        if not isinstance(event, dict):
            continue

        candidates = [event]
        bounces = event.get("bounces", [])
        if isinstance(bounces, list):
            for bounce in bounces:
                if isinstance(bounce, dict):
                    candidates.append(bounce)

        for item in candidates:
            loc = item.get(loc_key)
            if not isinstance(loc, dict):
                continue
            if "x" not in loc or "y" not in loc:
                continue

            frame_id = item.get("frame_index", item.get("frame_id", None))
            yield frame_id, float(loc["x"]), float(loc["y"])

def safe_show(fig):
    if os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


def draw_bounces_on_split_table(grouped_rallies, table_size=(1525, 2740), save_path=None,
                                loc_key="mapped_ball_location",  # or "draw_ball_location" / "draw_ball_location_split"
                                mirror_far_x=True,
                                split_size=(153, 137)):
    """
    Draw bounces on a vertically split table view (far/close halves), using the original split
    coordinate convention.

    Args:
        grouped_rallies (list): each item: {"rally_id": ..., "events": [event, ...]}
            event should contain:
              - event["frame_index"] (or frame_id)
              - event[loc_key] = {"x": float, "y": float} in FULL-table coords (origin top-left)
        table_size (tuple): (W, H) of full table in pixels
        save_path (str): optional path to save
        loc_key (str): which key to read location from ("mapped_ball_location" or "draw_ball_location")
        mirror_far_x (bool): mirror X for far-half so it visually matches the close-half orientation
    """
    W, H = table_size
    half_H = H / 2.0
    split_w, split_l = split_size

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

    # setup axes
    titles = ["Far Table Half", "Close Table Half"]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, W)
        ax.set_ylim(0, half_H)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # outline
        ax.add_patch(patches.Rectangle((0, 0), W, half_H, fill=False, linewidth=2, edgecolor='black'))
        # reference lines
        ax.axvline(x=W / 2.0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=half_H / 2.0, color='gray', linestyle='--', linewidth=1)

    # color per rally
    n_rallies = len(grouped_rallies)
    cmap = plt.get_cmap("tab20", max(n_rallies, 1))

    for rally_idx, rally in enumerate(grouped_rallies):
        rally_color = cmap(rally_idx)
        events = rally.get("events", [])

        for event in events:
            if not isinstance(event, dict):
                continue

            candidates = [event]
            bounces = event.get("bounces", [])
            if isinstance(bounces, list):
                for bounce in bounces:
                    if isinstance(bounce, dict):
                        candidates.append(bounce)

            for item in candidates:
                loc = item.get(loc_key)
                if not isinstance(loc, dict):
                    continue
                if "x" not in loc or "y" not in loc:
                    continue

                frame_id = item.get("frame_index", item.get("frame_id", None))

                if loc_key.endswith("_split") and isinstance(loc.get("table_half"), str):
                    side = 0 if loc.get("table_half") == "far" else 1
                    x_half = float(loc["x"])
                    y_half = float(loc["y"])
                else:
                    # Convert FULL table coords (top-left origin) -> original split convention
                    x = float(loc["x"])
                    y = float(loc["y"])

                    if y < half_H:
                        side = 0  # far
                        x_half = (W - x) if mirror_far_x else x
                        y_half = y
                    else:
                        side = 1  # close
                        x_half = x
                        y_half = H - y

                ax = axes[side]
                ax.plot(x_half, y_half, marker='o', linestyle='', color=rally_color, markersize=5)
                if frame_id is not None:
                    ax.text(x_half + 0.5, y_half, f"{frame_id}", fontsize=6, color=rally_color)

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    safe_show(fig)




def draw_bounces_on_table(grouped_rallies, table_size=(1525, 2740), save_path=None):

    W, H = table_size

    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')
    ax.set_title("Bounce Events on Table")

    # Draw table outline
    table_rect = patches.Rectangle((0, 0), W, H,
                                   fill=False, linewidth=2, edgecolor='black')
    ax.add_patch(table_rect)

    # Middle reference lines
    ax.axvline(x=W // 2, color='gray', linestyle='--', linewidth=1)
    ax.axhline(y=H // 2, color='gray', linestyle='--', linewidth=1)

    # Generate distinct colors for rallies
    num_rallies = len(grouped_rallies)
    cmap = plt.get_cmap('tab20', num_rallies)  # good categorical colormap

    for rally_idx, rally in enumerate(grouped_rallies):

        rally_color = cmap(rally_idx)

        events = rally['events']

        for frame_id, x, y in iter_bounce_locations(events, "mapped_ball_location"):

            ax.plot(x, y,
                    marker='o',
                    linestyle='',
                    color=rally_color,
                    markersize=5)

            ax.text(x + 5, y,
                    f"{frame_id}",
                    fontsize=6,
                    color=rally_color)

    if save_path:
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)

    safe_show(fig)