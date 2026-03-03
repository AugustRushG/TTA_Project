import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import numpy as np
import os

def safe_show(fig):
    if os.environ.get("DISPLAY"):
        plt.show()
    plt.close(fig)


def draw_bounces_on_split_table(grouped_rallies, table_size=(1525, 2740), save_path=None,
                                loc_key="mapped_ball_location",  # or "draw_ball_location"
                                mirror_far_x=True):
    """
    Draw bounces on a vertically split table view (far/close halves), both with origin at bottom-left.

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

    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

    # setup axes
    titles = ["Far Table Half (origin bottom-left)", "Close Table Half (origin bottom-left)"]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, W)
        ax.set_ylim(0, half_H)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        # outline
        ax.add_patch(patches.Rectangle((0, 0), W, half_H, fill=False, linewidth=2, edgecolor='black'))
        # reference lines
        ax.axvline(x=W / 2.0, color='gray', linestyle='--', linewidth=1)
        ax.axhline(y=half_H / 2.0, color='gray', linestyle='--', linewidth=1)

    # color per rally
    n_rallies = len(grouped_rallies)
    cmap = cm.get_cmap("tab20", max(n_rallies, 1))

    for rally_idx, rally in enumerate(grouped_rallies):
        rally_color = cmap(rally_idx)
        events = rally.get("events", [])

        for event in events:
            loc = event.get(loc_key, None)
            if not loc:
                continue

            frame_id = event.get("frame_index", event.get("frame_id", None))
            x = float(loc["x"])
            y = float(loc["y"])

            # Your mapped coords are in FULL TABLE space with origin at TOP-LEFT:
            # Convert to bottom-left full-table first:
            y_bl_full = H - y   # bottom-left full-table y

            # Decide far vs close using original y in top-left space:
            # far half is y in [0, H/2), close half is y in [H/2, H]
            if y < half_H:
                side = 0  # far
                # map to half-table bottom-left: far occupies top region in TL coords -> upper half
                # In bottom-left full coords, upper half is y in [H/2, H]
                y_half = y_bl_full - half_H  # now in [0, half_H]
                x_half = (W - x) if mirror_far_x else x
            else:
                side = 1  # close
                # bottom region in TL coords -> lower half
                y_half = y_bl_full  # already in [0, half_H]
                x_half = x

            ax = axes[side]
            ax.plot(x_half, y_half, marker='o', linestyle='', color=rally_color, markersize=5)
            if frame_id is not None:
                ax.text(x_half + 5, y_half, f"{frame_id}", fontsize=6, color=rally_color)

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
    cmap = cm.get_cmap('tab20', num_rallies)  # good categorical colormap

    for rally_idx, rally in enumerate(grouped_rallies):

        rally_color = cmap(rally_idx)

        events = rally['events']

        for event in events:
            if "mapped_ball_location" not in event:
                continue

            frame_id = event['frame_index']
            x = event["mapped_ball_location"]["x"]
            y = event["mapped_ball_location"]["y"]

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