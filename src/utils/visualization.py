import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_bounces_on_split_table(bounces, table_size=(1525, 2740), save_path=None):
    """
    Draw bounces on a vertically split table view (bottom and top halves), both with origin at bottom-left.

    Args:
        bounces (dict): 
            {frame_id: {"event_type": str, "mapped_ball_location": {"x":, "y":}}}
        table_size (tuple): (width, height) of the full table in pixels
        save_path (str): optional path to save figure
    """
    W, H = table_size
    half_H = H // 2

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))  # horizontal layout

    # Define colors per event type
    color_map = {
        'far_table_bounce': 'blue',
        'close_table_bounce': 'red',
        'net_bounce': 'green',
        'unknown': 'gray'
    }

    for idx, ax in enumerate(axes):
        ax.set_xlim(0, W)
        ax.set_ylim(0, half_H)
        ax.set_aspect('equal')
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        if idx == 0:
            ax.set_title("Far Table Half (0,0 at bottom-left)")
        else:
            ax.set_title("Close Table Half (0,0 at bottom-left)")

        # Draw table half outline
        table_rect = patches.Rectangle((0, 0), W, half_H, 
                                       fill=False, linewidth=2, edgecolor='black')
        ax.add_patch(table_rect)

        vertical_middle_line = W // 2 
        ax.axvline(x=vertical_middle_line, color='gray', linestyle='--', linewidth=1)
        horizontal_middle_line = half_H // 2
        ax.axhline(y=horizontal_middle_line, color='gray', linestyle='--', linewidth=1)

    for frame_id, info in bounces.items():
        if "draw_ball_location" not in info:
            continue

        x = info["draw_ball_location"]["x"]
        y = info["draw_ball_location"]["y"]
        event_type = info.get("event_type", "unknown")
        color = color_map.get(event_type, 'black')

        y_plot = H - y  # always flip Y to match bottom-left origin

        if y < half_H: # this means far table view
            side = 0
            x_plot = W - x  # mirror X to match far table view
            y_plot = y_plot - half_H  # adjust Y for far table view
            y_plot = half_H - y_plot  # flip Y for far table view
        else: # this means close table view
            side = 1
            x_plot = x

        ax = axes[side]
        ax.plot(x_plot, y_plot, marker='o', linestyle='', color=color, markersize=5)
        ax.text(x_plot + 5, y_plot, f"{frame_id}", fontsize=6, color=color)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()



def draw_bounces_on_table(bounces, table_size=(1525, 2740), save_path=None):
    """
    Draw bounces on a table view.

    Args:
        bounces (dict): 
            {frame_id: {"event_type": str, "mapped_ball_location": {"x":, "y":}}}
        table_size (tuple): (width, height) of the table in pixels
        save_path (str): optional path to save figure
    """
    W, H = table_size

    fig, ax = plt.subplots(figsize=(8, 12))  # bigger figure for clarity
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)  # origin at top-left
    ax.set_aspect('equal')
    ax.set_title("Bounce Events on Table")

    # Draw table outline
    table_rect = patches.Rectangle((0, 0), W, H, 
                                   fill=False, linewidth=2, edgecolor='black')
    ax.add_patch(table_rect)

    # Define colors per event type
    color_map = {
        'far_table_bounce': 'blue',
        'close_table_bounce': 'red',
        'net_bounce': 'green',
        'unknown': 'gray'
    }

    # Draw bounces
    for frame_id, info in bounces.items():
        if "mapped_ball_location" not in info:
            continue

        x = info["mapped_ball_location"]["x"]
        y = info["mapped_ball_location"]["y"]
        event_type = info.get("event_type", "unknown")

        color = color_map.get(event_type, 'black')  # fallback color

        ax.plot(x, y, marker='o', linestyle='', color=color, markersize=5)
        ax.text(x + 5, y, f"{frame_id}", fontsize=6, color=color)

    # Optional save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    plt.show()