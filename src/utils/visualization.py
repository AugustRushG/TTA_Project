import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_bounces_on_split_table(bounces, table_size=(1525, 2740), save_path=None):
    """
    Draw bounces on a split table view (two halves).

    Args:
        bounces (dict): 
            {frame_id: {"event_type": str, "mapped_ball_location": {"x":, "y":}}}
        table_size (tuple): (width, height) of the table in pixels
        save_path (str): optional path to save figure
    """
    W, H = table_size
    half_W = W // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Define colors per event type
    color_map = {
        'far_table_bounce': 'blue',
        'close_table_bounce': 'red',
        'net_bounce': 'green',
        'unknown': 'gray'
    }

    for idx, ax in enumerate(axes):
        ax.set_xlim(0, half_W)
        ax.set_ylim(0, H)
        ax.set_aspect('equal')
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.set_title(f"Table Side {idx+1}")

        # Draw half-table outline
        table_rect = patches.Rectangle((0, 0), half_W, H, 
                                       fill=False, linewidth=2, edgecolor='black')
        ax.add_patch(table_rect)

    # Draw bounces
    for frame_id, info in bounces.items():
        if "mapped_ball_location" not in info:
            continue

        x = info["mapped_ball_location"]["x"]
        y = info["mapped_ball_location"]["y"]
        event_type = info.get("event_type", "unknown")

        color = color_map.get(event_type, 'black')  # fallback color

        if x < half_W:
            side = 0
            x_plot = x
        else:
            side = 1
            x_plot = x - half_W

        ax = axes[side]
        ax.plot(x_plot, y, marker='o', linestyle='', color=color, markersize=5)
        ax.text(x_plot + 5, y, f"{frame_id}", fontsize=6, color=color)

    # Optional save
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