
import torch

def load_model_compiled(model, checkpoint_path, device):
    raw_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
    else:
        state_dict = raw_state
    fixed_state = {}

    # ---- Fix: Strip `_orig_mod.` prefix if present ----
    fixed_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]  # remove prefix
        else:
            new_k = k
        fixed_state[new_k] = v

    # ---- Load into your model ----
    missing, unexpected = model.load_state_dict(fixed_state, strict=False)
    print("\n=== State Dict Load Report ===")
    print("Missing keys in model:", missing)
    print("Unexpected keys in checkpoint:", unexpected)
    model.to(device)
    return model

def nms_on_dict(pred_events, event_windows=None, default_window=2):
    """
    Apply temporal NMS to a dict of predictions with event-type-dependent windows.

    Args:
        pred_events (dict): {frame_id: {'event_type': str, 'score': float}}
        event_windows (dict): {event_type: window_length}
        default_window (int): fallback window length if event_type not in event_windows.

    Returns:
        dict: filtered predictions.
    """
    # convert dict to list of (frame_id, event_type, score)
    events = [
        (fid, v['time'], v['event_type'], v['score'], v['time_in_mins'])
        for fid, v in pred_events.items()
    ]

    # sort by score descending
    events.sort(key=lambda x: x[3], reverse=True)

    selected = []
    suppressed = set()

    for frame_id, time, event_type, score, time_in_mins in events:
        if frame_id in suppressed:
            continue

        # keep this event
        selected.append((frame_id, time, event_type, score, time_in_mins))

        # get window length for this event type
        nms_window = event_windows.get(event_type, default_window) if event_windows else default_window

        # suppress neighboring frames of the same event_type
        for offset in range(-nms_window, nms_window + 1):
            suppressed.add(frame_id + offset)

    # rebuild the filtered dict
    filtered = {
        fid: {'frame_index': fid, 'time': time, 'time_in_mins': time_in_mins, 'event_type': etype, 'score': score}
        for fid, time, etype, score, time_in_mins in selected
    }

    # sort by frame_id
    filtered = dict(sorted(filtered.items()))

    return filtered