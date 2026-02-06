from .model import E2EModel
from .astrm import ASTRME2EModel
from .vit import VITModel
from .utils import load_model_compiled as load_event_detection_model
import json


def create_model(model_type, model_config_path, device, model_checkpoint_path):
    with open(model_config_path, 'r') as f:
        model_config_path = json.load(f)
    if model_type == 'e2e':
        model = E2EModel(model_config_path)
    elif model_type == 'astrm':
        model = ASTRME2EModel(model_config_path)
    elif model_type == 'vit':
        model = VITModel(model_config_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    load_event_detection_model(model, model_checkpoint_path, device)
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