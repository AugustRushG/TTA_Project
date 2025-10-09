import json


class analysis_utils:
    @staticmethod
    def load_json(file_path):
        """
        Load a JSON file and return its content.
        
        Args:
            file_path (str): Path to the JSON file.
        
        Returns:
            dict: Parsed JSON content.
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def save_json(data, file_path):
        """
        Save data to a JSON file.
        
        Args:
            data (dict): Data to save.
            file_path (str): Path where the JSON file will be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def count_bounces(input_data):
        """
        Count the number of bounces in the input JSON data.
        
        Args:
            input_json (str): Path to the input JSON file containing event data.
            
        Returns:
            int: Total number of bounces detected in the video.
        """
        
        close_table_bounces = 0
        far_table_bounces = 0
        for frame_idx, frame_info in input_data.items():
            if 'event_type' not in frame_info:
                continue
            event_type = frame_info['event_type']
            if event_type == 'close_table_bounce':
                close_table_bounces += 1
            elif event_type == 'far_table_bounce':
                far_table_bounces += 1
        print(f"Total close table bounces: {close_table_bounces}")
        print(f"Total far table bounces: {far_table_bounces}")
        
        return close_table_bounces + far_table_bounces, close_table_bounces, far_table_bounces
    
    def count_serves(input_data):
        """
        Count the number of serves in the input JSON data.
        
        Args:
            input_json (str): Path to the input JSON file containing event data.
            
        Returns:
            int: Total number of serves detected in the video.
        """
        
        far_table_serve = 0
        close_table_serve = 0
        for frame_idx, frame_info in input_data.items():
            if 'event_type' not in frame_info:
                continue
            event_type = frame_info['event_type']
            if event_type == 'far_table_serve':
                far_table_serve += 1
            elif event_type == 'close_table_serve':
                close_table_serve += 1
        print(f"Total far table serves: {far_table_serve}")
        print(f"Total close table serves: {close_table_serve}")
        
        return far_table_serve + close_table_serve, far_table_serve, close_table_serve

    def calculate_points(input_data,
                        serve_types=('close_table_serve', 'far_table_serve'),
                        bounce_suffix='_bounce',
                        serve_debounce=6,      # frames to suppress duplicate serves
                        skip_first_serve=True, # <- NEW
                        max_bounce_age=300     # <- NEW: max frames back a bounce can be paired (e.g., 10s @30fps)
                        ):
        """
        Calculate points from an event dict indexed by frame_id (or similar).
        Guards against duplicate serves; consumes each bounce at most once.
        Skips the very first non-debounced serve if skip_first_serve=True.
        Returns a new dict with points annotations preserved on the serve event.
        """

        # --- 0) Normalize to list of (key, item) and sort chronologically ---
        items = list(input_data.items())

        def sort_key(kv):
            k, v = kv
            if isinstance(v, dict):
                if 'frame' in v: return v['frame']
                if 'frame_id' in v: return v['frame_id']
                if 'time' in v: return v['time']
                if 'time_in_mins' in v: return v['time_in_mins']
            try:
                return int(k)
            except Exception:
                return k

        items.sort(key=sort_key)

        # Build an editable copy
        out = {k: dict(v) for k, v in items}

        close_table_points = 0
        far_table_points = 0

        used_bounce_idx = set()
        last_serve_idx = None
        serve_seen = 0  # <- counts non-debounced serves seen

        # Pre-index
        events = []
        for idx, (k, v) in enumerate(items):
            et = v.get('event_type')
            frame_like = v.get('frame', v.get('frame_id', None))
            events.append({
                'idx': idx,
                'key': k,
                'event_type': et,
                'frame_like': frame_like,
                'is_bounce': isinstance(et, str) and et.endswith(bounce_suffix),
                'is_serve': et in serve_types
            })

        for i, e in enumerate(events):
            if not e['is_serve']:
                continue

            # Debounce duplicate serves close in time
            if last_serve_idx is not None and (e['idx'] - last_serve_idx) <= serve_debounce:
                continue
            last_serve_idx = e['idx']

            # Skip the very first non-debounced serve
            if skip_first_serve and serve_seen == 0:
                serve_seen += 1
                continue
            serve_seen += 1

            # Find nearest previous *unused* bounce (and not too old)
            j = i - 1
            prev_bounce_idx = None
            prev_bounce_type = None

            while j >= 0:
                if events[j]['is_bounce'] and j not in used_bounce_idx:
                    # optional age guard (if you have frame numbers)
                    if (e['frame_like'] is not None) and (events[j]['frame_like'] is not None):
                        if (e['frame_like'] - events[j]['frame_like']) > max_bounce_age:
                            break  # too old; abort search
                    prev_bounce_idx = j
                    prev_bounce_type = events[j]['event_type']
                    break
                j -= 1

            if prev_bounce_idx is None:
                continue  # no suitable bounce found; abstain

            # Award exactly once for this bounce, then consume it
            if prev_bounce_type == 'close_table_bounce':
                far_table_points += 1
                out[events[i]['key']]['points'] = {'far': far_table_points, 'close': close_table_points}
                used_bounce_idx.add(prev_bounce_idx)

            elif prev_bounce_type == 'far_table_bounce':
                close_table_points += 1
                out[events[i]['key']]['points'] = {'far': far_table_points, 'close': close_table_points}
                used_bounce_idx.add(prev_bounce_idx)

            # else: other bounce type → ignore

        print(f"Total close table points: {close_table_points}")
        print(f"Total far table points: {far_table_points}")
        return out