import os
import cv2
import numpy as np
import json
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from .model import ScoreClassifier
MAX_SCORE = 15  # or whatever your max class is

def select_roi(frame, title="Select ROI", existing=None, allow_cancel=True):
    """
    frame: numpy HxWxC (BGR)
    existing: optional (x1,y1,x2,y2) to show as a hint (we can't prefill selectROI,
                but we can draw it before selecting)
    returns: (x1,y1,x2,y2) or None if cancelled
    """
    if frame is None or frame.size == 0:
        raise ValueError("Empty frame provided to ROI selector.")

    img = frame.copy()

    # Draw existing ROI as a hint
    if existing is not None:
        x1, y1, x2, y2 = map(int, existing)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, "Existing ROI (green) - draw new one if needed",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # OpenCV selectROI expects a window; make it resizable
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    # Returns (x, y, w, h). If user cancels, returns (0,0,0,0)
    x, y, w, h = cv2.selectROI(title, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)

    if w == 0 or h == 0:
        if allow_cancel:
            return None
        raise RuntimeError("ROI selection cancelled / invalid (w or h == 0).")

    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
    return (x1, y1, x2, y2)



def read_image(path):
    # read image use cv2 or PIL
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    return image
    



class ScoreboardChangeDetector:
    def __init__(self, frames_folder, video_fps, existing_close_coord=None, existing_far_coord=None):
        self.frames_folder = frames_folder
        self.video_fps = video_fps

        num_images = len([name for name in os.listdir(frames_folder)
                          if os.path.isfile(os.path.join(frames_folder, name))])
        print(f"Number of images in folder: {num_images}")

        random_image_index = np.random.randint(1, num_images + 1)
        print(f"Selected image index for region selection: {random_image_index}")

        sample_image_path = os.path.join(frames_folder, f"{random_image_index:06d}.jpg")
        sample_frame = read_image(sample_image_path)

        print("Please select the close scoreboard region in the displayed frame.")
        self.close_scoreboard_region = select_roi(sample_frame, title="Select CLOSE scoreboard ROI", existing=existing_close_coord)

        print("Please select the far scoreboard region in the displayed frame.")
        self.far_scoreboard_region = select_roi(sample_frame, title="Select FAR scoreboard ROI", existing=existing_far_coord)

    def detect_changes(
        self,
        stride=1,
        warmup=10,            # frames to initialize stable reference
        patience=3,           # consecutive frames that must indicate change
        diff_threshold=12.0,  # higher => less sensitive (tune this)
        min_interval_frames=10,  # prevent double-counting flicker
        blur_ksize=5,         # 0 to disable blur
        debug=False,
    ):
        """
        Detect scoreboard changes by image difference only.
        Assumes each ROI starts at 0 and increments by 1 whenever a stable change is detected.

        Returns:
            events: list of dicts like:
              {
                "frame": int,
                "time": "M:SS",
                "roi": "close"|"far",
                "old_score": int,
                "new_score": int,
                "diff": float
              }
            scores: final scores dict {"close": int, "far": int}
        """

        # --- helpers ---
        def crop(frame, roi):
            if roi is None:
                return None
            x1, y1, x2, y2 = map(int, roi)
            H, W = frame.shape[:2]
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                return None
            patch = frame[y1:y2, x1:x2]
            return patch if patch.size > 0 else None

        def preprocess(patch_bgr):
            # Convert to grayscale + optional blur to reduce noise / compression artifacts
            g = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
            if blur_ksize and blur_ksize > 0:
                k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
                g = cv2.GaussianBlur(g, (k, k), 0)
            return g

        def diff_score(a_gray, b_gray):
            # mean absolute pixel difference
            d = cv2.absdiff(a_gray, b_gray)
            return float(d.mean())

        # --- frame list ---
        frame_files = sorted([f for f in os.listdir(self.frames_folder)
                              if f.lower().endswith((".jpg", ".jpeg", ".png"))])

        # Scores start at 0
        scores = {"close": 0, "far": 0}
        events = []

        # Stable references (what ROI looked like last confirmed state)
        stable_ref = {"close": None, "far": None}

        # Debounce/candidate state per ROI
        cand_run = {"close": 0, "far": 0}
        cand_best_diff = {"close": 0.0, "far": 0.0}

        # Prevent double counting
        last_event_frame = {"close": -10**9, "far": -10**9}

        accepted = 0
        pbar = tqdm(frame_files, desc="Processing frames for scoreboard change detection")
        for i, fname in enumerate(pbar):
            if stride > 1 and (i % stride != 0):
                continue

            frame_path = os.path.join(self.frames_folder, fname)
            frame = read_image(frame_path)  # expects BGR numpy array
            if frame is None:
                continue

            frame_idx = int(os.path.splitext(fname)[0])

            for roi_name, roi in [("close", self.close_scoreboard_region),
                                  ("far", self.far_scoreboard_region)]:
                patch = crop(frame, roi)
                if patch is None:
                    continue

                cur = preprocess(patch)

                # Initialize stable reference for first few frames
                if stable_ref[roi_name] is None:
                    stable_ref[roi_name] = cur
                    continue

                # Warmup period: keep updating stable_ref to “settle”
                if accepted < warmup:
                    stable_ref[roi_name] = cur
                    continue

                # Cooldown: avoid counting multiple times too fast
                if frame_idx - last_event_frame[roi_name] < min_interval_frames:
                    continue

                d = diff_score(cur, stable_ref[roi_name])

                if debug:
                    print(f"[{roi_name}] frame={frame_idx} diff={d:.2f} score={scores[roi_name]}")

                if d >= diff_threshold:
                    cand_run[roi_name] += 1
                    cand_best_diff[roi_name] = max(cand_best_diff[roi_name], d)
                else:
                    # decay/reset candidate if similarity returns
                    cand_run[roi_name] = 0
                    cand_best_diff[roi_name] = 0.0
                    # also refresh stable reference slowly (helps with lighting drift)
                    stable_ref[roi_name] = cur

                # Confirm change if persistent
                if cand_run[roi_name] >= patience:
                    old_s = scores[roi_name]
                    scores[roi_name] = old_s + 1

                    events.append({
                        "frame": frame_idx,
                        "time": self.calculate_time_from_frame(frame_idx),
                        "roi": roi_name,
                        "old_score": old_s,
                        "new_score": scores[roi_name],
                        "diff": cand_best_diff[roi_name],
                    })

                    last_event_frame[roi_name] = frame_idx

                    # After confirming a change, update the stable reference to current
                    stable_ref[roi_name] = cur

                    # reset candidate
                    cand_run[roi_name] = 0
                    cand_best_diff[roi_name] = 0.0

            accepted += 1

        return events, scores

    def calculate_time_from_frame(self, frame_idx):
        total_seconds = frame_idx / self.video_fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}:{seconds:02d}"





class ResNetScoreboardChangeDetector:
    def __init__(self, frames_folder, video_fps, model_path, device, existing_close_coord=None, existing_far_coord=None):
        self.frames_folder = frames_folder
        self.video_fps = video_fps
        self.device = device
        self.load_pretrained_model(model_path, device)
        # count how many images in the folder
        num_images = len([name for name in os.listdir(frames_folder) if os.path.isfile(os.path.join(frames_folder, name))])
        print(f"Number of images in folder: {num_images}")
        # select random image from the folder
        random_image_index = np.random.randint(1, num_images + 1)
        print(f"Selected image index for region selection: {random_image_index}")
        sample_image_path = os.path.join(frames_folder, f"{random_image_index:06d}.jpg")
        sample_frame = self._read_image(sample_image_path)
        print("Please select the close scoreboard region in the displayed frame.")
        self.close_scoreboard_region = self.select_close_scoreboard_region(sample_frame, existing_close_coord)
        print("Please select the far scoreboard region in the displayed frame.")
        self.far_scoreboard_region = self.select_far_scoreboard_region(sample_frame, existing_far_coord)

    def _read_image(self, path):
        # read image use cv2 or PIL
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return image

    def load_pretrained_model(self, model_path, device):
        self.model = ScoreClassifier(num_classes=14, backbone="resnet34")
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.to(device)
        self.model.eval()  # set to eval mode
    

    def _select_roi(self, frame, title="Select ROI", existing=None, allow_cancel=True):
        """
        frame: numpy HxWxC (BGR)
        existing: optional (x1,y1,x2,y2) to show as a hint (we can't prefill selectROI,
                  but we can draw it before selecting)
        returns: (x1,y1,x2,y2) or None if cancelled
        """
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame provided to ROI selector.")

        img = frame.copy()

        # Draw existing ROI as a hint
        if existing is not None:
            x1, y1, x2, y2 = map(int, existing)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, "Existing ROI (green) - draw new one if needed",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # OpenCV selectROI expects a window; make it resizable
        cv2.namedWindow(title, cv2.WINDOW_NORMAL)

        # Returns (x, y, w, h). If user cancels, returns (0,0,0,0)
        x, y, w, h = cv2.selectROI(title, img, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(title)

        if w == 0 or h == 0:
            if allow_cancel:
                return None
            raise RuntimeError("ROI selection cancelled / invalid (w or h == 0).")

        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        return (x1, y1, x2, y2)

    def _save_rois(self):
        if not self.roi_save_path:
            return
        os.makedirs(os.path.dirname(self.roi_save_path), exist_ok=True)
        with open(self.roi_save_path, "w") as f:
            json.dump(self.rois, f, indent=2)

    def select_close_scoreboard_region(self, frame, existing=None):
        """
        User draws ROI for close scoreboard.
        Returns (x1,y1,x2,y2) or None if cancelled.
        """
        if existing == None:
            roi = self._select_roi(frame, title="Select CLOSE scoreboard ROI")
        else:
            roi = existing
       
        return roi

    def select_far_scoreboard_region(self, frame, existing=None):
        """
        User draws ROI for far scoreboard.
        Returns (x1,y1,x2,y2) or None if cancelled.
        """
        if existing == None:
            roi = self._select_roi(frame, title="Select FAR scoreboard ROI")
        else:
            roi = existing
        return roi

    def detect_changes(
        self,
        conf_threshold=0.90,
        patience=3,
        warmup=5,
        stride=1,
    ):
        """
        Detect scoreboard score changes using BOTH close and far ROIs independently,
        and emit events containing old/new for BOTH (with confidence).

        Event format:
        {
            "frame": int,
            "time_sec": float,
            "old": {
            "close": {"score": int, "conf": float} | None,
            "far":   {"score": int, "conf": float} | None
            },
            "new": {
            "close": {"score": int, "conf": float} | None,
            "far":   {"score": int, "conf": float} | None
            }
        }
        """
        device = self.device
        transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225]),
        ])
        def is_valid_transition(old_s, new_s, allow_reset=False, max_score=13):
            if old_s is None:
                return True
            if new_s == old_s:
                return True
            if new_s == old_s + 1 and new_s <= max_score:
                return True
            if allow_reset and old_s >= max_score - 1 and new_s == 0:
                # optional: allow reset near the end of a game
                return True
            return False
        
        def clamp_roi(roi, W, H):
            x1, y1, x2, y2 = map(int, roi)
            x1 = max(0, min(x1, W - 1))
            x2 = max(0, min(x2, W))
            y1 = max(0, min(y1, H - 1))
            y2 = max(0, min(y2, H))
            if x2 <= x1 or y2 <= y1:
                return None
            return (x1, y1, x2, y2)

        def crop_bgr(frame_bgr, roi):
            if roi is None:
                return None
            H, W = frame_bgr.shape[:2]
            roi = clamp_roi(roi, W, H)
            if roi is None:
                return None
            x1, y1, x2, y2 = roi
            patch = frame_bgr[y1:y2, x1:x2]
            if patch.size == 0:
                return None
            return patch

        def infer_patch(patch_bgr):
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(patch_rgb)
            x = transform(pil).unsqueeze(0).to(device)

            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            return int(pred.item()), float(conf.item())

        frame_files = sorted(
            [f for f in os.listdir(self.frames_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        )

        # Per-frame logs (optional debug)
        per_frame = []

        # Stable + candidate states per ROI
        # Each state is dict: {"score": int, "conf": float} or None
        stable = {"close": None, "far": None}

        candidate_score = {"close": None, "far": None}
        candidate_best_conf = {"close": 0.0, "far": 0.0}
        candidate_run = {"close": 0, "far": 0}
        candidate_start_frame = {"close": None, "far": None}

        accepted_count = {"close": 0, "far": 0}

        events = []

        started = False
        start_frame = None

        bqar = tqdm(frame_files, desc="Processing frames for scoreboard change detection")
        for idx, fname in enumerate(bqar):
            if stride > 1 and (idx % stride != 0):
                continue

            frame_path = os.path.join(self.frames_folder, fname)
            frame = self._read_image(frame_path)

            frame_index = int(os.path.splitext(fname)[0])
            time_sec = frame_index / float(self.video_fps)

            # Infer both rois (if available)
            preds = {"close": None, "far": None}  # each is {"score": s, "conf": c} or None

            close_patch = crop_bgr(frame, self.close_scoreboard_region)
            if close_patch is not None:
                s, c = infer_patch(close_patch)
                preds["close"] = {"score": s, "conf": c}

            far_patch = crop_bgr(frame, self.far_scoreboard_region)
            if far_patch is not None:
                s, c = infer_patch(far_patch)
                preds["far"] = {"score": s, "conf": c}

            per_frame.append({
                "frame": frame_index,
                "time_sec": time_sec,
                "close": preds["close"],
                "far": preds["far"],
            })

            # --- NEW: wait until BOTH close and far are confidently 0 to start ---
            if not started:
                c0 = preds["close"] is not None and preds["close"]["conf"] >= conf_threshold and preds["close"]["score"] == 0
                f0 = preds["far"]   is not None and preds["far"]["conf"]   >= conf_threshold and preds["far"]["score"] == 0

                if c0 and f0:
                    started = True
                    start_frame = frame_index

                    # initialize stable to 0-0
                    stable["close"] = preds["close"]
                    stable["far"]   = preds["far"]

                    # reset candidate state & counts so warmup starts cleanly AFTER 0-0
                    for roi_name in ["close", "far"]:
                        candidate_score[roi_name] = None
                        candidate_best_conf[roi_name] = 0.0
                        candidate_run[roi_name] = 0
                        candidate_start_frame[roi_name] = None
                        accepted_count[roi_name] = 0

                # do NOT run detection logic before started
                continue

            # Process each ROI independently
            roi_confirmed_change = {"close": False, "far": False}
            roi_new_state = {"close": None, "far": None}
            roi_old_state = {"close": None, "far": None}

            for roi_name in ["close", "far"]:
                cur = preds[roi_name]
                if cur is None:
                    continue

                # reject low confidence
                if cur["conf"] < conf_threshold:
                    continue

                accepted_count[roi_name] += 1

                # stable should already be initialized from the 0-0 start gate
                if stable[roi_name] is None:
                    continue

                # warmup: keep updating stable early
                if accepted_count[roi_name] <= warmup:
                    stable[roi_name] = cur
                    # also reset candidate
                    candidate_score[roi_name] = None
                    candidate_best_conf[roi_name] = 0.0
                    candidate_run[roi_name] = 0
                    candidate_start_frame[roi_name] = None
                    continue

                # If same as stable, reset candidate
                if cur["score"] == stable[roi_name]["score"]:
                    stable[roi_name] = cur  # refresh conf
                    candidate_score[roi_name] = None
                    candidate_best_conf[roi_name] = 0.0
                    candidate_run[roi_name] = 0
                    candidate_start_frame[roi_name] = None
                    continue


                old_s = stable[roi_name]["score"]
                # hard constraint: next score must be old+1 (or optional reset)
                if not is_valid_transition(old_s, cur["score"], allow_reset=False, max_score=13):
                    # ignore this prediction as impossible jump
                    continue

            
                # Build/advance candidate
                if candidate_score[roi_name] is None or cur["score"] != candidate_score[roi_name]:
                    candidate_score[roi_name] = cur["score"]
                    candidate_best_conf[roi_name] = cur["conf"]
                    candidate_run[roi_name] = 1
                    candidate_start_frame[roi_name] = frame_index
                else:
                    candidate_run[roi_name] += 1
                    candidate_best_conf[roi_name] = max(candidate_best_conf[roi_name], cur["conf"])

                # Confirm change
                if candidate_run[roi_name] >= patience:
                    roi_confirmed_change[roi_name] = True
                    roi_old_state[roi_name] = stable[roi_name]
                    roi_new_state[roi_name] = {"score": candidate_score[roi_name], "conf": candidate_best_conf[roi_name]}

                    # update stable
                    stable[roi_name] = roi_new_state[roi_name]

                    # reset candidate
                    candidate_score[roi_name] = None
                    candidate_best_conf[roi_name] = 0.0
                    candidate_run[roi_name] = 0
                    candidate_start_frame[roi_name] = None

            # Emit ONE combined event if either ROI changed on this frame
            if roi_confirmed_change["close"] or roi_confirmed_change["far"]:
                # old/new should include BOTH ROIs (even if only one changed)
                old_combined = {
                    "close": roi_old_state["close"] if roi_old_state["close"] is not None else stable["close"],
                    "far": roi_old_state["far"] if roi_old_state["far"] is not None else stable["far"],
                }
                new_combined = {
                    "close": roi_new_state["close"] if roi_new_state["close"] is not None else stable["close"],
                    "far": roi_new_state["far"] if roi_new_state["far"] is not None else stable["far"],
                }

                events.append({
                    "frame": frame_index,
                    "time_sec": time_sec,
                    "old": old_combined,
                    "new": new_combined,
                })

        return events, per_frame




def read_txt(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def calculate_pr_recall(pred, gt, tol=2.0):
    """
    pred: dict like { "188": {"score": 113021, "time": "0:06"}, ... }  OR any mapping with 'time'
    gt  : iterable of timecodes (e.g., ["0:06","0:34", ...]) or seconds (float/int)
    tol : tolerance in SECONDS for a match (|pred - gt| <= tol)
    """
    import numpy as np

    def time_to_seconds(t):
        # Accept "mm:ss", "hh:mm:ss", "ss(.ms)" or numeric
        if isinstance(t, (int, float)):
            return float(t)
        s = str(t).strip()
        if ":" not in s:
            return float(s)
        parts = s.split(":")
        parts = [float(p) for p in parts]
        if len(parts) == 2:   # mm:ss
            m, sec = parts
            return m * 60 + sec
        elif len(parts) == 3: # hh:mm:ss
            h, m, sec = parts
            return h * 3600 + m * 60 + sec
        else:
            raise ValueError(f"Unsupported time format: {t}")

    # --- collect prediction times (prefer value['time']; fallback to key if parseable) ---
    pred_seconds = []
    for k, v in pred.items():
        if isinstance(v, dict) and "time" in v:
            pred_seconds.append(time_to_seconds(v["time"]))
        else:
            # fallback: try key as seconds/timecode
            pred_seconds.append(time_to_seconds(k))

    gt_seconds = [time_to_seconds(t) for t in gt]

    # sort
    pred_seconds = np.array(sorted(pred_seconds), dtype=float)
    gt_seconds   = np.array(sorted(gt_seconds),   dtype=float)

    # two-pointer matching
    i = j = 0
    TP = FP = FN = 0
    matches = []  # (gt_time, pred_time, delta_seconds)

    while i < len(pred_seconds) and j < len(gt_seconds):
        d = pred_seconds[i] - gt_seconds[j]
        if abs(d) <= tol:
            TP += 1
            matches.append((gt_seconds[j], pred_seconds[i], d))
            i += 1
            j += 1
        elif pred_seconds[i] < gt_seconds[j] - tol:
            FP += 1
            i += 1
        else:
            FN += 1
            j += 1

    FP += (len(pred_seconds) - i)
    FN += (len(gt_seconds)   - j)

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # print summary (optional)
    print(f"TP={TP}  FP={FP}  FN={FN}  tol={tol}s")
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1_score:.4f}")

    return {
        "precision": precision, "recall": recall, "f1": f1_score,
        "TP": TP, "FP": FP, "FN": FN,
        "matches": matches,
        "pred_times_s": pred_seconds.tolist(),
        "gt_times_s": gt_seconds.tolist(),
        "tol_s": tol,
    }



if __name__ == "__main__":

    # frames_folder = "/home/august/github/TTA_Project/data/25WPE_SLO_M11_SF_Creange_FRA_v_von_Einem_AUS_game1_frames"
    frames_folder = "/home/august/github/TTA_Project/data/25WPF_TPE_M11_G_Chen_Po_Yen_TPE_v_Murakami_JPN_game1_frames"
    video_fps = 30  # adjust as needed
    detector = ScoreboardChangeDetector(frames_folder, video_fps)
    changes = detector.detect_changes()
    print(f"In total {len(changes)} scoreboard changes detected at frames:")

    # save timeline to json
    with open("score_timeline.json", "w") as f:
        json.dump(changes, f, indent=2)

    # gts = read_txt('/home/august/github/TTA_Project/src/scoreboard_detector/gt_SF.TXT')
    # with open("score_timeline.json", "r") as f:
    #     pred = json.load(f)
    # result = calculate_pr_recall(pred, gts, tol=5.0)
    # print(result)