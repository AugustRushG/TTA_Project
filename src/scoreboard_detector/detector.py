import os
import cv2
import numpy as np
import json

class ScoreboardChangeDetector:
    def __init__(self, frames_folder, video_fps, manual_scoreboard_region=None):
        self.frames_folder = frames_folder
        self.video_fps = video_fps

        # let user select scoreboard region from a sample frame
        sample_frame = self._read_image(os.path.join(frames_folder, os.listdir(frames_folder)[0]), crop=False)
        print("Please select the scoreboard region in the displayed frame.")
        self.scoreboard_region = self.select_scoreboard_region(sample_frame, mode="poly", manual_input=manual_scoreboard_region)
    
    def _to_bgr(self, frame):
        # Accept PIL or numpy; return numpy BGR
        if hasattr(frame, "mode"):  # PIL Image
            frame = np.array(frame)[:, :, ::-1].copy()  # RGB->BGR
        return frame

    def _order_quad_xy_tl(self, pts):
        """Order 4 (x,y) points to [TL, TR, BR, BL]."""
        pts = np.asarray(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(d)]
        bl = pts[np.argmax(d)]
        return np.stack([tl, tr, br, bl], axis=0)

    def select_scoreboard_region(self, frame, mode="poly", manual_input=None):
        """
        Let the user select the scoreboard region.
        Args:
            frame: numpy BGR image or PIL.Image
            mode : 'rect' (drag a rectangle) or 'poly' (click 4 corners)
        Returns:
            If mode='rect':
                (x, y, w, h)  integers in pixel coords (origin top-left)
            If mode='poly':
                quad_pts in *your* expected order:
                [left_bottom, right_bottom, right_top, left_top] as float32 (x,y)
            Returns None if user cancels (ESC).
        """

        if manual_input is not None:
            print("Using Manual Input for Scoreboard detector")
            return manual_input

        img = self._to_bgr(frame).copy()

        if mode == "rect":
            cv2.namedWindow("Select ROI - Drag, ENTER to confirm, ESC to cancel", cv2.WINDOW_NORMAL)
            r = cv2.selectROI("Select ROI - Drag, ENTER to confirm, ESC to cancel", img, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow("Select ROI - Drag, ENTER to confirm, ESC to cancel")
            x, y, w, h = map(int, r)
            if w == 0 or h == 0:
                return None
            return (x, y, w, h)

        elif mode == "poly":
            win = "Click 4 corners (any order). ENTER=confirm, r=reset, ESC=cancel"
            pts = []

            def on_mouse(event, x, y, flags, param):
                nonlocal pts
                if event == cv2.EVENT_LBUTTONDOWN:
                    if len(pts) < 4:
                        pts.append((x, y))
                elif event == cv2.EVENT_RBUTTONDOWN and pts:
                    pts.pop()  # quick undo with right-click

            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(win, on_mouse)

            while True:
                disp = img.copy()
                # draw clicks
                for i, (x, y) in enumerate(pts):
                    cv2.circle(disp, (x, y), 5, (0, 255, 255), -1)
                    cv2.putText(disp, str(i+1), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                # connect lines
                if len(pts) >= 2:
                    for i in range(1, len(pts)):
                        cv2.line(disp, pts[i-1], pts[i], (0, 255, 255), 2)
                if len(pts) == 4:
                    cv2.line(disp, pts[3], pts[0], (0, 255, 255), 2)

                cv2.putText(disp, "L-click: add  |  R-click: undo  |  r: reset  |  ENTER: confirm  |  ESC: cancel",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 200, 30), 2)

                cv2.imshow(win, disp)
                key = cv2.waitKey(20) & 0xFF
                if key == 27:  # ESC
                    cv2.destroyWindow(win)
                    return None
                elif key in (ord('\r'), 13):  # ENTER/RETURN
                    if len(pts) == 4:
                        break
                elif key in (ord('r'), ord('R')):
                    pts = []

            cv2.destroyWindow(win)

            # Order to TL,TR,BR,BL then convert to your expected [LB,RB,RT,LT]
            TL, TR, BR, BL = self._order_quad_xy_tl(pts)
            LB, RB, RT, LT = BL, BR, TR, TL
            quad_lb_rb_rt_lt = np.stack([LB, RB, RT, LT], axis=0).astype(np.float32)
            return quad_lb_rb_rt_lt

        else:
            raise ValueError("mode must be 'rect' or 'poly'")
        
    def warp_scoreboard_tltrbrbl(self, frame, src, out_size):
        W, H = out_size
        dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(np.float32(src), dst)
        warped = cv2.warpPerspective(frame, M, (W, H), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        return warped, M

    def _crop_scoreboard(self, frame):
        if self.scoreboard_region is None:
            raise ValueError("Scoreboard region not set.")

        pts = np.asarray(self.scoreboard_region, dtype=np.float32)
        lb, rb, rt, lt = pts
        pts = np.array([lt, rt, rb, lb], dtype=np.float32)  # TL,TR,BR,BL

        # Polygon: pts is TL,TR,BR,BL (or any order you already normalized to TL,TR,BR,BL)
        if pts.shape == (4, 2):
            # 1) Crop to the quad's bounding box
            x_min = int(np.floor(pts[:, 0].min()))
            x_max = int(np.ceil(pts[:, 0].max()))
            y_min = int(np.floor(pts[:, 1].min()))
            y_max = int(np.ceil(pts[:, 1].max()))

            # clip to image bounds
            Hf, Wf = frame.shape[:2]
            x_min = max(0, min(x_min, Wf-1)); x_max = max(0, min(x_max, Wf-1))
            y_min = max(0, min(y_min, Hf-1)); y_max = max(0, min(y_max, Hf-1))
            if x_max <= x_min or y_max <= y_min:
                raise ValueError("Scoreboard quad is out of bounds or degenerate.")

            cropped = frame[y_min:y_max+1, x_min:x_max+1].copy()

            # 2) Shift points to local coords of the crop
            src_local = pts.copy()
            src_local[:, 0] -= x_min
            src_local[:, 1] -= y_min

            # 3) Warp
            warped, _ = self.warp_scoreboard_tltrbrbl(cropped, src_local, (100, 100))
            # cv2.imshow("Cropped Scoreboard", warped); cv2.waitKey(1)  # beware headless envs
            return warped

        # Rectangle (x, y, w, h)
        elif len(pts) == 4 and not hasattr(pts[0], "__len__"):
            x, y, w, h = map(int, pts)
            return frame[y:y+h, x:x+w]

        else:
            raise ValueError("Invalid scoreboard region format.")


    def _read_image(self, path, crop=True):
        # read image use cv2 or PIL
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {path}")
        if crop:
            image = self._crop_scoreboard(image)
        return image
    
    def nms_score_changes(self, change_times, fps, window_s=5.0):
        """
        Temporal NMS (no side). Keeps the strongest detection within ±window_s seconds.
        Args:
            change_times: dict {frame:int -> score:int}
            fps:          video framerate
            window_s:     seconds for suppression window
        Returns:
            dict {frame:int -> score:int} (sorted by frame)
        """
        if not change_times:
            return {}

        window_f = int(round(window_s * fps))

        # sort by score desc, then frame asc
        items = sorted(change_times.items(), key=lambda kv: (-kv[1], kv[0]))
        kept = []
        suppressed = set()

        for i, (fi, si) in enumerate(items):
            if fi in suppressed:
                continue
            kept.append((fi, si))
            # suppress neighbors in temporal window
            for fj, sj in items[i+1:]:
                if abs(fj - fi) <= window_f:
                    suppressed.add(fj)

        return dict(sorted(kept, key=lambda kv: kv[0]))


    
    def detect_changes(self):

        change_times = {}
        prev_gray = None
        state = "STABLE"     # STABLE | CANDIDATE
        pending_frame = None

        # thresholds as mean-per-pixel on 0..255 scale
        tau_enter_mean    = 12.0   # flag change (vs previous frame)
        tau_confirm_mean  =  9.0   # confirm change (hysteresis)
        tau_block_mean    = 30.0   # BIG change vs the FIRST frame => board moved/blocked

        # sort numerically by frame index in filename (e.g., "123.jpg")
        files = sorted(
            [f for f in os.listdir(self.frames_folder) if f.lower().endswith(('.jpg', '.png'))],
            key=lambda f: int(os.path.splitext(f)[0])
        )

        original_gray = None  # keep the very first ROI as reference

        for frame_file in files:
            frame_path = os.path.join(self.frames_folder, frame_file)

            # _read_image should return rectified/cropped ROI (BGR uint8)
            frame = self._read_image(frame_path)
            if frame is None:
                continue

            # light denoise + gray
            roi  = cv2.GaussianBlur(frame, (3, 3), 0)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
          

            if original_gray is None:
                original_gray = gray.copy()

            if prev_gray is None:
                prev_gray = gray
                continue
            

            H, W = gray.shape
            N = H * W
            tau_enter_sum   = tau_enter_mean   * N
            tau_confirm_sum = tau_confirm_mean * N
            tau_block_sum   = tau_block_mean   * N

            # local change (vs previous frame)
            d_prev = float(np.sum(cv2.absdiff(gray, prev_gray)))
            # global change (vs first frame)
            d_orig = float(np.sum(cv2.absdiff(gray, original_gray)))

            frame_num = int(os.path.splitext(frame_file)[0])

            # --- block/movement guard: if too far from the first frame, skip everything ---
            if d_orig >= tau_block_sum:
                # Treat as board moving/blocked; don't arm/confirm, just advance.
                state = "STABLE"
                pending_frame = None
                prev_gray = gray
                continue

            # --- regular 2-stage detect (whole-ROI) ---
            if state == "STABLE":
                if d_prev >= tau_enter_sum:
                    state = "CANDIDATE"
                    pending_frame = frame_num
            else:  # CANDIDATE
                if d_prev >= tau_confirm_sum:
                    # commit event at the first (pending) frame
                    change_times[pending_frame] = int(d_prev)
                # reset either way
                state = "STABLE"
                pending_frame = None

            prev_gray = gray

        # temporal de-duplication (no side)
        change_times = self.nms_score_changes(change_times, self.video_fps, window_s=5.0)

        for f, s in change_times.items():
            t = self.calculate_time_from_frame(f)
            change_times[f] = {'frame':f, 'score': s, 'time': t}

        return change_times

    
    def calculate_time_from_frame(self, frame_idx):
        total_seconds = frame_idx / self.video_fps
        minutes = int(total_seconds // 60)
        seconds = int(total_seconds % 60)
        return f"{minutes}:{seconds:02d}"


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