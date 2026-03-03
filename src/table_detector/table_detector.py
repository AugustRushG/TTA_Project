import cv2
import numpy as np
import matplotlib.pyplot as plt


class TableDetector:
    """
    Click 6 points ONCE: TL, TR, ML, MR, BL, BR
    Builds two homographies that share the same ML/MR seam points.
    """

    def __init__(self, image_path, topdown_width=334, topdown_height=600):
        self.image_path = image_path
        self.topdown_width = int(topdown_width)
        self.topdown_height = int(topdown_height)

        self.img_bgr = cv2.imread(self.image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {self.image_path}")
        self.img_h, self.img_w = self.img_bgr.shape[:2]

        # destination seam y (float) so both halves share EXACT same line
        self.y_mid = (self.topdown_height - 1) / 2.0

        Wm1 = self.topdown_width - 1
        Hm1 = self.topdown_height - 1
        ym = self.y_mid

        # TL, TR, ML, MR
        self.pts_dst_top = np.array([
            [0.0, 0.0],
            [Wm1, 0.0],
            [0.0, ym],
            [Wm1, ym],
        ], dtype=np.float32)

        # ML, MR, BL, BR
        self.pts_dst_bottom = np.array([
            [0.0, ym],
            [Wm1, ym],
            [0.0, Hm1],
            [Wm1, Hm1],
        ], dtype=np.float32)

        self.H_top = None
        self.H_bottom = None

        # seam in IMAGE space (computed from ML/MR)
        self.seam_y_img = None

        # store clicked points
        self.corners6 = None  # [TL, TR, ML, MR, BL, BR]

    def _show_and_click(self, title, n_points):
        img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots()
        ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis("off")

        pts = plt.ginput(n_points, timeout=0)
        plt.close(fig)

        if len(pts) != n_points:
            raise ValueError(f"Expected {n_points} points, got {len(pts)}")

        return np.array(pts, dtype=np.float32)

    def annotate_table_6pts(self):
        """
        Click 6 points once:
        TL, TR, mid-left, mid-right, BL, BR
        """
        print("Annotate 6 points: TL, TR, mid-left, mid-right, BL, BR")
        self.corners6 = self._show_and_click(
            "Click 6 points: TL, TR, mid-left, mid-right, BL, BR", 6
        )
        return self.corners6

    def set_corners_manual_6pts(self, corners6):
        corners6 = np.asarray(corners6, dtype=np.float32)
        if corners6.shape != (6, 2):
            raise ValueError("corners6 must be shape (6,2): TL,TR,ML,MR,BL,BR")
        self.corners6 = corners6

    @staticmethod
    def _reproj_err(H, src, dst):
        src = np.float32(src).reshape(-1, 1, 2)
        proj = cv2.perspectiveTransform(src, H).reshape(-1, 2)
        dst = np.float32(dst).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        return err

    def compute_homographies(self, corners6=None, sanity_check=True):
        """
        corners6: optional (6,2). If None, uses self.corners6 (or asks for clicks).
        """
        if corners6 is not None:
            self.set_corners_manual_6pts(corners6)
        if self.corners6 is None:
            self.annotate_table_6pts()

        TL, TR, ML, MR, BL, BR = self.corners6

        # shared seam points ML/MR are used in BOTH homographies
        src_top = np.array([TL, TR, ML, MR], dtype=np.float32)
        src_bottom = np.array([ML, MR, BL, BR], dtype=np.float32)

        # calculate top to mid distance 
        distance_top_mid_left = np.linalg.norm(ML - TL)
        distance_top_mid_right = np.linalg.norm(MR - TR)
        distance_top_mid = (distance_top_mid_left + distance_top_mid_right) / 2
        distance_mid_bottom_left = np.linalg.norm(BL - ML)
        distance_mid_bottom_right = np.linalg.norm(BR - MR)
        distance_mid_bottom = (distance_mid_bottom_left + distance_mid_bottom_right) / 2
        print(f"Distance TL-ML: {distance_top_mid:.2f}, ML-BL: {distance_mid_bottom:.2f}")
        # Since the camera is closer to the bottom half, the top half will be more compressed. 
        # To compensate for this, we can adjust the destination points to create a more balanced top-down view.
        gap_adjust_left = distance_top_mid_left - distance_mid_bottom_left
        gap_adjust_right = distance_top_mid_right - distance_mid_bottom_right
        print(f"Gap adjust left: {gap_adjust_left:.2f}, right: {gap_adjust_right:.2f}")
        self.pts_dst_top[2][1] += gap_adjust_left * 0.5  # adjust ML y
        self.pts_dst_top[3][1] += gap_adjust_right * 0.5  # adjust MR y
        self.pts_dst_bottom[0][1] -= gap_adjust_left * 0.5  # adjust ML y
        self.pts_dst_bottom[1][1] -= gap_adjust_right * 0.5  # adjust MR y

        self.H_top, _ = cv2.findHomography(src_top, self.pts_dst_top, method=0)
        self.H_bottom, _ = cv2.findHomography(src_bottom, self.pts_dst_bottom, method=0)

        if self.H_top is None or self.H_bottom is None:
            raise RuntimeError("Failed to compute one or both homographies.")

        # seam in image space = average y of ML/MR (robust vs using img_h//2)
        self.seam_y_img = float((ML[1] + MR[1]) * 0.5)

        if sanity_check:
            err_top = self._reproj_err(self.H_top, src_top, self.pts_dst_top)
            err_bot = self._reproj_err(self.H_bottom, src_bottom, self.pts_dst_bottom)
            print(f"[sanity] top reprojection error (px): {err_top}")
            print(f"[sanity] bot reprojection error (px): {err_bot}")

        print("Homographies computed (shared seam points). seam_y_img =", self.seam_y_img)
        return self.H_top, self.H_bottom

    def transform_ball(self, ball_x, ball_y, blend_band_px=0):
        """
        Map a ball point to topdown space.

        blend_band_px:
            0  -> hard switch top/bottom by seam_y_img
            >0 -> blend top & bottom results within +/- band around seam (recommended ~10-30)
        """
        if self.H_top is None or self.H_bottom is None or self.seam_y_img is None:
            raise RuntimeError("Call compute_homographies() first.")

        pt = np.array([[[float(ball_x), float(ball_y)]]], dtype=np.float32)

        p_top = cv2.perspectiveTransform(pt, self.H_top)[0, 0]
        p_bot = cv2.perspectiveTransform(pt, self.H_bottom)[0, 0]

        if blend_band_px <= 0:
            # hard switch using seam derived from ML/MR
            return p_top if ball_y <= self.seam_y_img else p_bot

        # smooth blend around seam in IMAGE space
        y0 = self.seam_y_img - blend_band_px
        y1 = self.seam_y_img + blend_band_px
        if ball_y <= y0:
            return p_top
        if ball_y >= y1:
            return p_bot

        t = (ball_y - y0) / (y1 - y0)  # 0..1
        return (1.0 - t) * p_top + t * p_bot

    def warp_table(self, save_path=None):
        """
        Warp both halves and stitch them on the seam line in topdown space.
        """
        if self.H_top is None or self.H_bottom is None:
            raise RuntimeError("Call compute_homographies() first.")

        warped_top = cv2.warpPerspective(
            self.img_bgr, self.H_top, (self.topdown_width, self.topdown_height)
        )
        warped_bottom = cv2.warpPerspective(
            self.img_bgr, self.H_bottom, (self.topdown_width, self.topdown_height)
        )

        out = np.zeros_like(warped_top)

        seam_row = int(round(self.y_mid))  # seam in topdown space
        out[:seam_row + 1] = warped_top[:seam_row + 1]
        out[seam_row + 1:] = warped_bottom[seam_row + 1:]

        if save_path:
            cv2.imwrite(save_path, out)
            print(f"Warped table saved to {save_path}")

        return out