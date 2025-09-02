import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

class TableDetector:
    def __init__(self, image_path, topdown_width=334, topdown_height=600):
        self.image_path = image_path
        self.topdown_width = topdown_width
        self.topdown_height = topdown_height

        # destination points for each half
        self.pts_dst_top = np.array([
            [0, 0],
            [topdown_width-1, 0],
            [0, topdown_height//2-1],
            [topdown_width-1, topdown_height//2-1]
        ], dtype=np.float32)

        self.pts_dst_bottom = np.array([
            [0, topdown_height//2],
            [topdown_width-1, topdown_height//2],
            [0, topdown_height-1],
            [topdown_width-1, topdown_height-1]
        ], dtype=np.float32)

        self.H_top = None
        self.H_bottom = None
        
    def annotate_half(self, title):
        img_bgr = cv2.imread(self.image_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots()
        ax.imshow(img_rgb)
        ax.set_title(title)
        ax.axis('off')

        # Block until 4 clicks (no timeout)
        pts = plt.ginput(4, timeout=0)   # list of (x, y)
        plt.close(fig)

        if len(pts) != 4:
            raise ValueError(f"Expected 4 points, got {len(pts)}")
        return np.array(pts, dtype=np.float32)
    
    def expand_corners_anisotropic(self, corners, scale_x=1.0, scale_y=1.1):
        center = corners.mean(axis=0, keepdims=True)
        deltas = corners - center
        deltas[:,0] *= scale_x
        deltas[:,1] *= scale_y
        return deltas + center

    def detect_corners_and_compute(self, corners_top=None, corners_bottom=None):
        # top half
        print("Annotate TOP half: TL, TR, mid-left, mid-right")
        if corners_top is None:
            corners_top = self.annotate_half("Click 4 corners for TOP half (TL, TR, mid-left, mid-right)")
        # corners_top = self.expand_corners_anisotropic(corners_top, scale_x=1.0, scale_y=1.4)
        self.H_top, _ = cv2.findHomography(corners_top, self.pts_dst_top)

        # bottom half
        print("Annotate BOTTOM half: mid-left, mid-right, BL, BR")
        if corners_bottom is None:
            corners_bottom = self.annotate_half("Click 4 corners for BOTTOM half (mid-left, mid-right, BL, BR)")
        self.H_bottom, _ = cv2.findHomography(corners_bottom, self.pts_dst_bottom)

        if self.H_top is None or self.H_bottom is None:
            raise RuntimeError("Failed to compute one or both homographies.")
        print("Homographies computed for both halves.")

    def transform_ball(self, ball_x, ball_y):
        """
        Transform a ball point by choosing top or bottom H depending on y position.
        """
        if self.H_top is None or self.H_bottom is None:
            raise RuntimeError("You must annotate and compute homographies first.")
        
        # simple heuristic: if closer to top half or bottom half
        img_height = cv2.imread(self.image_path).shape[0]
        if ball_y < img_height // 2:
            H = self.H_top
        else:
            H = self.H_bottom
        
        pt = np.array([[[ball_x, ball_y]]], dtype=np.float32)
        top_down_pt = cv2.perspectiveTransform(pt, H)
        return top_down_pt[0][0]

    def warp_table(self, save_path=None):
        """
        Warps both halves and stitches them together.
        """
        img_bgr = cv2.imread(self.image_path)
        warped_top = cv2.warpPerspective(img_bgr, self.H_top,
                                         (self.topdown_width, self.topdown_height))
        warped_bottom = cv2.warpPerspective(img_bgr, self.H_bottom,
                                            (self.topdown_width, self.topdown_height))

        # blend both halves
        mask = np.zeros_like(warped_top)
        mask[:self.topdown_height//2, :, :] = warped_top[:self.topdown_height//2, :, :]
        mask[self.topdown_height//2:, :, :] = warped_bottom[self.topdown_height//2:, :, :]

        if save_path:
            cv2.imwrite(save_path, mask)
            print(f"Warped table saved to {save_path}")

        return mask
