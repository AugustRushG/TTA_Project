import matplotlib.pyplot as plt
import cv2
import numpy as np

class TableDetector:
    def __init__(self, image_paths, topdown_width=334, topdown_height=600):
        """
        image_paths: list of image paths
        """
        self.image_paths = image_paths
        self.topdown_width = topdown_width
        self.topdown_height = topdown_height
        self.pts_dst = np.array([
            [0, 0],
            [topdown_width-1, 0],
            [topdown_width-1, topdown_height-1],
            [0, topdown_height-1]
        ], dtype=np.float32)
        self.H = None

    def onclick(self, event):
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata), int(event.ydata)
            self.corners.append((x, y))
            print(f"Clicked: ({x}, {y})")

            self.ax.plot(x, y, 'ro')
            self.fig.canvas.draw()

            if len(self.corners) == 4:
                plt.close(self.fig)

    def detect_average_corners(self):
        all_corners = []

        for img_path in self.image_paths:
            self.corners = []  # reset for each image

            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            self.fig, self.ax = plt.subplots()
            self.ax.imshow(img_rgb)
            self.ax.set_title("Click 4 table corners")
            self.fig.canvas.mpl_connect('button_press_event', self.onclick)

            plt.show()

            if len(self.corners) != 4:
                raise ValueError(f"Expected 4 corners, got {len(self.corners)} on {img_path}")

            ordered = self.order_corners(self.corners)
            all_corners.append(ordered)

        # average corners
        all_corners = np.array(all_corners)  # shape: (N,4,2)
        avg_corners = np.mean(all_corners, axis=0)
        avg_corners = self.expand_corners(avg_corners, scale=1.2)

        # based on the average corners, expand the corners a bit to help with homography


        self.pts_src = avg_corners.astype(np.float32)
        self.compute_homography()
        return avg_corners
    
    def expand_corners(self, corners, scale=1.05):
        """
        Expand the corners outward from the center by a given scale.

        Args:
            corners (np.ndarray): shape (4,2), ordered TL, TR, BR, BL
            scale (float): scaling factor (>1 to expand, <1 to shrink)

        Returns:
            np.ndarray: expanded corners, shape (4,2)
        """
        center = corners.mean(axis=0, keepdims=True)  # shape: (1,2)
        expanded = (corners - center) * scale + center
        return expanded

    def compute_homography(self):
        self.H, _ = cv2.findHomography(self.pts_src, self.pts_dst)
        if self.H is None:
            raise RuntimeError("Failed to compute homography.")
        print("Homography matrix computed.")

    def transform_ball(self, ball_x, ball_y):
        if self.H is None:
            raise RuntimeError("You must detect corners and compute homography first.")
        pt = np.array([[[ball_x, ball_y]]], dtype=np.float32)
        top_down_pt = cv2.perspectiveTransform(pt, self.H)
        return top_down_pt[0][0]  # (x, y)

    def warp_table(self, image_path=None, save_path=None):
        if image_path is None:
            image_path = self.image_paths[0]  # default to first image

        img_bgr = cv2.imread(image_path)
        warped = cv2.warpPerspective(
            img_bgr, self.H,
            (self.topdown_width, self.topdown_height)
        )
        if save_path:
            cv2.imwrite(save_path, warped)
        return warped

    @staticmethod
    def order_corners(corners):
        pts = np.array(corners, dtype=np.float32)

        s = pts.sum(axis=1)        # x + y
        diff = np.diff(pts, axis=1)[:, 0]  # x - y

        ordered = np.zeros((4, 2), dtype=np.float32)
        ordered[0] = pts[np.argmin(s)]     # TL
        ordered[2] = pts[np.argmax(s)]     # BR
        ordered[1] = pts[np.argmin(diff)]  # TR
        ordered[3] = pts[np.argmax(diff)]  # BL

        return ordered
