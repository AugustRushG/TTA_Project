import cv2
import numpy as np
import torch

class CenterCropResizeFrame:
    def __init__(self, size=(224, 224), crop_ratio=None):
        self.size = size
        self.crop_ratio = crop_ratio

    def __call__(self, frame):
        input_is_tensor = isinstance(frame, torch.Tensor)

        # Convert tensor -> numpy
        if input_is_tensor:
            if frame.ndim == 3:
                # CHW -> HWC
                if frame.shape[0] in [1, 3]:
                    frame_np = frame.permute(1, 2, 0).cpu().numpy()
                else:
                    frame_np = frame.cpu().numpy()
            else:
                frame_np = frame.cpu().numpy()
        else:
            frame_np = frame

        h, w = frame_np.shape[:2]
        new_w, new_h = self.size

        if self.crop_ratio is not None:
            ratio_w, ratio_h = self.crop_ratio
        else:
            ratio_w, ratio_h = new_w, new_h

        target_aspect = ratio_w / ratio_h
        frame_aspect = w / h

        if frame_aspect > target_aspect:
            crop_h = h
            crop_w = int(h * target_aspect)
        else:
            crop_w = w
            crop_h = int(w / target_aspect)

        left = (w - crop_w) // 2
        top = (h - crop_h) // 2

        cropped = frame_np[top:top + crop_h, left:left + crop_w]
        resized = cv2.resize(cropped, (new_w, new_h))

        # Convert back to tensor if input was tensor
        if input_is_tensor:
            if resized.ndim == 3:
                resized = torch.from_numpy(resized).permute(2, 0, 1)
            else:
                resized = torch.from_numpy(resized)

            resized = resized.to(frame.dtype)

        return resized
