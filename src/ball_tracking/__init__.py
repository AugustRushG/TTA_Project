from .TOTNet import TOTNet
from .TrackNetV4 import TrackNetV4
from .TOTNet_OF import TOTNetOF
from .utils import load_pretrained_model as load_ball_tracking_model, extract_coords2d, extract_coords
from .transform import CenterCropResizeFrame
# from .wasb import build_wasb
import torch

class BallTrackingModel:
    def __init__(self, num_frames, image_size, model_choice, totnet_channels=64):
        self.num_frames = num_frames
        self.image_size = image_size
        self.model_choice = model_choice
        self.totnet_channels = totnet_channels

    def load_model(self):
        if self.model_choice == 'tracknetv4':
            print("Building TrackNetV4 model...")
            model = TrackNetV4(in_channels=self.num_frames*3, out_channels=1)
        elif self.model_choice == 'TOTNet':
            print("Building Motion Light model...")
            model = TOTNet(input_shape=self.image_size, spatial_channels=self.totnet_channels, num_frames=self.num_frames)
        elif self.model_choice == 'TOTNet_OF':
            print("Building Motion Light Optical Flow model...")
            model = TOTNetOF(input_shape=self.image_size, spatial_channels=self.totnet_channels, num_frames=self.num_frames)
        else:
            raise ValueError(f"Unknown model choice: {self.configs.model_choice}")

        return model
