from .TOTNet import build_motion_model_light as build_ball_tracking_model
from .TOTNet_OF import build_motion_model_light_opticalflow
from .utils import load_pretrained_model as load_ball_tracking_model, extract_coords2d, extract_coords
# from .wasb import build_wasb
import torch

class BallTrackingModel:
    def __init__(self, model_choice, model_args, checkpoint_path):
        self.model_choice = model_choice
        if model_choice == 'wasb':
            self.model = build_wasb(model_args)
        elif model_choice == 'TOTNet_OF':
            self.model = build_motion_model_light_opticalflow(model_args)
        elif model_choice == 'TOTNet':
            self.model = build_ball_tracking_model(model_args)
        load_ball_tracking_model(self.model, checkpoint_path, device=model_args.device)
        self.model.eval()

    def predict(self, frames):
        with torch.no_grad():
            if self.model_choice in ['TOTNet', 'TOTNet_OF']:
                outputs = self.model(frames)
            else:
                B, N, C, H, W = frames.shape
                frames = frames.permute(0, 2, 1, 3, 4).contiguous()  # Shape: [B, C, N, H, W]
                # Reshape to combine frames into the channel dimension
                frames = frames.view(B, N * C, H, W)  # Shape: [B, N*C, H, W]
                outputs = self.model(frames)
        return outputs

    def extract_coordinates(self, outputs):
        return extract_coords(outputs)
    
    def extract_coordinates_2d(self, outputs, H, W):
        return extract_coords2d(outputs, H, W)