import timm
from timm.models.regnet import Bottleneck
import torch.nn as nn
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
import torch
import torch.nn.functional as F

from .modules import FCPrediction, GRUPrediction

class LocalSpatialBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # We pool over channels, so input to conv is 2 maps: [max_pool, avg_pool]
        # Shape before conv: (B*T, 2, H, W)
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B*T, C, H, W)
        B_, C, H, W = x.shape

        # Channel-wise pooling -> (B*T, 1, H, W) each
        max_pool, _ = torch.max(x, dim=1, keepdim=True)   # (B*T, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)     # (B*T, 1, H, W)

        # Concatenate along channel -> (B*T, 2, H, W)
        pooled = torch.cat([max_pool, avg_pool], dim=1)

        # 7x7 conv over spatial dims -> (B*T, 1, H, W)
        mask_spatial = self.conv(pooled)
        mask_spatial = self.sigmoid(mask_spatial)         # (B*T, 1, H, W)

        return mask_spatial

class LocalTemporalBlock(nn.Module):
    """
    Implements the Local Temporal block Ft(x) based on the formula:
    Ft(x) = σ( f_conv_1x1x1( BN( ReLU( f_conv_3x1x1(x) ) ) ) )
    
    This block expects an input of shape (B*T, C, H, W) and 
    unfolds/refolds the temporal dimension internally.
    """
    def __init__(self, num_channels, num_frames):
        """
        Args:
            num_channels (int): Number of channels (C) in the input feature map.
            num_frames (int): Number of frames (T) in the input feature map.
        """
        super(LocalTemporalBlock, self).__init__()
        
        self.num_frames = num_frames
        
        # f_conv_3x1x1(x)
        # This is a 3D convolution with kernel (T=3, H=1, W=1)
        # We use padding=(1, 0, 0) to preserve the temporal (T) dimension
        self.conv1 = nn.Conv3d(
            in_channels=num_channels, 
            out_channels=num_channels, 
            kernel_size=(3, 1, 1), # (T, H, W)
            padding=(1, 0, 0)      # (T, H, W)
        )
        
        # ReLU
        self.relu = nn.ReLU(inplace=True)
        
        # BN (BatchNorm3d)
        self.bn = nn.BatchNorm3d(num_channels)
        
        # f_conv_1x1x1(x)
        # This is a 3D convolution with kernel (T=1, H=1, W=1)
        self.conv2 = nn.Conv3d(
            in_channels=num_channels, 
            out_channels=num_channels, 
            kernel_size=(1, 1, 1), 
            padding=0
        )
        
        # σ (Sigmoid)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map with shape (B*T, C, H, W)
        Returns:
            torch.Tensor: The temporal mask Ft(x) with shape (B*T, C, H, W)
        """
        # x shape: (B*T, C, H, W)
        B_T, C, H, W = x.shape
        
        # --- 1. Unfold Time Dimension ---
        B = B_T // self.num_frames
        x_unfolded = x.view(B, self.num_frames, C, H, W)
        
        # Permute to (B, C, T, H, W) for Conv3d
        x_5d = x_unfolded.permute(0, 2, 1, 3, 4).contiguous()

        # --- 2. Apply Formula ---
        # f_conv_3x1x1(x)
        x_out = self.conv1(x_5d)
        
        # BN( ReLU(...) )
        x_out = self.bn(self.relu(x_out))
        
        # f_conv_1x1x1(...)
        x_out = self.conv2(x_out)
        
        # σ(...)
        mask_5d = self.sigmoid(x_out) # Shape: (B, C, T, H, W)
        
        # --- 3. Refold Time Dimension ---
        # Permute back to (B, T, C, H, W)
        mask_unfolded = mask_5d.permute(0, 2, 1, 3, 4).contiguous()
        
        # Reshape back to (B*T, C, H, W)
        mask_folded = mask_unfolded.view(B_T, C, H, W)
        
        return mask_folded
    
class GlobalTemporalBlock(nn.Module):
    """
    Implements the Global Temporal block Gt(x) for a "folded" batch input.
    
    This version returns ONLY the temporal mask Gt(x), expanded to
    the full (B*T, C, H, W) shape for compatibility with the formula.
    
    Gt(x) = σ( fFC( fFC( fGAP(x) ) ) )
    """
    def __init__(self, num_channels, num_frames, reduction_ratio=16):
        """
        Args:
            num_channels (int): Number of channels (C) in the input feature map.
            num_frames (int): Number of frames (T) in the input feature map.
            reduction_ratio (int): Reduction ratio for the intermediate FC layer.
        """
        super(GlobalTemporalBlock, self).__init__()
        
        self.num_channels = num_channels
        self.num_frames = num_frames
        
        reduced_dim = max(1, num_frames // reduction_ratio)
        
        self.fc1 = nn.Linear(in_features=num_frames, out_features=reduced_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_features=reduced_dim, out_features=num_frames)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input feature map with shape (B*T, C, H, W)
        Returns:
            torch.Tensor: The global temporal mask Gt(x), 
                          shape (B*T, C, H, W)
        """
        # Get original shape dimensions
        BT, C, H, W = x.shape
        
        # --- 1. Unfold Time Dimension ---
        if BT % self.num_frames != 0:
            raise ValueError(
                f"Input batch size ({BT}) is not divisible by "
                f"num_frames ({self.num_frames})."
            )
        
        B = BT // self.num_frames

        
        # Reshape to (B, T, C, H, W)
        x_unfolded = x.view(B, self.num_frames, C, H, W)
        
        # Permute to (B, C, T, H, W)
        x_permuted = x_unfolded.permute(0, 2, 1, 3, 4).contiguous()

        # --- 2. Apply Global Temporal Logic ---
        
        # fGAP(x) - Global Average Pooling over spatial (H, W)
        # Input: (B, C, T, H, W) -> Output: (B, C, T)
        s = x_permuted.mean(dim=[-1, -2]) 
        
        # fFC( fFC( ... ) )
        # Input: (B, C, T)
        s = self.fc1(s)     # Output: (B, C, reduced_dim)
        s = self.relu(s)   # Output: (B, C, reduced_dim)
        s = self.fc2(s)     # Output: (B, C, T)
        
        # σ(...) - Sigmoid activation
        # Output: (B, C, T)
        temporal_weights_compact = self.sigmoid(s)
        
        # --- 3. Expand and Refold Mask ---
        
        # Reshape weights to (B, C, T, 1, 1) for broadcasting
        temporal_weights = temporal_weights_compact.unsqueeze(-1).unsqueeze(-1)
        
        # Expand the weights to the full spatial size (H, W)
        # Shape: (B, C, T, H, W)
        # .expand() is memory-efficient; it just repeats the existing values
        mask_5d = temporal_weights.expand(-1, -1, -1, H, W)
        
        # Permute back to (B, T, C, H, W)
        mask_unfolded = mask_5d.permute(0, 2, 1, 3, 4).contiguous()
        
        # Reshape back to (B*T, C, H, W)
        mask_folded = mask_unfolded.view(BT, C, H, W)
        
        return mask_folded

        
    
class ASTRMBlock(nn.Module):
    def __init__(self, input_dim, num_frames=100, reduction_ratio=4):
        super(ASTRMBlock, self).__init__()
        self.input_dim = input_dim
        self.local_spatial = LocalSpatialBlock(input_dim)
        self.local_temporal = LocalTemporalBlock(input_dim, num_frames=num_frames)
        self.global_temporal = GlobalTemporalBlock(input_dim, num_frames=num_frames, reduction_ratio=reduction_ratio)
    def forward(self, x):
        # Fs(x)
        # Assumes self.local_spatial(x) returns the mask Fs(x)
        fs_mask = self.local_spatial(x) 

        # Ft(x)
        # Assumes self.local_temporal(x) returns the mask Ft(x)
        lt_mask = self.local_temporal(x)

        # Gt(x)
        # Assumes self.global_temporal(x) returns the mask Gt(x)
        gt_mask = self.global_temporal(x)

        # --- 2. Apply the formula sequentially ---
        # We assume ⊙ and * both mean element-wise multiplication (*) in PyTorch

        # Step 1: y = x ⊙ (1 + Fs(x))
        y = x * (1 + fs_mask)

        # Step 2: z = y ⊙ (1 + Ft(x))
        # The mask is lt_mask (computed from x), not a new mask computed from y
        z = y * (1 + lt_mask)

        # Step 3: Ψ(x) = z * Gt(x)
        # Note: The formula does NOT have (1 + Gt(x)), just Gt(x)
        output = z * gt_mask
        # output in shape (B*T, C, H, W)
        return output


def insert_astrm_after_conv1(net, num_frames, reduction_ratio=16):
    """
    Recursively finds all 'Bottleneck' blocks in a timm RegNet
    and inserts an ASTRMBlock *after* conv1 (ConvNormAct).

    Args:
        net (nn.Module): The timm RegNetY model.
        num_frames (int): The 'T' dimension for ASTRM.
        reduction_ratio (int): Reduction ratio for ASTRM.

    Returns:
        nn.Module: The modified network.
    """
    for name, module in net.named_children():

        if isinstance(module, Bottleneck):
            # original conv1: ConvNormAct
            orig_conv1 = module.conv1

            # Get the output channels of conv1
            # (ConvNormAct has .conv which is a Conv2d)
            num_channels = orig_conv1.conv.out_channels

            # Create ASTRM block
            astrm = ASTRMBlock(
                input_dim=num_channels,
                num_frames=num_frames,
                reduction_ratio=reduction_ratio,
            )

            # Wrap conv1 + ASTRM in a Sequential
            # so the Bottleneck forward still calls `self.conv1(x)`
            # but internally it now does conv1 -> ASTRM
            module.conv1 = nn.Sequential(
                orig_conv1,
                astrm
            )

            print(f"Inserted ASTRM after {name}.conv1 (C={num_channels})")

            # (Optional) if you ALSO want to remove SE:
            # module.se = nn.Identity()

        # recurse into children
        elif list(module.children()):
            insert_astrm_after_conv1(module, num_frames, reduction_ratio)

    return net


class ASTRME2EModel(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.num_classes = model_config['num_classes']
        clip_len = model_config['num_frames_clip']
        base_model = model_config['base_model']
        if base_model in ['regnety_002', 'regnety_008', 'convnext_tiny']:
            features = timm.create_model(base_model, pretrained=True)
            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()

        elif base_model in ['resnet18', 'resnet50']:
            if base_model == 'resnet18':
                features = tv_models.resnet18(weights=ResNet18_Weights.DEFAULT)
            elif base_model == 'resnet50':
                features = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT)
            elif base_model == 'resnet101':
                features = tv_models.resnet101(weights=ResNet101_Weights.DEFAULT)
            feat_dim = features.fc.in_features
            features.fc = nn.Identity()
        
        elif base_model in ['convnext_tiny', 'convnext_small', 'convnext_base']:
            if base_model == 'convnext_tiny':
                features = timm.create_model('convnext_tiny', pretrained=True)
            elif base_model == 'convnext_small':
                features = timm.create_model('convnext_small', pretrained=True)
            elif base_model == 'convnext_base':
                features = timm.create_model('convnext_base', pretrained=True)

            feat_dim = features.head.fc.in_features
            features.head.fc = nn.Identity()

        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        # replace_astrm_blocks(features, num_frames=clip_len, reduction_ratio=4)
        features = insert_astrm_after_conv1(features, num_frames=clip_len, reduction_ratio=4)


        self._features = features
        self._feat_dim = feat_dim
        hidden_dim = feat_dim
        gru_layers = model_config['gru_layers']
        if gru_layers == 0:
            self._pred_fine = FCPrediction(feat_dim, self.num_classes)
        else:
            self._pred_fine = GRUPrediction(
                feat_dim, self.num_classes, hidden_dim,
                num_layers=gru_layers)
            
    
    def forward(self, x):
        batch_size, true_clip_len, channels, height, width = x.shape

        clip_len = true_clip_len

        im_feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._feat_dim)

        if true_clip_len != clip_len:
            # Undo padding
            im_feat = im_feat[:, :true_clip_len, :]

        return self._pred_fine(im_feat)


    def print_stats(self):
        print('Model params:',
            sum(p.numel() for p in self.parameters()))
        print('  CNN features:',
            sum(p.numel() for p in self._features.parameters()))
        print('  Temporal:',
            sum(p.numel() for p in self._pred_fine.parameters()))

    def predict(self, seq, device):
        device_type = device.type if isinstance(device, torch.device) else str(device)
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != device:
            seq = seq.to(device)

        self.eval()
        with torch.no_grad():
            with torch.autocast(device_type=device_type):
                pred = self.forward(seq) # [B, T, C]
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, axis=2)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().float().numpy(), pred.cpu().float().numpy()



if __name__ == '__main__':
    num_frames = 100
    model_config = {
        "base_model" : "regnety_002",
        "temporal_shift_mode": "gsm",
        "num_frames_clip" : 100,
        "num_classes" : 7,
        "cbam": False,
        "gru_layers": 1
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ASTRME2EModel(model_config).to(device)

    dummy_input = torch.randn([8, num_frames, 3, 224, 224])
    output = model(dummy_input)
    print(output.shape)
    # prediction_result = model.predict(dummy_input, device)
    # print(prediction_result[0].shape)

    # compute complexity
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (num_frames, 3, 224, 224), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
    gflops = 2 * macs / 1e9
    params_millions = params / 1e6
    print('{:<30}  {:<8}'.format('Gflops Computational complexity: ', gflops))
    print('{:<30}  {:<8}'.format('Macs Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters in millions: ', params_millions))
