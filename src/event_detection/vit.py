import torch
import torch.nn as nn
import torch.nn.functional as F

import timm  # <- NEW
from .shift import make_vit_shift
from .modules import FCPrediction, GRUPrediction, DINOHead
from contextlib import nullcontext

class VITModel(nn.Module):

    def __init__(self, model_config, semi_supervised=False, unsupervised=False, freeze_backbone=False):
        super().__init__()
        self.num_classes = model_config['num_classes']
        clip_len = model_config['num_frames_clip']
        self._require_clip_len = clip_len
        base_model = model_config['base_model']

        # ------------------------------------------------------------------
        # 1. Build timm ViT backbone
        # ------------------------------------------------------------------
        if base_model in ['vit16', 'vit_b_16', 'vit_base_patch16_224']:
            timm_name = 'vit_base_patch16_224'
        elif base_model in ['vit_small_16', 'vit_s_16']:
            timm_name = 'vit_small_patch16_224'
        elif base_model in ['vit_tiny_16', 'vit_t_16']:
            timm_name = 'vit_tiny_patch16_224'
        else:
            raise ValueError(f'Unknown base_model: {base_model}')

        # pretrained=True loads ImageNet (or similar) weights
        model = timm.create_model(timm_name, pretrained=True)

        # ------------------------------------------------------------------
        # 3. Strip classifier head and record feature dim
        # ------------------------------------------------------------------
        feat_dim = getattr(model, 'embed_dim',
                           getattr(model, 'num_features', None))
        if feat_dim is None:
            raise RuntimeError('Cannot infer feature dim from timm ViT model.')
        
        # ------------------------------------------------------------------
        # 2. Optional temporal shift (should still work if make_vit_shift
        #    expects .blocks etc., as timm ViT structure is similar)
        # ------------------------------------------------------------------
        if model_config.get('temporal_shift_mode', None) is not None:
            if 'shift_distances' in model_config:
                shift_distances = model_config['shift_distances']
            else:
                shift_distances = None
            make_vit_shift(model, clip_len,shift_mode=model_config['temporal_shift_mode'], feature_dim=feat_dim, shift_distances=shift_distances)


        # timm uses .head as classifier; replace with Identity
        model.head = nn.Identity()

        self._features = model
        self._feat_dim = feat_dim

        if freeze_backbone:
            print("Freezing ViT backbone...")
            for param in self._features.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------
        # 4. Temporal head (same as your original)
        # ------------------------------------------------------------------
        hidden_dim = feat_dim
        gru_layers = model_config['gru_layers']
        pred_dim = self.num_classes
        self.unsupervised = unsupervised
        self.semi_supervised = semi_supervised
        if self.unsupervised:
            # self._pred_fine = None
            self._pred_fine = DINOHead(
                feat_dim, 20000, nlayers=1, use_bn=True, norm_last_layer=True
            )
            # self._pred_fine = GRUPrediction(
            #     feat_dim, pred_dim, hidden_dim,
            #     num_layers=gru_layers
            # )
        elif self.semi_supervised:
            self._pred_fine = GRUPrediction(
                feat_dim, pred_dim, hidden_dim,
                num_layers=gru_layers
            )
        else:
            self._pred_fine = GRUPrediction(
                feat_dim, pred_dim, hidden_dim,
                num_layers=gru_layers
            )

    
    def __str__(self):
        lines = []
        lines.append(self.__class__.__name__ + "(")

        # Backbone
        lines.append(f"  Backbone: ViT ({self._features.__class__.__name__})")
        lines.append(f"  Feature dim: {self._feat_dim}")

        # Clip / temporal
        lines.append(f"  Required clip length: {self._require_clip_len}")

        # Training regime
        if self.unsupervised:
            regime = "unsupervised"
        elif self.semi_supervised:
            regime = "semi-supervised"
        else:
            regime = "supervised"
        lines.append(f"  Regime: {regime}")

        # Temporal head
        lines.append(f"  Temporal head: {self._pred_fine.__class__.__name__}")

        # Frozen backbone?
        frozen = not any(p.requires_grad for p in self._features.parameters())
        lines.append(f"  Backbone frozen: {frozen}")

        # Parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lines.append(f"  Params: {total_params:,} total / {trainable_params:,} trainable")

        lines.append(")")
        return "\n".join(lines)

    def extract_features(self, x):
        # x: [B, T, C, H, W]
        batch_size, true_clip_len, channels, height, width = x.shape

        clip_len = true_clip_len
        if self._require_clip_len > 0:
            # TSM module requires fixed clip len
            assert true_clip_len <= self._require_clip_len, \
                f'Expected <= {self._require_clip_len}, got {true_clip_len}'
            if true_clip_len < self._require_clip_len:
                x = F.pad(
                    x,
                    (0,) * 7 + (self._require_clip_len - true_clip_len,)
                )
                clip_len = self._require_clip_len

        # timm ViT expects [B', C, H, W]
        im_feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._feat_dim)

        if true_clip_len != clip_len:
            im_feat = im_feat[:, :true_clip_len, :]

        return im_feat

    

    def forward(self, x, return_features=False, return_pred=False):
        # x: [B, T, C, H, W]
        batch_size, true_clip_len, channels, height, width = x.shape

        clip_len = true_clip_len
        if self._require_clip_len > 0:
            assert true_clip_len <= self._require_clip_len, \
                f'Expected <= {self._require_clip_len}, got {true_clip_len}'
            if true_clip_len < self._require_clip_len:
                x = F.pad(x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                clip_len = self._require_clip_len

        im_feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._feat_dim)

        if true_clip_len != clip_len:
            im_feat = im_feat[:, :true_clip_len, :]

        # --- New behavior ---
        if return_features and not return_pred:
            return im_feat                  # ONLY features, head not called

        pred = self._pred_fine(im_feat)     # head called only when needed

        if return_features and return_pred:
            return im_feat, pred
        else:
            return pred
        
    def print_stats(self):
        print('Model params:',
              sum(p.numel() for p in self.parameters()))
        print('  ViT features:',
              sum(p.numel() for p in self._features.parameters()))
        print('  Temporal head:',
              sum(p.numel() for p in self._pred_fine.parameters()))

    def predict(self, seq, device, use_amp=True):
        device_type = device.type if isinstance(device, torch.device) else str(device)
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != device:
            seq = seq.to(device)

        enable_amp = use_amp and (device_type == 'cuda')
        autocast_ctx = torch.amp.autocast(device_type=device_type) if enable_amp else nullcontext()

        self.eval()
        with torch.no_grad():
            with autocast_ctx:
                pred = self.forward(seq)  # [B, T, C]
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, dim=2)
            pred_cls = torch.argmax(pred, dim=2)
            return pred_cls.cpu().float().numpy(), pred.cpu().float().numpy()


if __name__ == '__main__':
    import time
    model_config = {
        'num_classes': 8,
        'num_frames_clip': 100,
        'base_model': 'vit_tiny_16',   # or 'vit_small_16', 'vit_tiny_16', 'vit16'
        'gru_layers': 1,
        'temporal_shift_mode': 'multiscale_transformer_gsm',
        'shift_distances': [1,2,3]
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VITModel(model_config).to(device)
    print(model)
    exit()

    model.eval()
    model.print_stats()
    x = torch.randn(1, 100, 3, 224, 224).to(device)
    t0 = time.time()
    y = model(x)
    prediction = model.predict(x, device=device)
    print(prediction[0].shape, prediction[1].shape)
    extracted_features = model.extract_features(x)
    t1 = time.time()
    print(y.shape, t1 - t0)
    print(extracted_features.shape)

    #compute complexity
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (100, 3, 224, 224), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
    gflops = 2 * macs / 1e9
    print('{:<30}  {:<8}'.format('Computational complexity in GFlops: ', gflops))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
