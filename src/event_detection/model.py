import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights, ResNet50_Weights
import torch
from contextlib import nullcontext
from .common import step, BaseRGBModel
from .shift import make_temporal_shift as own_make_temporal_shift
from tqdm import tqdm
from .modules import FCPrediction, GRUPrediction


class OwnE2EModel(BaseRGBModel):

    class Impl(nn.Module):

        def __init__(self, model_config):
            super().__init__()
            num_classes = model_config['num_classes']
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
                feat_dim = features.fc.in_features
                features.fc = nn.Identity()
            
            elif base_model in ['convnext_tiny']:
                features = timm.create_model('convnext_tiny', pretrained=True)
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()

            else:
                raise ValueError(f"Unsupported base model: {base_model}")
            
            temporal_shift_mode = model_config['temporal_shift_mode']
            if temporal_shift_mode is not None:
                # Add Temporal Shift Modules
                if temporal_shift_mode == 'hgsm' or temporal_shift_mode == 'hgsf' or temporal_shift_mode == 'hagsm':
                   module_config  = {'dilations': model_config['dilations'], 'num_heads': model_config['num_heads']}
                else :
                   module_config = None
                own_make_temporal_shift(features, clip_len, mode=temporal_shift_mode, configs=module_config)
                self._require_clip_len = clip_len

            self._features = features
            self._feat_dim = feat_dim
            hidden_dim = feat_dim
            gru_layers = model_config['gru_layers']
            if gru_layers == 0:
                self._pred_fine = FCPrediction(feat_dim, num_classes)
            else:
                self._pred_fine = GRUPrediction(
                    feat_dim, num_classes, hidden_dim,
                    num_layers=gru_layers)
        
        def forward(self, x):
            batch_size, true_clip_len, channels, height, width = x.shape

            clip_len = true_clip_len
            if self._require_clip_len > 0:
                # TSM module requires clip len to be known
                assert true_clip_len <= self._require_clip_len, \
                    'Expected {}, got {}'.format(
                        self._require_clip_len, true_clip_len)
                if true_clip_len < self._require_clip_len:
                    x = F.pad(
                        x, (0,) * 7 + (self._require_clip_len - true_clip_len,))
                    clip_len = self._require_clip_len

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

    def __init__(self, model_config, device='cuda', multi_gpu=False):
        self.device = device
        self._multi_gpu = multi_gpu
        self._model = OwnE2EModel.Impl(model_config)
        self._model.print_stats()

        if multi_gpu:
            self._model = nn.DataParallel(self._model)

        self._model.to(device)
        self._num_classes = model_config['num_classes']
    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None,
              acc_grad_iter=1, fg_weight=5):
        if optimizer is None:
            self._model.eval()
        else:
            optimizer.zero_grad()
            self._model.train()

        ce_kwargs = {}
        if fg_weight != 1:
            ce_kwargs['weight'] = torch.FloatTensor(
                [1] + [fg_weight] * (self._num_classes - 1)).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = loader.dataset.load_frame_gpu(batch, self.device)
                label = batch['label'].to(self.device)

                # Depends on whether mixup is used
                label = label.flatten() if len(label.shape) == 2 \
                    else label.view(-1, label.shape[-1])

                with torch.autocast(device_type=self.device):
                    pred = self._model(frame)

                    loss = 0.
                    if len(pred.shape) == 3:
                        pred = pred.unsqueeze(0)

                    for i in range(pred.shape[0]):
                        loss += F.cross_entropy(
                            pred[i].reshape(-1, self._num_classes), label,
                            **ce_kwargs)

                if optimizer is not None:
                    step(optimizer, scaler, loss / acc_grad_iter,
                         lr_scheduler=lr_scheduler,
                         backward_only=(batch_idx + 1) % acc_grad_iter != 0)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq, use_amp=True):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)

        self._model.eval()
        with torch.no_grad():
            with torch.autocast(device_type=self.device) if use_amp else nullcontext():
                pred = self._model(seq)
            if isinstance(pred, tuple):
                pred = pred[0]
            if len(pred.shape) > 3:
                pred = pred[-1]
            pred = torch.softmax(pred, axis=2)
            pred_cls = torch.argmax(pred, axis=2)
            return pred_cls.cpu().numpy(), pred.cpu().numpy()



if __name__ == '__main__':
    import json
    model_config_path = 'event_detection/model_configs/e2e_200_hagsm.json'
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    device = 'cpu'
    model = OwnE2EModel(model_config, device)
    # print(model._model)
    # exit()
    dummy_input = torch.randn([8, 100, 3, 224, 224])
    output = model._model(dummy_input)
    print(output.shape)