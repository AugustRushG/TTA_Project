import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights
import torch
from .shift import make_temporal_shift 
from .modules import FCPrediction, GRUPrediction


class E2EModel(nn.Module):

    def __init__(self, model_config):
        super().__init__()
        self.num_classes = model_config['num_classes']
        clip_len = model_config['num_frames_clip']
        base_model = model_config['base_model']

        if base_model in ['regnety_002', 'regnety_008', 'convnext_tiny', 'convnext_small', 'convnext_base']:
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

        else:
            raise ValueError(f"Unsupported base model: {base_model}")
        
        temporal_shift_mode = model_config['temporal_shift_mode']
        if temporal_shift_mode is not None:
            # Add Temporal Shift Modules
            if temporal_shift_mode == 'hgsm' or temporal_shift_mode == 'hgsf' or temporal_shift_mode == 'hagsm' or temporal_shift_mode == 'agsm':
                module_config  = {'dilations': model_config['dilations'], 'num_heads': model_config['num_heads']}
            else :
                module_config = None
            make_temporal_shift(features, clip_len, mode=temporal_shift_mode, configs=module_config)
            self._require_clip_len = clip_len

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
            
    def extract_features(self, x):
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

        return self._pred_fine.extract_features(im_feat)
    
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
    import time
    model_config = {
        'num_classes': 8,
        'num_frames_clip': 100,
        'base_model': 'regnety_002',
        'temporal_shift_mode': 'hagsm', # None, tsm, gsm, gsf, hgsm, hgsf, hagsm, agsm
        'dilations': [1, 2, 3],
        'num_heads': 2,
        'gru_layers': 1,
    }
    model = E2EModel(model_config)
    # model.print_stats()
    # x = torch.randn(2, 8, 3, 224, 224)
    # t0 = time.time()
    # y = model(x)
    # prediction = model.predict(x, device='cpu')
    # print(prediction[0].shape, prediction[1].shape)
    # extracted_features = model.extract_features(x)
    # t1 = time.time()
    # print(y.shape, t1 - t0)
    # print(extracted_features.shape)

    # compute complexity
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model, (100, 3, 224, 224), as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
    gflops = 2 * macs / 1e9
    print('{:<30}  {:<8}'.format('Computational complexity: ', gflops))
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
