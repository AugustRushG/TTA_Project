
import torch
from .model import OwnE2EModel
from utils.io import load_json

def load_check(model, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    inner = model._model
    incompat = inner.load_state_dict(state_dict, strict=False)

    all_model_keys = set(inner.state_dict().keys())
    loaded_keys    = all_model_keys - set(incompat.missing_keys)
    print(f"✅ Loaded {len(loaded_keys)}/{len(all_model_keys)} params")
    print("Missing keys:")
    print("\n".join(incompat.missing_keys))
    print("Unexpected keys in checkpoint:")
    print("\n".join(incompat.unexpected_keys))

    return model

if __name__ == '__main__':
    config_path = 'event_detection/model_configs/e2e_200.json'
    checkpoint_path = 'event_detection/checkpoints/E2E_TTA.pt'
    device = 'cpu'

    model_config = load_json(config_path)

    model = OwnE2EModel(model_config, device)
    load_check(model, checkpoint_path)
