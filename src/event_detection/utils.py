
import torch

def load_model_compiled(model, checkpoint_path, device):
    raw_state = torch.load(checkpoint_path, map_location=device)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        state_dict = raw_state["state_dict"]
    else:
        state_dict = raw_state
    fixed_state = {}

    # ---- Fix: Strip `_orig_mod.` prefix if present ----
    fixed_state = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]  # remove prefix
        else:
            new_k = k
        fixed_state[new_k] = v

    # ---- Load into your model ----
    missing, unexpected = model.load_state_dict(fixed_state, strict=False)
    print("\n=== State Dict Load Report ===")
    print("Missing keys in model:", missing)
    print("Unexpected keys in checkpoint:", unexpected)
    model.to(device)
    return model

