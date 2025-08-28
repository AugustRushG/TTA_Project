import torch
import os

def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters


def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.batch_size = int(configs.batch_size / configs.ngpus_per_node)
            configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs.gpu_idx is not None:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model

def post_process(coords_logits):
    # Assuming coords_logits is [B, 2] where each is a continuous logit, 
    # and we treat each dimension independently.

    x_coord_logits = coords_logits[:, 0]  # Shape: [B]
    y_coord_logits = coords_logits[:, 1]  # Shape: [B]

    # If you have only 2 values per sample, we need to expand them with classes
    # for each axis to use softmax correctly, otherwise `coords_logits` should be [B, W, H] or similar.

    # Use softmax for probabilistic interpretation
    x_coord_probs = torch.softmax(x_coord_logits, dim=-1)
    y_coord_probs = torch.softmax(y_coord_logits, dim=-1)

    # Use argmax to find the class (coordinate)
    x_coord_pred = torch.argmax(x_coord_probs, dim=-1)
    y_coord_pred = torch.argmax(y_coord_probs, dim=-1)

    # Stack the predictions to get [B, 2]
    return torch.stack([x_coord_pred, y_coord_pred], dim=1)



def load_pretrained_model(model, pretrained_path, device):
    """Load weights from the pretrained model"""
    assert os.path.isfile(pretrained_path), f"=> no checkpoint found at '{pretrained_path}'"
    print("=> Loading pretrained weights from '{}'".format(pretrained_path))
    try:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        pretrained_dict = checkpoint['state_dict']

        if hasattr(model, 'module'):
            model_state_dict = model.module.state_dict()
        else:
            model_state_dict = model.state_dict()

        # Filter only matching keys
        matched_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
        unmatched_keys = [k for k in pretrained_dict if k not in model_state_dict]

        model_state_dict.update(matched_dict)

        if hasattr(model, 'module'):
            model.module.load_state_dict(model_state_dict)
        else:
            model.load_state_dict(model_state_dict)

        print(f"=> Loaded pretrained weights successfully from '{pretrained_path}'")
        print(f"=> {len(matched_dict)} layers loaded. {len(unmatched_keys)} unmatched.")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint from '{pretrained_path}': {e}")
    
    return model



def extract_coords(pred_heatmap):
    """_summary_

    Args:
        pred_heatmap : tuple of tensors (pred_x_logits, pred_y_logits)
        - pred_x_logits: Tensor of shape [B, W] with predicted logits for x-axis
        - pred_y_logits: Tensor of shape [B, H] with predicted logits for y-axis
    Return:
        out (tensor) : Tensor in shape [B,2] which represents coords for each 
    """
    pred_x_logits, pred_y_logits = pred_heatmap

    # Predicted coordinates are extracted by taking the argmax over logits
    x_pred_indices = torch.argmax(pred_x_logits, dim=1)  # [B]
    y_pred_indices = torch.argmax(pred_y_logits, dim=1)  # [B]

    # Convert indices to float for calculations
    x_pred = x_pred_indices.float()
    y_pred = y_pred_indices.float()

    # Stack the predicted x and y coordinates
    pred_coords = torch.stack([x_pred, y_pred], dim=1)  # [B, 2]

    return pred_coords


def extract_coords2d(pred_heatmap, H, W):
    """
    Extracts (x, y) coordinates from a predicted flattened 2D heatmap.

    Args:
        pred_heatmap (Tensor):
            Predicted heatmap of shape [B, H*W].
            Can be probabilities or logits.
        H (int): height of the heatmap.
        W (int): width of the heatmap.

    Returns:
        Tensor: [B, 2] with (x, y) coordinates for each batch item.
    """
    if pred_heatmap.dim() != 2 or pred_heatmap.size(1) != H * W:
        raise ValueError(f"Expected shape [B, {H*W}], got {list(pred_heatmap.shape)}")

    B = pred_heatmap.size(0)

    # Argmax to get the flat index of the highest-probability pixel
    flat_idx = pred_heatmap.argmax(dim=1)  # [B]

    # Convert flat index to (x, y)
    x_pred = (flat_idx % W).float()        # [B]
    y_pred = (flat_idx // W).float()       # [B]

    # Stack into [B, 2]
    pred_coords = torch.stack([x_pred, y_pred], dim=1)

    return pred_coords