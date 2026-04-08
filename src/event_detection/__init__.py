from .E2E import E2EModel
from .ASTRM import ASTRME2EModel
from .VIT import VITModel
from .utils import load_model_compiled, nms_on_dict

model_map = {
    'E2E': E2EModel,
    'ASTRM': ASTRME2EModel,
    'VIT': VITModel,
}


def get_model(model_name, *args, **kwargs):
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not recognized. Available models: {list(model_map.keys())}")
    return model_map[model_name](*args, **kwargs)





