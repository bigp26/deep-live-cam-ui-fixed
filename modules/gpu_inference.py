import torch
from torch.cuda.amp import autocast

torch.backends.cudnn.benchmark = True

_model_cache = None


def load_model(model_path: str):
    """
    Load model once and cache it.
    """
    global _model_cache

    if _model_cache is None:
        model = torch.load(model_path, map_location="cuda")
        model = model.cuda().eval()

        # enable pytorch compiler
        try:
            model = torch.compile(model)
        except Exception:
            pass

        _model_cache = model

    return _model_cache


def run_inference(model, input_tensor: torch.Tensor):
    """
    GPU optimized inference
    """
    with torch.no_grad():
        with autocast():
            return model(input_tensor.half())


def warmup(model):
    """
    GPU warmup to avoid first-frame latency.
    """
    dummy = torch.randn(1, 3, 256, 256, device="cuda").half()

    for _ in range(10):
        with autocast():
            model(dummy)
