import os
import sys
import gc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(BASE_DIR, "packages")
if os.path.isdir(PKG_DIR):
    sys.path.insert(0, PKG_DIR)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm


IMG_SIZE = 224
DEVICE = torch.device("cpu")

MODEL_PATH = os.path.join(BASE_DIR, "results", "weather_best.pth")

CLASS_NAMES = ["cloudy", "rainy", "snowy", "sunny"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_model = None
_model_name = None


def _safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _load_model():
    global _model, _model_name

    if _model is not None:
        return _model

    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found: results/weather_best.pth")

    checkpoint = _safe_torch_load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        model_name = checkpoint.get("model_name", "efficientnet_b0")
    else:
        state_dict = checkpoint
        model_name = "efficientnet_b0"

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=4,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model.to(DEVICE)

    _model = model
    _model_name = model_name

    return _model


def _cv2_to_rgb(x):
    if x is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    if not isinstance(x, np.ndarray):
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    if x.ndim == 2:
        return cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)

    if x.ndim == 3 and x.shape[2] == 4:
        return cv2.cvtColor(x, cv2.COLOR_BGRA2RGB)

    if x.ndim == 3 and x.shape[2] == 3:
        return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

    return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)


def _preprocess_rgb(img_rgb):
    img_rgb = cv2.resize(
        img_rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR,
    )

    img = img_rgb.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))

    tensor = torch.from_numpy(img).unsqueeze(0)
    tensor = tensor.to(DEVICE)
    return tensor


def predict(X):
    """
    Platform inference interface.

    Args:
        X: image loaded by cv2.imread, BGR ndarray.

    Returns:
        One of: cloudy / rainy / snowy / sunny
    """
    model = _load_model()
    img_rgb = _cv2_to_rgb(X)
    tensor = _preprocess_rgb(img_rgb)

    with torch.inference_mode():
        logits = model(tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())

    pred = CLASS_NAMES[pred_idx]

    del tensor, logits
    return pred


if __name__ == "__main__":
    model = _load_model()
    print("model_name =", _model_name)
    print("class_names =", CLASS_NAMES)

    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    print("predict(dummy) =", predict(dummy))

    gc.collect()