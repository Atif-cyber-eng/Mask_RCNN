import os
from mrcnn import model as modellib
from tumor_config import TumorConfig
import numpy as np

class InferenceConfig(TumorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_model(weights_path):
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
    model.load_weights(weights_path, by_name=True)
    return model

def get_display_masks(mask):
    """
    Convert a boolean mask into an RGB array suitable for matplotlib.imshow overlay.
    Mask is a 2D boolean array. We return an (H,W,3) array with mask in red channel.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    # red overlay for mask
    rgb[:, :, 0] = mask.astype(float)
    return rgb
