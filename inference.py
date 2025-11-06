# inference.py
import os
from mrcnn import model as modellib
from tumor_config import TumorConfig

class InferenceConfig(TumorConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def load_model(weights_path):
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=os.getcwd())
    model.load_weights(weights_path, by_name=True)
    return model
