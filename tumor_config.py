# tumor_config.py
from mrcnn.config import Config

class TumorConfig(Config):
    NAME = "tumor"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + tumor
    DETECTION_MIN_CONFIDENCE = 0.6
