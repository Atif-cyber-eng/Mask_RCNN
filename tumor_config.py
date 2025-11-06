from mrcnn.config import Config

class TumorConfig(Config):
    NAME = "tumor"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # background + tumor
    DETECTION_MIN_CONFIDENCE = 0.6
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
