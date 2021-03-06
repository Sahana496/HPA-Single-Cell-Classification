from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch

def get_config():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("hpa_train_final",)
    cfg.DATASETS.TEST = ("hpa_val_final",)
    cfg.DATALOADER.NUM_WORKERS = 10
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.SOLVER.IMS_PER_BATCH = 32
    cfg.SOLVER.BASE_LR = 0.000025
    cfg.SOLVER.WARMUP_ITERS = 5000
    cfg.SOLVER.MAX_ITER = 30000 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = ()
    cfg.MODEL.ROI_BOX_HEAD.FOCAL_ALPHA = 0.25
    cfg.MODEL.ROI_BOX_HEAD.FOCAL_GAMMA = 2.0

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 19
    cfg.MODEL.ROI_HEADS.NAME = 'FocalLossROIHeads'
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.OUTPUT_DIR = './output/focal_25_alpha'
    
    return cfg