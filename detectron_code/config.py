from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
import os

def get_config(train_metadata):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("hpa_train",)
    cfg.DATASETS.TEST = ("hpa_val",)
    cfg.DATALOADER.NUM_WORKERS = 10
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 32
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.MAX_ITER = 30000 #adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = ()

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(train_metadata.thing_classes)
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.OUTPUT_DIR = './output/all_data'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    
    return cfg
