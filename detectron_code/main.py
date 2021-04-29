from detectron2.data.datasets import register_coco_instances
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.structures import BoxMode

from detectron2.engine import DefaultPredictor

from detectron2.data import MetadataCatalog
from detectron2.data.catalog import DatasetCatalog
import logging
from trainer import *
from utils import *
from dataloader import *
from config import *

setup_logger()

data_dir = "../../data/train"
register_coco_instances("hpa_train", {}, "../data/hpa_train.json", data_dir)
register_coco_instances("hpa_val", {}, "../data/hpa_val.json", data_dir)

train_metadata = MetadataCatalog.get("hpa_train")
dataset_dicts_train = DatasetCatalog.get("hpa_train")

val_metadata = MetadataCatalog.get("hpa_val")
dataset_dicts_val = DatasetCatalog.get("hpa_val")

cfg = get_config(train_metadata)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()