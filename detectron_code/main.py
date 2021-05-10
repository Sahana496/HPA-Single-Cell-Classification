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
register_coco_instances("hpa_train", {}, "../data/train_val_single_coco.json", data_dir)
register_coco_instances("hpa_test", {}, "../data/test_coco.json", data_dir)
classes = ['Nucleoplasm', 'Nuclear membrane', 'Nucleoli', 'Nucleoli fibrillar center',
            'Nuclear speckles', 'Nuclear bodies', 'Endoplasmic reticulum', 'Golgi apparatus', 'Intermediate filaments','Actin filaments', 'Microtubules', 'Mitotic spindle', 'Centrosome', 'Plasma membrane', 'Mitochondria',
            'Aggresome', 'Cytosol', 'Vesicles and punctate cytosolic patterns', 'Negative']

train_metadata = MetadataCatalog.get("hpa_train").set(thing_classes=classes)

val_metadata = MetadataCatalog.get("hpa_test").set(thing_classes=classes)

cfg = get_config(train_metadata)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()