from detectron2.data import build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils
import torch
import numpy as np
from utils import *
import copy

data_dir = '../../data/train'

def custom_train_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    file = dataset_dict['file_name'].split('/')
    image = load_RGBY_image(data_dir, file[-1], image_size=512)
    image = image[:, :, ::-1] #Flip to convert to BGR
    
    transform_list = [
                          T.RandomFlip(prob=0.2, horizontal=False, vertical=True),
                          T.RandomFlip(prob=0.2, horizontal=True, vertical=False),
                      ]
    
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict

def custom_val_mapper(dataset_dict):
    # Implement a mapper, similar to the default DatasetMapper, but with your own customizations
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    file = dataset_dict['file_name'].split('/')
    image = load_RGBY_image(data_dir, file[-1], image_size=512)
    image = image[:, :, ::-1] #Flip to convert to BGR
    
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
    
    instances = detection_utils.annotations_to_instances(dataset_dict['annotations'], image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

    return dataset_dict