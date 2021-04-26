import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import functools
import operator
import torchvision
from os import listdir
from os.path import isfile, isdir, join
import shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pycocotools import mask
from skimage import measure


inpath = "../data/train/"  # the train folder download from kaggle
outpath = "../data/train_out/"  # the folder putting all nuclei image
mask_dir = "../data/hpa_cell_mask/"
images_name = listdir(inpath)
cocoformat = {"licenses":[], "info":[], "images":[], "annotations":[], "categories":[]}

classes = ['Nucleoplasm', 'Nuclear membrane', 'Nucleoli', 'Nucleoli fibrillar center',
            'Nuclear speckles', 'Nuclear bodies', 'Endoplasmic reticulum', 'Golgi apparatus', 'Intermediate filaments',
            'Actin filaments', 'Microtubules', 'Mitotic spindle', 'Centrosome', 'Plasma membrane', 'Mitochondria',
            'Aggresome', 'Cytosol', 'Vesicles and punctate cytosolic patterns', 'Negative']
    
def read_img(path, image_id, color, image_size=None):
    filename = f'{path}/{image_id}_{color}.png'
    assert os.path.exists(filename), f'not found {filename}'
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if image_size is not None:
        img = cv2.resize(img, (image_size, image_size))
    if img.max() > 255:
        img_max = img.max()
        img = (img/255).astype('uint8')
    return img

# image loader, using rgb only here
def load_RGBY_image(path, image_id, image_size=None):
    red = read_img(path, image_id, "red", image_size)
    green = read_img(path, image_id, "green", image_size)
    blue = read_img(path, image_id, "blue", image_size)

    stacked_images = np.transpose(np.array([red, green, blue]), (1,2,0))
    return stacked_images
    

def load_mask(image_id, image_size=None):
    """Generate instance masks for an image.
   Returns:
    masks: A bool array of shape [height, width, instance count] with
        one mask per instance.
    class_ids: a 1D array of class IDs of the instance masks.
    """
    # Read mask files from 
    masks = []
    class_ids = []
    cell_mask = np.load(f'{mask_dir}/{image_id}.npz')['arr_0']
    if image_size is not None:
        cell_mask = cv2.resize(cell_mask, (image_size, image_size), interpolation=cv2.INTER_NEAREST)

    #find number of cells in the image
    mask_ids = np.unique(cell_mask)

    #Remove background
    mask_ids = mask_ids[1:]

    #create binary mask for every cell in the image
    masks = cell_mask == mask_ids[:,None, None]

    # Return mask, and array of class IDs of each instance. Since we have
    # one class ID, we return an array of ones
    return masks, mask_ids


for i, c in enumerate(classes):
    cat = {"id": int(i)+1, 
           "name": c, 
           "supercategory": "hpa",
          }
    cocoformat["categories"].append(cat)
    
mask_id = 1
df = pd.read_csv('./data/train_split.csv')

image_ids = list(df['ID']) 


#image class
classes = df['Label'].values

for i, img_id in enumerate(image_ids):
    print("Processing Image {}:{} ".format(i, img_id))
    image = load_RGBY_image(inpath, img_id, image_size=512)
    masks, obj_ids = load_mask(img_id, image_size=512)
    
    
    im = {"id": int(i+1), 
          "width": int(image.shape[1]), 
          "height": int(image.shape[0]), 
          "file_name": img_id
         }
    
    cocoformat["images"].append(im)
    
    num_objs = len(obj_ids)
    for j in range(num_objs):
        ground_truth_binary_mask = masks[j]
        fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)

        encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
        ground_truth_area = mask.area(encoded_ground_truth)
        ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
        contours = measure.find_contours(ground_truth_binary_mask, 0.5)

        annotation = {
                "segmentation": [],
                "area": ground_truth_area.tolist(),
                "iscrowd": 0,
                "image_id": int(i+1),
                "bbox": ground_truth_bounding_box.tolist(),
                "category_id": int(classes[i])+1,
                "id": int(mask_id)
            }

        for contour in contours:
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            annotation["segmentation"].append(segmentation)
            
        mask_id = mask_id+1
        cocoformat["annotations"].append(annotation)

with open("./data/hpa_cocoformat_train.json", "w") as f:
    json.dump(cocoformat, f)


