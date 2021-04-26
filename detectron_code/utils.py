import os
import numpy as np
import json
import cv2
import pandas as pd

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
