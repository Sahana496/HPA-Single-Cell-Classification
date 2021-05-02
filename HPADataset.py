import os
import numpy as np
import torch
from PIL import Image
import pandas as pd
import cv2
import functools
import operator

class HPADataset(object):
    def __init__(self, root, csv, transforms=None):
        self.img_dir = os.path.join(root, 'train')
        self.mask_dir = os.path.join(root, 'hpa_cell_mask')
        self.transforms = transforms
        
        self.df = pd.read_csv(csv)
        
        #dropping this image since it has no masks
        self.df.drop(self.df[self.df['ID']=='940f418a-bba4-11e8-b2b9-ac1f6b6435d0'].index, inplace = True)  
        
        #Split labels
#         self.df["Label"] = self.df["Label"].str.split("|")
        
#         #fetch only images that have one label
#         self.df["Label Count"] = self.df['Label'].str.len()
#         self.df = self.df[self.df['Label Count'] == 1]
        self.imgs = list(self.df['ID'])  
        
        #image class
        self.classes = self.df['Label'].values #functools.reduce(operator.iconcat, self.df['Label'].values, [])

    
    def __getitem__(self, idx):
        
        # load images and masks
        img = self.load_RGBY_image(self.img_dir, self.imgs[idx], image_size=512)
#         img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
        img = Image.fromarray(np.uint8(img))
        
        masks, obj_ids = self.load_mask(self.imgs[idx])
     
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
       
        boxes = []
        labels = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(int(self.classes[idx]))
       
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        if len(boxes) == 0:
            print(self.imgs[idx], boxes.shape)
  
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
       
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
    
    def read_img(self, path, image_id, color, image_size=None):
        filename = f'{path}/{image_id}_{color}.png'
        assert os.path.exists(filename), f'not found {filename}'
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if image_size is not None:
            img = cv2.resize(img, (image_size, image_size))
        if img.dtype == 'uint16':
            img = (img/256).astype('uint8')
        return img

    # image loader, using rgb only here
    def load_RGBY_image(self, path, image_id, image_size=None):
        red = self.read_img(path, image_id, "red", image_size)
        green = self.read_img(path, image_id, "green", image_size)
        blue = self.read_img(path, image_id, "blue", image_size)

        stacked_images = np.transpose(np.array([red, green, blue]), (1,2,0))
        return stacked_images
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Read mask files from 
        masks = []
        class_ids = []
        cell_mask = np.load(f'{self.mask_dir}/{image_id}.npz')['arr_0']
        cell_mask = cv2.resize(cell_mask, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        #find number of cells in the image
        mask_ids = np.unique(cell_mask)
#         print(mask_ids)
        #Remove background
        mask_ids = mask_ids[1:]
        
        #create binary mask for every cell in the image
        masks = cell_mask == mask_ids[:,None, None]

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return masks, mask_ids