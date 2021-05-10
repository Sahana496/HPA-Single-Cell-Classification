# HPA-Single-Cell-Classification

## Motivation
Humans are made up of trillions of cells and these cells are organized into several groups based on functionality. Cells of the same cell type are nearly identical and are considered to carry out the same functions. However, these cells may still show marked intrinsic cell-to-cell variability in gene expression. This is termed as single cell variability (SCV). This cellular heterogeneity can occur due to differences in the localization of proteins. Identifying these differences in single cells, why and how they occur, is important to understand the pathological processes associated with the development of a disease. This can ultimately help us find better treatments for those diseases.

The challenge is a weakly supervised multi-label classification problem. Given images of cells from microscopes and labels of protein location assigned together for all cells in the image, we need to develop a model capable of segmenting and classifying each individual cell with precise labels. The hypothesis is that we will be able to more accurately model the spatial organization of the human cell which may accelerate our growing understanding of how human cells function and how diseases develop.  

## Task
- Weakly-supervised multi-label classification
    - Weak supervision indicates the training is conducted on data with labels that are noisy and/or imprecise
    - In our case, the labels are provided at the image level but classification is output at the cell level
    - Multi-label indicates that the instances can have more than one classification
    - In our case, there could be multiple organelles labeled in each cell

The Data is downloaded from Kaggle (https://www.kaggle.com/c/hpa-single-cell-image-classification/data).

## Implementation
Mask R-CNN framework with three different backbones:
    - ResNet101-FPN
    - ResNeXt101-FPN
    - DenseNet121

ResNet101 and ResNeXt101 with Feature Pyramid Network was implemented using the Detectron2 library.
Densenet121 was implemented using Torchvision library from PyTorch.

## Running code
To train MaskRCNN with Densenet121 backbone, run-model.sbatch batchfile must be submitted (after updating the path to your Singularity container).
To train MaskRCNN with ResNet101-FPN and ResNeXt101-FPN backbone, detectron_code/run-model.sbatch batchfile must be submitted (after updating the path to your Singularity container).
The notebook **Densenet-Eval.ipynb** contains the code for visualizing the output predictions from MaskRCNN with a DenseNet121 backbone.
The notebook **detectron_code/Resnext101-Eval.ipynb** contains the code for visualizing the output predictions from MaskRCNN with a ResNeXt121-FPN backbone.





