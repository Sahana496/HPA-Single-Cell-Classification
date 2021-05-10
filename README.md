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
