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
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN
from utils.engine import train_one_epoch, evaluate
import utils.utils
import utils.transforms as T
from HPADataset import HPADataset
import time
import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torchvision.models._utils
from torchvision.models.detection.anchor_utils import AnchorGenerator

def get_densenet_maskrcnn(num_classes):
    backbone = torchvision.models.densenet121(pretrained = True)
    modules=list(backbone.children())[:-1]
    backbone=torch.nn.Sequential(*modules)
    backbone.out_channels=1024
    
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    model = MaskRCNN(backbone, num_classes=num_classes, rpn_anchor_generator=anchor_generator)
    
    return model

def _validate_trainable_layers(pretrained, trainable_backbone_layers, max_value, default_value):
    # dont freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(max_value))
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers

def maskrcnn_resnet101_fpn(num_classes=20, model='resnet101', pretrained_backbone=True,
                           trainable_backbone_layers=5, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained_backbone, trainable_backbone_layers, 5, 3)

    backbone = resnet_fpn_backbone(model, pretrained_backbone, trainable_layers=trainable_backbone_layers)
    model = MaskRCNN(backbone, num_classes, **kwargs)
    return model

def get_instance_segmentation_model(num_classes, model_type, pretrained = True):
    # load an instance segmentation model pre-trained on COCO
#     model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = maskrcnn_resnet101_fpn(model=model_type, num_classes=num_classes, pretrained_backbone = pretrained)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.Normalize())
    else:
        transforms.append(T.Normalize())
        
    return T.Compose(transforms)
    
def main(args):

    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    # use our dataset and defined transformations
    ROOT = args.data_path
    ROOT = '../data'
    dataset_train = HPADataset(ROOT,csv='./data/train_val_single.csv', transforms=get_transform(train=True))
    dataset_val = HPADataset(ROOT,csv='./data/test_split.csv', transforms=get_transform(train=False))

    # split the dataset in train and validation set
    torch.manual_seed(1)

    num_classes = 20
    print("Creating data loaders")

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=16, shuffle=True, num_workers=args.workers,
        collate_fn=utils.utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=16, shuffle=False, num_workers=args.workers,
        collate_fn=utils.utils.collate_fn)

    print("Creating model")
    if args.model == 'resnet101' or args.model == 'resnet50':
        model = get_instance_segmentation_model(num_classes, args.model)
    elif args.model == 'densenet':
        model = get_densenet_maskrcnn(num_classes)
    else:
        print("Invalid model")
        exit()
    
    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
#     lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        evaluate(model, data_loader_val, device=device)
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        if args.checkpoint_dir:
            utils.utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.checkpoint_dir, 'model_{}.pth'.format(epoch)))

        # evaluate after every epoch
        evaluate(model, data_loader_val, device=device)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Detection Training')

    parser.add_argument('--data-path', default='../data', help='dataset')
    parser.add_argument('--model', default='resnet101', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.0025, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=3, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[2, 5, 7], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--checkpoint-dir', default='./checkpoints', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.checkpoint_dir:
        utils.utils.mkdir(args.checkpoint_dir)

    main(args)
