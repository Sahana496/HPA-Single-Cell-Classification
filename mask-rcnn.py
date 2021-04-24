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
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from utils.engine import train_one_epoch, evaluate
import utils.utils
import utils.transforms as T
from HPADataset import HPADataset
import time

def get_instance_segmentation_model(num_classes, model_type, pretrained):
    # load an instance segmentation model pre-trained on COCO
    if model_type == 'maskrcnn_resnet50_fpn':    
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrained)
    else:
        print("Model not found")
        exit()

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
    return T.Compose(transforms)

# def main():
#     # use our dataset and defined transformations
#     ROOT = '../data'
#     dataset = HPADataset(ROOT, transforms=get_transform(train=True))
#     dataset_val = HPADataset(ROOT, transforms=get_transform(train=False))

#     # split the dataset in train and validation set
#     torch.manual_seed(1)
#     indices = torch.randperm(len(dataset)).tolist()
#     dataset = torch.utils.data.Subset(dataset, indices[:-50])
#     dataset_val = torch.utils.data.Subset(dataset_val, indices[-50:])

#     # define training and validation data loaders
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=8, shuffle=True, num_workers=8,
#         collate_fn=utils.utils.collate_fn)

#     data_loader_val = torch.utils.data.DataLoader(
#         dataset_val, batch_size=8, shuffle=False, num_workers=8,
#         collate_fn=utils.utils.collate_fn)
    
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#     print(device)
#     # our dataset has 20 classes - background + 19 cell types
#     num_classes = 20

#     # get the model using our helper function
#     model = get_instance_segmentation_model(num_classes)
#     # move model to the right device
#     model.to(device)

#     # construct an optimizer
#     params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.SGD(params, lr=0.005,
#                                 momentum=0.9, weight_decay=0.0005)

#     # and a learning rate scheduler which decreases the learning rate by
#     # 10x every 3 epochs
#     lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                                    step_size=3,
#                                                    gamma=0.1)
    
#     # let's train it for 10 epochs
#     num_epochs = 10
#     checkpoint_dir ='./checkpoints'
#     print("Training started")
#     for epoch in range(num_epochs):
#         print("Epoch: {}".format(epoch))
#         # train for one epoch, printing every 10 iterations
#         train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
#         # update the learning rate
#         lr_scheduler.step()
#         # evaluate on the test dataset
#         print("Evaluating model")
#         evaluate(model, data_loader_val, device=device)
#         print("Saving model")
#         torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'epoch-{}.pth'.format(epoch)))
    
    
def main(args):
#     utils.utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    # Data loading code
    print("Loading data")

    # use our dataset and defined transformations
    ROOT = args.data_path
    dataset_train = HPADataset(ROOT, transforms=get_transform(train=True))
    dataset_val = HPADataset(ROOT, transforms=get_transform(train=False))

    # split the dataset in train and validation set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:-50])
    dataset_val = torch.utils.data.Subset(dataset_val, indices[-50:])

    num_classes = 20
    print("Creating data loaders")
#     if args.distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
#         test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_val)
#     else:
    train_sampler = torch.utils.data.RandomSampler(dataset_train)
    test_sampler = torch.utils.data.SequentialSampler(dataset_val)

#     if args.aspect_ratio_group_factor >= 0:
#         group_ids = create_aspect_ratio_groups(dataset_train, k=args.aspect_ratio_group_factor)
#         train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)
#     else:
    train_batch_sampler = torch.utils.data.BatchSampler(
            train_sampler, args.batch_size, drop_last=True)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_sampler=train_batch_sampler, num_workers=args.workers,
        collate_fn=utils.utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.utils.collate_fn)

    print("Creating model")
    model = get_instance_segmentation_model(num_classes, args.model, args.pretrained)

    model.to(device)

    model_without_ddp = model
#     if args.distributed:
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
#         model_without_ddp = model.module

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

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
#         if args.distributed:
#             train_sampler.set_epoch(epoch)
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
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.005, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
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
