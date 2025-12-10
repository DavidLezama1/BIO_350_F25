# Import standard libraries for argument parsing, OS interaction, randomness, and numerics
import argparse
import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend so plots can be saved without a display

# Import PyTorch core modules
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim

# Import custom dataset, losses, utilities, networks, and training engine from the deep_learning package
from deep_learning.data_loader import ExtendedImageFolder
from deep_learning.losses import OnlineTripletLoss, OnlineContrastiveLoss
from deep_learning.utils import HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector, HardNegativePairSelector
from deep_learning.utils import load_checkpoint, save_checkpoint, plot_embedding, save_embedding_plot, getCriterion
from deep_learning.networks import NormalizedEmbeddingNet, SoftmaxNet, models
from deep_learning.engine import Engine

# Collect all valid torchvision architectures exposed via models.__dict__
ARCHITECTURES = sorted(name for name in models.__dict__
                       if name.islower() and not name.startswith("__")
                       and callable(models.__dict__[name]))
# Supported loss types: standard classification, metric-learning triplet, or siamese
LOSS_TYPES = ['softmax', 'triplet', 'siamese']

# Create an argument parser for command-line options
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Path to the training dataset (folder structured like ImageFolder)
parser.add_argument(
    '--train_data', help='path to train dataset', default='crops')
# Path to the validation dataset
parser.add_argument(
    '--val_data', help='path to validation dataset', default='smalls')
# CNN architecture to use (e.g. resnet18, resnet50, densenet121, etc.)
parser.add_argument('--arch', '-a', default='resnet18', choices=ARCHITECTURES,
                    help='model architecture: ' + ' | '.join(ARCHITECTURES) + ' (default: resnet18)')
# Number of DataLoader workers
parser.add_argument('-j', '--num_workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
# Total number of training epochs
parser.add_argument('--epochs', default=5, type=int,
                    help='number of total epochs to run')
# Mini-batch size
parser.add_argument('-b', '--batch_size', default=256,
                    type=int, help='mini-batch size (default: 256)')
# Learning rate for optimizer
parser.add_argument('--lr', '--learning_rate', default=0.00001,
                    type=float, help='initial learning rate')
# Weight decay (L2 regularization)
parser.add_argument('--weight_decay', '--wd', default=0.0005,
                    type=float, help='weight decay (default: 5e-4)')
# Optional checkpoint path to resume from
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
# Prefix used when saving checkpoints and embedding plots
parser.add_argument('--checkpoint_prefix', default='',
                    type=str, help='path to latest checkpoint (default: none)')
# How often (in epochs) to plot embeddings with t-SNE
parser.add_argument('--plot_freq', dest='plot_freq', type=int,
                    help='plot embedding frequence', default=1)
# Whether to use pretrained weights; note action='store_false' means default is True unless flag is passed
parser.add_argument('--pretrained', dest='pretrained',
                    action='store_false', help='use pre-trained model')
# Loss type: softmax (classification), triplet, or siamese (contrastive)
parser.add_argument('--loss_type', default='triplet', choices=LOSS_TYPES,
                    help='loss type: ' + ' | '.join(LOSS_TYPES) + ' (default: triplet loss)')
# Margin used for triplet or siamese loss
parser.add_argument('--margin', default=1.0, type=float,
                    help='margin for siamese or triplet loss')
# Dimensionality of the learned embedding vectors
parser.add_argument('-f', '--feat_dim', default=256,
                    type=int, help='embedding size (default: 256)')
# Raw image size (width, height) before transforms
parser.add_argument('--raw_size', nargs=2, default=[256, 256], type=int,
                    help='The width, height and number of channels of images for loading from disk')
# Processed image size (width, height) after transforms
parser.add_argument('--processed_size', nargs=2, default=[224, 224], type=int,
                    help='The width and height of images after preprocessing')
# Number of classes per balanced batch (P). -1 means “use all classes”.
parser.add_argument('--balanced_P', default=-1, type=int,
                    help='The number of classes in each balanced batch')
# Number of samples per class per balanced batch (K)
parser.add_argument('--balanced_K', default=10, type=int,
                    help='The number of examples from each class in each balanced batch')
# Strategy for triplet/contrastive sampling (hardest, random, semi-hard, or hard_pair)
parser.add_argument('--strategy', default='random', choices=['hardest', 'random', 'semi_hard', 'hard_pair'],
                    help='The number of examples from each class in each balanced batch')


def main():
    # Parse command-line arguments
    args = parser.parse_args()
    print(args)

    # Load a checkpoint if necessary (for resuming training)
    checkpoint = {}
    if args.resume != '':
        checkpoint = load_checkpoint(args.resume)
        # Ensure loss type and feature dimension match the checkpoint
        args.loss_type = checkpoint['loss_type']
        args.feat_dim = checkpoint['feat_dim']

    # Setup the training dataset and the validation dataset using ExtendedImageFolder
    train_dataset = ExtendedImageFolder(args.train_data)
    if args.val_data is not None:
        val_dataset = ExtendedImageFolder(args.val_data)
    
    # Determine number of classes from the training dataset
    num_classes = len(train_dataset.classes)
    # If balanced_P is -1, use all classes per batch
    if args.balanced_P == -1:
        args.balanced_P = num_classes

    # Setup data loaders depending on the type of loss
    if args.loss_type.lower() == 'softmax':
        # For softmax (classification), use standard DataLoader
        train_loader = train_dataset.getSingleLoader(batch_size = 128, shuffle = True, num_workers = args.num_workers)
        train_embd_loader = train_loader
        if args.val_data is not None:
            val_loader = val_dataset.getSingleLoader(batch_size = 128, shuffle = False, num_workers = args.num_workers, transfm = 'val')
            val_embd_loader = val_loader
    else:
        # For metric-learning losses, use balanced batches (P classes × K samples)
        train_loader = train_dataset.getBalancedLoader(P = args.balanced_P, K = args.balanced_K, num_workers = args.num_workers)
        # Separate loader for embedding extraction (no balancing needed)
        train_embd_loader = train_dataset.getSingleLoader(num_workers=args.num_workers)
        if args.val_data is not None:
            val_loader = val_dataset.getBalancedLoader(P = args.balanced_P, K = args.balanced_K, num_workers = args.num_workers, transfm = 'val')
            val_embd_loader = val_dataset.getSingleLoader(batch_size = 128, shuffle = False, num_workers = args.num_workers, transfm = 'val')
    
    # Check if a GPU is available; fall back to CPU otherwise
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE: ", device)

    # Create a model: classification net for softmax, embedding net for triplet/siamese
    if args.loss_type.lower() == 'softmax':
        model = torch.nn.DataParallel(SoftmaxNet(
            args.arch, args.feat_dim, num_classes, use_pretrained=args.pretrained))
    else:
        model = torch.nn.DataParallel(NormalizedEmbeddingNet(
            args.arch, args.feat_dim, use_pretrained=args.pretrained))

    # Setup loss criterion using helper (CrossEntropy, Triplet, or Contrastive)
    criterion = getCriterion(args.loss_type, args.strategy, args.margin)

    # Define optimizer (Adam) over all model parameters
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1

    # Load a checkpoint if provided (to resume from previous training state)
    if checkpoint:
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    # Setup a deep learning engine (handles training/validation/prediction loops)
    e = Engine(device, model, criterion, optimizer)
    # Train the model over the requested number of epochs
    for epoch in range(start_epoch, args.epochs + 1):
        # Train for one epoch; compute accuracy only for softmax
        e.train_one_epoch(train_loader, epoch, True if args.loss_type.lower() == 'softmax' else False)
        # Optionally plot embeddings every plot_freq epochs
        if epoch % args.plot_freq == 0 and epoch > 0:
            a, b, _ = e.predict(
                train_embd_loader, load_info=True, dim=args.feat_dim)
            save_embedding_plot("%s_train_%d.jpg"%(args.checkpoint_prefix, epoch), a, b, {})
            a, b, _ = e.predict(
                val_embd_loader, load_info=True, dim=args.feat_dim)
            save_embedding_plot("%s_val_%d.jpg"%(args.checkpoint_prefix, epoch), a, b, {})
        # Evaluate on validation set if provided
        if args.val_data is not None:
            e.validate(val_loader, True if args.loss_type.lower()
                       == 'softmax' else False)
        # Save a checkpoint each epoch with metadata and model state
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_type': args.loss_type,
            'num_classes': num_classes,
            'feat_dim': args.feat_dim
        }, False, "%s%s_%s_%04d.tar" % (args.checkpoint_prefix, args.loss_type, args.arch, epoch))


if __name__ == '__main__':
    # Entry point: call main() when script is executed
    main()
