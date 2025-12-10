# Standard library imports for argument parsing, filesystem operations,
# randomness, system interaction, timing, and serialization
import argparse
import os
import random
import sys
import time
import pickle

# Progress bar and numerical computing
from tqdm import tqdm
import numpy as np

# PyTorch core modules
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from torch.utils.data import DataLoader, TensorDataset

# Scikit-learn classifier and evaluation utilities
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Project-specific deep learning components
from deep_learning.data_loader import ExtendedImageFolder
from deep_learning.engine import Engine
from deep_learning.losses import OnlineContrastiveLoss, OnlineTripletLoss
from deep_learning.networks import NormalizedEmbeddingNet, SoftmaxNet
from deep_learning.utils import HardestNegativeTripletSelector, RandomNegativeTripletSelector, SemihardNegativeTripletSelector, HardNegativePairSelector
from deep_learning.utils import load_checkpoint, save_checkpoint, getCriterion
from deep_learning.active_learning_manager import ActiveLearningManager

# Active learning strategy mapping and factory
from active_learning_methods.constants import get_AL_sampler, get_wrapper_AL_mapping

# Print full NumPy arrays without truncation (useful for debug)
np.set_printoptions(threshold=np.inf)

# Initialize wrapper mappings for active learning methods (bandits, mixtures, etc.)
get_wrapper_AL_mapping()

# Supported loss types for the underlying embedding model
LOSS_TYPES = ['softmax', 'triplet', 'siamese']

# Supported active learning strategies (uncertainty, diversity, clustering, etc.)
STRATEGY_TYPES = ['uniform', 'graph_density', 'entropy', 'confidence',
     'kcenter', 'margin', 'informative_diverse', 'margin_cluster_mean', 'hierarchical']

# Argument parser for configuring the active learning run
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Path to the dataset over which active learning will be performed
parser.add_argument('--run_data', help='path to train dataset', default='SS_crops_256')

# DataLoader workers
parser.add_argument('-j', '--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')

# Batch size used when updating embeddings (via env.updateEmbedding)
parser.add_argument('-b', '--batch_size', default=256, type=int, help='mini-batch size (default: 256)')

# Path to the pretrained embedding checkpoint file
parser.add_argument('--base_model', default='', type=str, help='path to latest checkpoint (default: none)')

# Prefix/folder name for saving experiment outputs and snapshots
parser.add_argument('--experiment_name', default='', type=str, help='prefix name for output files')

# Number of points to query in each active learning batch
parser.add_argument('-N', '--active_batch', default=100, type=int, help='number of queries per batch')

# Total labeling budget (maximum number of queried points)
parser.add_argument('-A', '--active_budget', default= 30000, type=int, help='number of queries per batch')

# Parameters for balanced finetuning batches: P classes, K samples each
parser.add_argument('--finetuning_P', default=16, type=int,
                    help='The number of classes in each balanced batch')
parser.add_argument('--finetuning_K', default=4, type=int,
                    help='The number of examples from each class in each balanced batch')

# Learning rate for finetuning the embedding model
parser.add_argument('--finetuning_lr', default=0.0001,
                    type=float, help='initial learning rate')

# Choice of active learning strategy from STRATEGY_TYPES
parser.add_argument('--active_learning_strategy', default='margin', choices=STRATEGY_TYPES, help='Active learning strategy')

# Loss type used for the embedding model during finetuning
parser.add_argument('--loss_type', default='triplet', choices=LOSS_TYPES,
                    help='loss type: ' + ' | '.join(LOSS_TYPES) + ' (default: triplet loss)')

# Margin hyperparameter for triplet/siamese loss
parser.add_argument('--margin', default=1.0, type=float,
                    help='margin for siamese or triplet loss')

# Strategy for selecting triplets/pairs when finetuning
parser.add_argument('--finetuning_strategy', default='random', choices=['hardest', 'random', 'semi_hard', 'hard_pair'],
                    help='data selection strategy')

# Number of epochs to finetune the embedding model each time
parser.add_argument('--num_finetune_epochs', default= 100, type=int,
                    help='number of total epochs to run for finetuning')

# Whether to normalize embedding vectors (e.g., to [0,1]) when used by AL sampler
parser.add_argument('--normalize_embedding', action="store_true",
                    help='If normalize embedding values or not')



def main():
    # Parse arguments from the command line
    args = parser.parse_args()
    print(args)

    # Check if a GPU is available, otherwise use CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE: ", device)

    # Load a pretrained embedding model checkpoint
    checkpoint = load_checkpoint(args.base_model)

    # Setup experiment directory and default experiment name if none is given
    if args.experiment_name == '':
        args.experiment_name = "experiment_%s_%s"%(checkpoint['loss_type'], args.active_learning_strategy)
    if not os.path.exists(args.experiment_name):
        os.mkdir(args.experiment_name)

    # Rebuild the embedding model architecture from checkpoint metadata
    if checkpoint['loss_type'].lower() == 'center' or checkpoint['loss_type'].lower() == 'softmax':
        embedding_net = SoftmaxNet(checkpoint['arch'], checkpoint['feat_dim'], checkpoint['num_classes'], False)
    else:
        embedding_net = NormalizedEmbeddingNet(checkpoint['arch'], checkpoint['feat_dim'], False)

    # Wrap model for multi-GPU (if available) and move to device
    model = torch.nn.DataParallel(embedding_net).to(device)
    # Load saved weights
    model.load_state_dict(checkpoint['state_dict'])

    # Setup the target dataset on which active learning will run
    target_dataset = ExtendedImageFolder(args.run_data)

    # Setup finetuning loss (softmax, triplet, or siamese) with given strategy and margin
    criterion = getCriterion(args.loss_type, args.finetuning_strategy, args.margin)

    # Create an active learning manager to track pools and embeddings
    env = ActiveLearningManager(target_dataset, model, device, criterion, args.normalize_embedding)

    # Placeholder for the active learning sampler (initialized after first embedding update)
    sampler = None

    # Number of points per AL batch
    N = args.active_batch

    # Create a classifier to train on the selected labeled embeddings
    classifier = MLPClassifier(hidden_layer_sizes=(150, 100), alpha=0.0001, max_iter= 2000)

    # Main active learning loop
    print("Active learning loop is started")
    # Number of points queried so far (size of active labeled pool)
    numQueries = len(env.active_pool)

    # Continue querying until we hit the active learning budget
    while numQueries <= args.active_budget:
        # Active Learning: choose which images to label next
        if numQueries == 0:
            # First iteration: pick an initial random labeled set of 1000 images
            indices = np.random.choice(env.default_pool, 1000, replace=False).tolist()
        else:
            # Later iterations: use active learning sampler based on current classifier
            indices = sampler.select_batch(N= N, already_selected= env.active_pool, model= classifier)

        # Move queried indices from default_pool (unlabeled) to active_pool (labeled)
        env.active_pool.extend(indices)
        env.default_pool = list(set(env.default_pool).difference(indices))
        numQueries = len(env.active_pool)

        # Periodically finetune the embedding model and recompute embeddings
        # Here, every time numQueries reaches 1000, 3000, 5000, ... (i.e., 1000 mod 2000)
        if numQueries % 2000 == 1000:
            # Finetune embedding on the currently labeled pool
            env.finetune_embedding(args.num_finetune_epochs, args.finetuning_P, args.finetuning_K, args.finetuning_lr)
            # Save finetuned embedding checkpoint
            save_checkpoint({
            'arch': checkpoint['arch'],
            'state_dict': model.state_dict(),
            'loss_type' : checkpoint['loss_type'],
            'feat_dim' : checkpoint['feat_dim']
            }, False, "%s/%s%s_%s_%04d.tar" 
            % (args.experiment_name, 'finetuned', checkpoint['loss_type'], checkpoint['arch'], numQueries))

            # Update cached embedding vectors for the entire dataset
            env.updateEmbedding(batch_size=args.batch_size, num_workers=args.num_workers)
            # Reinitialize the AL sampler with the latest embeddings
            sampler = get_AL_sampler(args.active_learning_strategy)(env.embedding, None, random.randint(0, sys.maxsize * 2 + 1)) 

        # Gather labeled pool embeddings and labels, and train the classifier
        X_train, y_train = env.getTrainSet()
        classifier.fit(X_train, y_train)

        # Evaluate classifier on the remaining pool (treated here as test set)
        X_test, y_test= env.getTestSet()
        print("Number of Queries: %d,  Accuracy: %.4f"%(numQueries, classifier.score(X_test, y_test)))

        # Compute confusion matrix and per-class accuracy
        cm = confusion_matrix(y_test, classifier.predict(X_test))
        pc_acc = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]).diagonal()

        # Save a snapshot of the current AL state (classifier, pools, metrics)
        torch.save({'classifier': pickle.dumps(classifier) , "pools": env.get_pools(), 
            "confusion_matrix":cm, "per_class_accuracy": pc_acc, "class_to_idx": target_dataset.class_to_idx},
            "%s/%s_%04d.pth"%(args.experiment_name, 'AL_snapshot', numQueries), pickle_protocol=4)
        sys.stdout.flush()

# Standard Python entry point guard
if __name__ == '__main__':
    main()
