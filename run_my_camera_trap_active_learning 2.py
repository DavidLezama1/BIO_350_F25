"""
run_my_camera_trap_active_learning.py

Custom driver script that:
- loads a pretrained embedding model,
- builds an ExtendedImageFolder dataset from your images,
- runs an active learning strategy,
- trains an MLP classifier on labeled embeddings,
- saves snapshots of results.

This is a simplified, understandable version of the full AL pipeline.
"""

# Import all the necessary libraries
import os
import sys
import random
import pickle
import argparse
import numpy as np
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

# Import project modules
from deep_learning.data_loader import ExtendedImageFolder
from deep_learning.engine import Engine
from deep_learning.utils import load_checkpoint, save_checkpoint, getCriterion
from deep_learning.networks import NormalizedEmbeddingNet, SoftmaxNet
from deep_learning.active_learning_manager import ActiveLearningManager
from active_learning_methods.constants import get_AL_sampler, get_wrapper_AL_mapping

# Loads additional sampler types (bandit, simulate_batch)
get_wrapper_AL_mapping()


# -------------------------------------------------------
# Argument parser (lets you choose dataset, model, settings)

def parse_args():
    # Create a command-line argument parser so the script can be configured from the terminal
    parser = argparse.ArgumentParser(description="Custom Camera Trap Active Learning")

    # Root folder with subfolders per class (ImageFolder-style dataset)
    parser.add_argument('--run_data', type=str, required=True,
                        help='Your image dataset root (with class folders).')

    # Pretrained embedding checkpoint produced by train_embedding.py
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to pretrained embedding checkpoint.')

    # Folder name where all AL snapshots and checkpoints will be written
    parser.add_argument('--experiment_name', type=str, default='camera_trap_AL_experiment',
                        help='Folder where results will be saved.')

    # DataLoader worker threads
    parser.add_argument('-j', '--num_workers', type=int, default=4,
                        help='Dataloader workers.')

    # Batch size used when computing embeddings over the full dataset
    parser.add_argument('-b', '--batch_size', type=int, default=256,
                        help='Batch size for embedding updates.')

    # Number of new points the AL sampler will request per round
    parser.add_argument('-N', '--active_batch', type=int, default=100,
                        help='How many samples to query each AL round.')

    # Total labeling budget (max number of labeled samples)
    parser.add_argument('-A', '--active_budget', type=int, default=5000,
                        help='Total number of samples to label.')

    # Name of the active learning strategy to use (e.g., margin, kcenter, uniform, etc.)
    parser.add_argument('--active_learning_strategy', type=str, default='margin',
                        help='Which AL method to use.')

    # Loss type originally used to train the embedding model (softmax, triplet, siamese)
    parser.add_argument('--loss_type', type=str, default='triplet',
                        help='Loss used to train the embedding model.')

    # Margin value for triplet / contrastive losses (if applicable)
    parser.add_argument('--margin', type=float, default=1.0,
                        help='Margin for triplet/contrastive losses.')

    # P = number of classes per balanced batch, K = samples per class
    parser.add_argument('--finetuning_P', type=int, default=16)
    parser.add_argument('--finetuning_K', type=int, default=4)
    parser.add_argument('--finetuning_lr', type=float, default=1e-4)
    parser.add_argument('--num_finetune_epochs', type=int, default=10)

    # Whether to min–max normalize embeddings before passing to the AL sampler
    parser.add_argument('--normalize_embedding', action='store_true',
                        help='Normalize embeddings before AL.')

    # Strategy for choosing triplets / pairs inside the metric-learning loss
    parser.add_argument('--finetuning_strategy', type=str, default='random',
                        help='Triplet/contrastive pair selection strategy.')

    # Parse and return all CLI arguments
    return parser.parse_args()


# -------------------------------------------------------
# Main function: This will run the entire AL workflow

def main():
    # Parse command-line arguments
    args = parse_args()
    print("Arguments:", args)

    # Choose GPU if available, otherwise fall back to CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    # Load pretrained embedding checkpoint (weights + metadata)
    checkpoint = load_checkpoint(args.base_model)

    # Use loss type from checkpoint if stored (keeps it consistent)
    if 'loss_type' in checkpoint:
        args.loss_type = checkpoint['loss_type']

    # Feature dimension and backbone architecture from checkpoint
    feat_dim = checkpoint.get('feat_dim', 256)
    arch = checkpoint['arch']

    # Create experiment folder if it does not already exist
    if not os.path.exists(args.experiment_name):
        os.mkdir(args.experiment_name)

    print("Experiment directory:", args.experiment_name)

    # ---------------------------------------------------
    # Build embedding network (feature extractor)
    
    # If the original model was trained with softmax, use SoftmaxNet; otherwise NormalizedEmbeddingNet
    if checkpoint['loss_type'] == 'softmax':
        embedding_net = SoftmaxNet(arch, feat_dim, checkpoint['num_classes'])
    else:
        embedding_net = NormalizedEmbeddingNet(arch, feat_dim)

    # Wrap model in DataParallel and move to the chosen device
    model = torch.nn.DataParallel(embedding_net).to(device)
    model.load_state_dict(checkpoint['state_dict'])  # Load pretrained weights
    
    # ---------------------------------------------------
    # Load dataset
    
    # ExtendedImageFolder expects run_data with subfolders per class
    target_dataset = ExtendedImageFolder(args.run_data)
    print("Found", len(target_dataset), "images.")

    # Criterion (loss) used when finetuning the embedding network
    criterion = getCriterion(args.loss_type, args.finetuning_strategy, args.margin)

    # Active Learning Manager handles:
    # embedding extraction
    # managing default and active pools
    # finetuning the embedding model
    env = ActiveLearningManager(
        dataset=target_dataset,
        embedding_model=model,
        device=device,
        criterion=criterion,
        normalize=args.normalize_embedding
    )

    # AL sampler will be instantiated once embeddings are available
    sampler = None
    N = args.active_batch

    # Simple MLP classifier trained on top of embeddings
    classifier = MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=2000)

    # ---------------------------------------------------
    # Active learning loop
    
    print("Starting AL loop...")
    # Number of currently labeled samples in active_pool
    num_queries = len(env.active_pool)

    # How often to finetune and how many random points to label initially
    FINETUNE_INTERVAL = 2000
    INITIAL_RANDOM = 1000  # initial random labels for warm start

    # Continue querying until we reach the active_budget
    while num_queries <= args.active_budget:
        print("\n=== Active learning step ===")
        print("Currently labeled:", num_queries)

        # Select samples to label on this AL step
        if num_queries == 0:
            # First step: randomly pick an initial seed set to bootstrap the classifier
            pool = env.default_pool
            initial_n = min(INITIAL_RANDOM, len(pool))
            print("Initial random seed size:", initial_n, "pool size:", len(pool))

            # Use Python's random.sample (no replacement)
            indices = random.sample(pool, initial_n)
        else:
            # Subsequent steps: use the AL sampler (e.g., margin, kcenter, etc.)
            indices = sampler.select_batch(
                N=N,
                already_selected=env.active_pool,
                model=classifier
            )

        # Move selected samples from default_pool (unlabeled) to active_pool (labeled)
        env.active_pool.extend(indices)
        env.default_pool = list(set(env.default_pool) - set(indices))
        num_queries = len(env.active_pool)

        print("Total labeled:", num_queries)

        # ---------------------------------------------------
        # Finetune embedding model occasionally
        
        # Finetune at the first big random initialization or every FINETUNE_INTERVAL thereafter
        if num_queries == INITIAL_RANDOM or (num_queries > INITIAL_RANDOM and num_queries % FINETUNE_INTERVAL == 0):
            print("Finetuning embedding model...")
            env.finetune_embedding(
                epochs=args.num_finetune_epochs,
                P=args.finetuning_P,
                K=args.finetuning_K,
                lr=args.finetuning_lr,
                num_workers=args.num_workers
            )

            # Save updated embedding model checkpoint for this AL stage
            ckpt_path = os.path.join(args.experiment_name, f"finetuned_{num_queries}.tar")
            save_checkpoint({'state_dict': model.state_dict(), 'feat_dim': feat_dim}, False, ckpt_path)

            print("Updating embeddings...")
            # Recompute embeddings for the full dataset using the finetuned model
            env.updateEmbedding(batch_size=args.batch_size, num_workers=args.num_workers)

            # Initialize AL sampler with updated embeddings and a fresh random seed
            print("Initializing sampler:", args.active_learning_strategy)
            seed = random.randint(0, 2**62)
            sampler = get_AL_sampler(args.active_learning_strategy)(
                env.embedding, None, seed
            )

        # ---------------------------------------------------
        # Train MLP classifier on current labeled embeddings
        
        # Get embeddings + labels for currently labeled points and fit classifier
        X_train, y_train = env.getTrainSet()
        classifier.fit(X_train, y_train)

        # Evaluate performance on the "test" set (here, default_pool embeddings)
        X_test, y_test = env.getTestSet()
        acc = classifier.score(X_test, y_test)
        print(f"Accuracy after {num_queries} labels: {acc:.4f}")

        # Compute confusion matrix and per-class accuracy for monitoring
        cm = confusion_matrix(y_test, classifier.predict(X_test))
        pc_acc = (cm.astype(float) / cm.sum(axis=1)[:, None]).diagonal()

        # Save snapshot of the current AL state
        snapshot_path = os.path.join(args.experiment_name, f"AL_snapshot_{num_queries}.pth")
        torch.save({
            'classifier': pickle.dumps(classifier),
            'pools': env.get_pools(),
            'confusion_matrix': cm,
            'per_class_accuracy': pc_acc
        }, snapshot_path)

        # Stop when budget is reached
        if num_queries >= args.active_budget:
            break

    print("Active learning completed.")


if __name__ == "__main__":
    main()
