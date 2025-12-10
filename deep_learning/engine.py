from .utils import *

import time
import sys
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler

# utility function
def generate_description(desc, loss, top1 = None, top5 = None):
    """Create a human-readable description string for tqdm progress bars.

    Args:
        desc (str): Short prefix describing what is happening (e.g., 'Training epoch 1:').
        loss (AverageMeter): Tracks average loss value.
        top1 (AverageMeter, optional): Tracks average top-1 accuracy.
        top5 (AverageMeter, optional): Tracks average top-5 accuracy.

    Returns:
        str: Formatted description string with current averages.
    """
    if top1 is not None and top5 is not None:
        return '%s Avg. Loss: %.4f\tAvg. Top-1 Acc.: %.3f\tAvg. Top-5 Acc.: %.3f' % (
            desc, loss.avg, top1.avg, top5.avg
        )
    else:
        return '%s Avg. Loss: %.4f' % (desc, loss.avg)


class Engine():
    """Helper class that wraps a model, loss, and optimizer for training and evaluation.

    This class provides:
      - One-batch training and validation functions.
      - One-epoch training and validation loops with tqdm logging.
      - Convenience methods for running prediction and extracting embeddings.
    """

    def __init__(self, device, model, criterion, optimizer):
        """Initialize engine with model, loss function, optimizer, and device.

        Args:
            device (torch.device): Device to run computation on (CPU or GPU).
            model (torch.nn.Module): Neural network model.
            criterion (torch.nn.Module or callable): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        """
        # Move model to the appropriate device (e.g., GPU if available).
        self.model = model.to(device)
        self.criterion = criterion
        if self.criterion is not None:
            self.criterion.to(device)
        self.optimizer = optimizer
        self.device = device

    def train_one_batch(self, input, target, iter_num, calcAccuracy):
        """Run a single optimization step on one batch of training data.

        Args:
            input (Tensor): Input batch of images.
            target (Tensor): Ground-truth labels for the batch.
            iter_num (int): Iteration index (not used here, but kept for consistency/logging).
            calcAccuracy (bool): If True, compute and return top-1 and top-5 accuracy.

        Returns:
            float or (float, Tensor, Tensor):
                - If calcAccuracy is False: returns loss value (scalar).
                - If calcAccuracy is True: returns (loss_value, acc1, acc5).
        """
        # Move data to the chosen device.
        input = input.to(self.device)
        target = target.to(self.device)

        # Forward pass: model expected to return (logits, embedding).
        output, _ = self.model(input)

        # Optionally compute accuracy metrics.
        if calcAccuracy:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # Compute loss.
        loss = self.criterion(output, target)

        # Standard PyTorch optimization step.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if not calcAccuracy:
            # Only return scalar loss for logging.
            return loss.item()
        else:
            # Return loss and accuracy metrics.
            return loss.item(), acc1, acc5

    def train_one_epoch(self, train_loader, epoch_num, calcAccuracy):
        """Train the model for one full epoch.

        Args:
            train_loader (DataLoader): Iterable over training batches.
            epoch_num (int): Current epoch index, used for logging.
            calcAccuracy (bool): If True, track and display accuracy metrics.
        """
        # Track running average of loss across the epoch.
        losses = AverageMeter()
        if calcAccuracy:
            # Track running averages of top-1 and top-5 accuracies.
            top1 = AverageMeter()
            top5 = AverageMeter()

        # Switch model into training mode (enables dropout, batchnorm updates, etc.).
        self.model.train()

        # Wrap the loader with tqdm to get a progress bar.
        train_loader = tqdm(train_loader)
        for i, batch in enumerate(train_loader):
            # Each batch is expected to be (inputs, targets, [optional extra info]).
            input = batch[0]
            target = batch[1]

            # Train on this batch and optionally get accuracy metrics.
            if calcAccuracy:
                loss, acc1, acc5 = self.train_one_batch(input, target, i, True)
                # Update running accuracy meters.
                top1.update(acc1, input.size(0))
                top5.update(acc5, input.size(0))
            else:
                loss = self.train_one_batch(input, target, i, False)

            # Update running loss meter.
            losses.update(loss, input.size(0))

            # Update tqdm status line with the current averages.
            if calcAccuracy:
                train_loader.set_description(
                    generate_description(
                        "Training epoch %d:" % epoch_num,
                        losses,
                        top1=top1,
                        top5=top5
                    )
                )
            else:
                train_loader.set_description(
                    generate_description("Training epoch %d:" % epoch_num, losses)
                )

    def validate_one_batch(self, input, target, iter_num, calcAccuracy):
        """Run a forward pass and compute validation loss/accuracy for one batch.

        Args:
            input (Tensor): Input batch of images.
            target (Tensor): Ground-truth labels.
            iter_num (int): Batch index (not directly used).
            calcAccuracy (bool): If True, compute top-1 and top-5 accuracy.

        Returns:
            float or (float, Tensor, Tensor):
                - If calcAccuracy is False: loss value.
                - If calcAccuracy is True: (loss_value, acc1, acc5).
        """
        with torch.no_grad():
            # Move data to device.
            input = input.to(self.device)
            target = target.to(self.device)

            # compute output
            output, _ = self.model(input)

            # measure accuracy and record loss
            if calcAccuracy:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            loss = self.criterion(output, target)

            if not calcAccuracy:
                return loss.item()
            else:
                return loss.item(), acc1, acc5

    def validate(self, val_loader, calcAccuracy):
        """Evaluate the model on a validation set.

        Args:
            val_loader (DataLoader): Iterable over validation batches.
            calcAccuracy (bool): If True, print and track accuracy metrics.

        Returns:
            float: Average validation loss across all batches.
        """
        # Running average loss.
        losses = AverageMeter()
        if calcAccuracy:
            # Running averages for top-1 and top-5 accuracy.
            top1 = AverageMeter()
            top5 = AverageMeter()

        # switch to evaluate mode (disables dropout, etc.)
        self.model.eval()
        val_loader = tqdm(val_loader)
        for i, batch in enumerate(val_loader):
            input = batch[0]
            target = batch[1]
            if calcAccuracy:
                loss, acc1, acc5 = self.validate_one_batch(input, target, i, True)
                # Update accuracy meters.
                top1.update(acc1, input.size(0))
                top5.update(acc5, input.size(0))
            else:
                loss = self.validate_one_batch(input, target, i, False)

            # Update loss meter.
            losses.update(loss, input.size(0))

            # Update tqdm display.
            if calcAccuracy:
                val_loader.set_description(
                    generate_description("Validation:", losses, top1=top1, top5=top5)
                )
            else:
                val_loader.set_description(
                    generate_description("Validation:", losses)
                )

        # Return average validation loss.
        return losses.avg

    def predict_one_batch(self, input, iter_num):
        """Forward one batch through the model and return logits only.

        Args:
            input (Tensor): Batch of inputs.
            iter_num (int): Batch index (not used directly).

        Returns:
            Tensor: Model output logits for the batch.
        """
        with torch.no_grad():
            input = input.to(self.device)
            # compute output
            output, _ = self.model(input)

        return output

    def predict(self, dataloader, load_info=False, dim=256):
        """Run prediction over an entire dataloader and collect outputs.

        Args:
            dataloader (DataLoader): DataLoader over the dataset to predict on.
            load_info (bool): If True, also return file paths of each sample.
            dim (int): Expected dimensionality of the output/logits.

        Returns:
            If load_info is False:
                (embeddings, labels)
            If load_info is True:
                (embeddings, labels, paths)
            where:
                embeddings: NumPy array of shape (N, dim)
                labels: NumPy array of ground-truth labels
                paths: list of file paths (if load_info=True)
        """
        # switch to evaluate mode
        self.model.eval()
        # Preallocate arrays to store model outputs and labels.
        embeddings = np.zeros((len(dataloader.dataset), dim), dtype=np.float32)
        labels = np.zeros(len(dataloader.dataset), dtype=np.int64)
        if load_info:
            # If requested, store paths alongside embeddings.
            paths = [None] * len(dataloader.dataset)
        k = 0
        dataloader = tqdm(dataloader, desc="Prediction:")
        for i, batch in enumerate(dataloader):
            images = batch[0]
            target = batch[1]
            # If dataset returns paths, they will be in batch[2].
            if load_info:
                paths[k:k + len(batch[2])] = batch[2]
            # Get logits for the current batch.
            embedding = self.predict_one_batch(images, i)
            # Save logits and labels as NumPy arrays.
            embeddings[k:k + len(images)] = embedding.data.cpu().numpy()
            labels[k:k + len(images)] = target.numpy()
            k += len(images)

        if load_info:
            return embeddings, labels, paths
        else:
            return embeddings, labels

    def embedding_one_batch(self, input, iter_num):
        """Forward one batch through the model and return only embeddings.

        Args:
            input (Tensor): A batch of images.
            iter_num (int): Batch index (not directly used).

        Returns:
            Tensor: Embedding vectors for the batch.
        """
        with torch.no_grad():
            input = input.to(self.device)
            # compute output; we ignore logits and keep only embedding.
            _, output = self.model(input)

        return output

    def embedding(self, dataloader, dim=256, normalize=False):
        """Extract embeddings for all samples in a dataloader.

        Args:
            dataloader (DataLoader): DataLoader yielding input batches.
            dim (int): Embedding dimensionality.
            normalize (bool): If True, apply MinMax scaling to [0, 1] per feature.

        Returns:
            np.ndarray: Array of shape (N, dim) with embeddings for the dataset.
        """
        # switch to evaluate mode
        self.model.eval()
        embeddings = np.zeros((len(dataloader.dataset), dim), dtype=np.float32)
        k = 0
        dataloader = tqdm(dataloader, desc="Embedding:")
        for i, batch in enumerate(dataloader):
            images = batch[0]
            # Compute embeddings for this batch.
            embedding = self.embedding_one_batch(images, i)
            embeddings[k:k + len(images)] = embedding.data.cpu().numpy()
            k += len(images)

        if normalize:
            # Optionally scale each embedding dimension between 0 and 1.
            scaler = MinMaxScaler()
            return scaler.fit_transform(embeddings)
        else:
            return embeddings
