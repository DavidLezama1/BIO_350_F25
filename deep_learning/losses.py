import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    """

    def __init__(self, margin, pair_selector):
        """
        Args:
            margin (float): Margin that pushes negative pairs at least this distance apart.
            pair_selector: Object with get_pairs(embeddings, targets) that returns
                           indices for positive and negative pairs in the batch.
        """
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        """Compute contrastive loss for a batch of embeddings.

        Args:
            embeddings (Tensor): Batch of feature vectors with shape (N, D).
            target (Tensor): Ground-truth labels with shape (N,).

        Returns:
            Tensor: Scalar contrastive loss averaged over all positive and negative pairs.
        """
        # Use the pair selector to get index pairs for positives and negatives.
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)

        # Ensure the index tensors are on the same device as embeddings (GPU if available).
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()

        # For positive pairs, penalize the squared L2 distance directly (pull embeddings together).
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)

        # For negative pairs, only penalize when distance is less than the margin (push embeddings apart).
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()
        ).pow(2)

        # Concatenate positive and negative losses and average them.
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        """
        Args:
            margin (float): Triplet margin that enforces a minimum gap between positive and negative distances.
            triplet_selector: Object with get_triplets(embeddings, targets) that returns
                              (anchor, positive, negative) index triplets.
        """
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        """Compute triplet loss over a batch of embeddings.

        Args:
            embeddings (Tensor): Batch of feature vectors with shape (N, D).
            target (Tensor): Ground-truth labels with shape (N,).

        Returns:
            Tensor: Scalar triplet loss averaged over all sampled triplets.
        """
        # Get (anchor, positive, negative) triplets from the selector.
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        # Move triplet indices to GPU if needed.
        if embeddings.is_cuda:
            triplets = triplets.cuda()

        # Squared L2 distance between anchor and positive.
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)
        # Squared L2 distance between anchor and negative.
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)

        # Triplet loss: max(0, d(ap) - d(an) + margin).
        losses = torch.relu(ap_distances - an_distances + self.margin)

        # Return mean loss over all triplets.
        return losses.mean()
