import numpy as np
import sys
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import torch
import matplotlib.patches as mpatches
import shutil
from itertools import combinations
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from .losses import *

# Fixed categorical color palette used to visualize embeddings by class.
indexcolors =["#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",

        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#E704C4", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#FEFFE6",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",

        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"]


def reduce_dimensionality(X):
    """Reduce high-dimensional embeddings to 2D using t-SNE for visualization."""
    print("Calculating TSNE")
    # Use scikit-learn's TSNE (no MulticoreTSNE needed)
    embedding = TSNE(n_components=2).fit_transform(X)
    return embedding


def save_embedding_plot(name, embedd, labels, info):
  """Create and save a 2D embedding plot with class-colored points and legend.

  Args:
      name (str): output filename for the saved plot (e.g., 'tsne.png').
      embedd (ndarray): high-dimensional embeddings of shape (N, D).
      labels (ndarray): integer class labels for each embedding.
      info (dict): mapping from class name -> class index used to build legend.
  """
  fig = plt.figure(figsize=(10,10))
  # Map integer labels to fixed colors.
  colors= [indexcolors[int(i)] for i in labels.squeeze()]
  # Reduce to 2D for plotting.
  embedding= reduce_dimensionality(embedd)
  # Scatter plot of points in embedding space.
  plt.scatter(embedding[:,0],embedding[:,1], s=1, c= colors)
  # Build legend based on info dict (sorted by value).
  legend_texts= [ x[0] for x in sorted(info.items(), key=lambda kv: kv[1])]
  patches=[]
  for i,label in enumerate(legend_texts):
    patches.append(mpatches.Patch(color=indexcolors[i], label=label))
  plt.legend(handles=patches)
  plt.xlabel('Dim 1', fontsize=12)
  plt.ylabel('Dim 2', fontsize=12)
  plt.grid(True)
  plt.savefig(name)
  plt.close(fig)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Save model training state and optionally keep a separate 'best' checkpoint."""
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def load_checkpoint(filename='model_best.pth.tar'):
    """Load a checkpoint to CPU (safe for machines without GPUs)."""
    return torch.load(filename, map_location=torch.device('cpu'))


class AverageMeter(object):
    """Computes and stores the average and current value.

    Used to track running averages for loss/accuracy over an epoch.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all counters to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Update the running average with a new value.

        Args:
            val (float): new observed value.
            n (int): number of samples this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for specified values of k.

    Args:
        output (Tensor): model logits of shape (N, C).
        target (Tensor): ground-truth labels of shape (N,).
        topk (tuple): e.g. (1,) or (1, 5).

    Returns:
        list[Tensor]: accuracies for each requested k, in percent.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Get indices of top-k predictions along class dimension.
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        # Compare predictions to ground truth.
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # Flatten first k rows and count correct predictions.
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def pdist(vectors):
    """Pairwise squared distance matrix for a batch of vectors.

    Args:
        vectors (Tensor): shape (N, D)

    Returns:
        Tensor: shape (N, N) where entry (i, j) is ||v_i - v_j||^2.
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


def getCriterion(loss_type, strategy, margin):
    """Factory function to create the appropriate loss object.

    Args:
        loss_type (str): 'softmax' or 'siamese' or triplet-based.
        strategy (str): triplet selection strategy (e.g., 'hard_pair', 'hardest').
        margin (float): margin hyperparameter for contrastive/triplet losses.

    Returns:
        torch.nn.Module: an initialized loss object.
    """
    if loss_type.lower() == 'softmax':
        # Standard cross-entropy classification loss.
        return torch.nn.CrossEntropyLoss()
    
    elif loss_type.lower() == 'siamese' and strategy == 'hard_pair':
        # Contrastive loss with hardest negative pair selection.
        return OnlineContrastiveLoss(
                margin, HardestNegativeTripletSelector(margin))

    else:
        # Triplet loss with selectable negative-sample strategy.
        selector = None
        if strategy == 'hardest':
            selector = HardestNegativeTripletSelector(margin)
        elif strategy == 'random':
            selector = RandomNegativeTripletSelector(margin)
        elif strategy == 'semi_hard':
            selector = SemihardNegativeTripletSelector(margin)
        
        return  OnlineTripletLoss(margin, selector)


class PairSelector:
    """
    Base class for pair selectors.

    Implementation should return indices of positive pairs and negative pairs
    that will be passed to compute Contrastive Loss:

        return positive_pairs, negative_pairs
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllPositivePairSelector(PairSelector):
    """
    Discards embeddings and generates all possible pairs given labels.

    If balance is True, negative pairs are a random sample to match the number
    of positive samples.
    """
    def __init__(self, balance=True):
        super(AllPositivePairSelector, self).__init__()
        self.balance = balance

    def get_pairs(self, embeddings, labels):
        # Move labels to CPU and NumPy for easier indexing.
        labels = labels.cpu().data.numpy()
        # Generate all index pairs (i, j), i < j.
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        # Positive pairs: same label.
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        # Negative pairs: different label.
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        if self.balance:
            # Randomly subsample negatives to match number of positives.
            negative_pairs = negative_pairs[torch.randperm(len(negative_pairs))[:len(positive_pairs)]]

        return positive_pairs, negative_pairs


class HardNegativePairSelector(PairSelector):
    """
    Creates all possible positive pairs.

    For negative pairs, selects pairs with smallest distance (hard negatives),
    matching the number of positive pairs.
    """

    def __init__(self, cpu=True):
        super(HardNegativePairSelector, self).__init__()
        self.cpu = cpu

    def get_pairs(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        # Compute pairwise squared distance matrix.
        distance_matrix = pdist(embeddings.data)

        labels = labels.cpu().data.numpy()
        all_pairs = np.array(list(combinations(range(len(labels)), 2)))
        all_pairs = torch.LongTensor(all_pairs)
        positive_pairs = all_pairs[(labels[all_pairs[:, 0]] == labels[all_pairs[:, 1]]).nonzero()]
        negative_pairs = all_pairs[(labels[all_pairs[:, 0]] != labels[all_pairs[:, 1]]).nonzero()]
        # Extract distances for negative pairs.
        negative_distances = distance_matrix[negative_pairs[:, 0], negative_pairs[:, 1]]
        negative_distances = negative_distances.cpu().data.numpy()
        # Select the closest negatives (hardest negatives).
        top_negatives = np.argpartition(negative_distances, len(positive_pairs))[:len(positive_pairs)]
        top_negative_pairs = negative_pairs[torch.LongTensor(top_negatives)]

        return positive_pairs, top_negative_pairs


class TripletSelector:
    """
    Base class for triplet selectors.

    Implementations should return indices of anchors, positives, and negatives:
        return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_pairs(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets.

    This can be extremely large and is usually impractical except for tiny datasets.
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        # For each class, generate all anchor-positive pairs and pair them with each negative.
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for each positive pair.
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    """Return index of hardest negative (max loss), or None if all losses ≤ 0."""
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    """Randomly pick one negative among those with positive loss."""
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    """Pick a semi-hard negative: 0 < loss < margin."""
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes a negative sample according to a selection function.

    Margin should match the margin used in triplet loss.

    negative_selection_fn should take an array of loss_values for a given
    anchor-positive pair and all negative samples, and return a negative index
    for that pair (or None if no valid negative).
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings.data)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                # Triplet loss values for all possible negatives.
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        # Fallback: if no valid triplets were found, add one dummy triplet.
        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)
        #print(triplets.shape[0])
        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): 
    """Factory for FunctionNegativeTripletSelector using hardest negative."""
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=hardest_negative,
                                           cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): 
    """Factory for FunctionNegativeTripletSelector using random hard negative."""
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=random_hard_negative,
                                           cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): 
    """Factory for FunctionNegativeTripletSelector using semi-hard negatives."""
    return FunctionNegativeTripletSelector(margin=margin,
                                           negative_selection_fn=lambda x: semihard_negative(x, margin),
                                           cpu=cpu)


def plot_embedding(embedding, labels=None, title="Embedding"):
    """
    Dummy placeholder for compatibility with the original repository.

    The original repo expected this function but did not include a concrete
    implementation. Your project does not need it for training or active
    learning; this stub simply logs that it was called.

    Args:
        embedding (ndarray or Tensor): 2D or high-dimensional embedding.
        labels (optional): labels corresponding to embedding points.
        title (str): title for a potential plot (unused here).
    """
    print("plot_embedding() was called, but no visualization is implemented.")
    return None
