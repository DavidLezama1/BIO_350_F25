import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import BasicBlock
import torch.utils.model_zoo as model_zoo
import torch
import torch.nn.functional as F


class EmbeddingNet(nn.Module):
    """Backbone network that turns images into fixed-size embeddings.

    This wraps a torchvision classification architecture (e.g., ResNet, VGG, DenseNet)
    and replaces the final classification layer with a linear layer that outputs
    `feat_dim`-dimensional feature vectors.

    These embeddings are used later for metric learning (contrastive/triplet loss)
    and active learning.
    """

    def __init__(self, architecture, feat_dim, use_pretrained=False):
        """
        Args:
            architecture (str): Name of a torchvision model, e.g. 'resnet18', 'vgg16'.
            feat_dim (int): Size of the output embedding vector.
            use_pretrained (bool): If True, load ImageNet-pretrained weights.
        """
        super(EmbeddingNet, self).__init__()
        self.feat_dim = feat_dim

        # Instantiate the chosen architecture from torchvision's model zoo.
        self.inner_model = models.__dict__[architecture](pretrained=use_pretrained)

        # For ResNet-style models, replace the final fully-connected layer.
        if architecture.startswith('resnet'):
            in_feats = self.inner_model.fc.in_features
            self.inner_model.fc = nn.Linear(in_feats, feat_dim)

        # For Inception-style models, replace the final fully-connected layer.
        elif architecture.startswith('inception'):
            in_feats = self.inner_model.fc.in_features
            self.inner_model.fc = nn.Linear(in_feats, feat_dim)

        # For DenseNet models, the classifier attribute holds the final layer.
        if architecture.startswith('densenet'):
            in_feats = self.inner_model.classifier.in_features
            self.inner_model.classifier = nn.Linear(in_feats, feat_dim)

        # For VGG models, the last classifier layer is at index '6'.
        if architecture.startswith('vgg'):
            in_feats = self.inner_model.classifier._modules['6'].in_features
            self.inner_model.classifier._modules['6'] = nn.Linear(in_feats, feat_dim)

        # For AlexNet, the last classifier layer is also at index '6'.
        if architecture.startswith('alexnet'):
            in_feats = self.inner_model.classifier._modules['6'].in_features
            self.inner_model.classifier._modules['6'] = nn.Linear(in_feats, feat_dim)

    def forward(self, x):
        """Run a forward pass through the underlying backbone model.

        Args:
            x (Tensor): Input image batch of shape (N, C, H, W).

        Returns:
            Tensor: Embedding batch of shape (N, feat_dim).
        """
        return self.inner_model.forward(x)


class NormalizedEmbeddingNet(EmbeddingNet):
    """Embedding network that returns the same tensor as both logits and embedding.

    This is used in parts of the codebase where the model is expected to return
    a (logits, embedding) tuple. Here we simply use the embedding as both outputs.
    """

    def __init__(self, architecture, feat_dim, use_pretrained=False):
        """
        Args:
            architecture (str): Name of a torchvision model.
            feat_dim (int): Size of the output embedding vector.
            use_pretrained (bool): If True, load ImageNet-pretrained weights.
        """
        # Call the parent class constructor directly.
        EmbeddingNet.__init__(self, architecture, feat_dim, use_pretrained=use_pretrained)

    def forward(self, x):
        """Forward pass that returns (embedding, embedding).

        Args:
            x (Tensor): Input image batch.

        Returns:
            Tuple[Tensor, Tensor]: (embedding, embedding), where both elements
                                   are the same feature tensor.
        """
        embedding = self.inner_model.forward(x)
        return embedding, embedding


class SoftmaxNet(nn.Module):
    """Classifier network that adds a softmax head on top of an EmbeddingNet.

    The model first computes an embedding and then passes it through a linear
    classifier with ReLU activation for standard supervised classification.
    """

    def __init__(self, architecture, feat_dim, num_classes, use_pretrained=False):
        """
        Args:
            architecture (str): Name of a torchvision backbone (e.g. 'resnet18').
            feat_dim (int): Dimension of the embedding layer.
            num_classes (int): Number of output classes for classification.
            use_pretrained (bool): If True, use ImageNet-pretrained backbone weights.
        """
        super(SoftmaxNet, self).__init__()
        # Reuse the embedding network as the feature extractor.
        self.embedding = EmbeddingNet(architecture, feat_dim, use_pretrained=use_pretrained)
        # Final classification layer mapping embeddings to class logits.
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        """Forward pass that returns both logits and the underlying embedding.

        Args:
            x (Tensor): Input image batch.

        Returns:
            Tuple[Tensor, Tensor]:
                - x: Logits of shape (N, num_classes).
                - embed: Embeddings of shape (N, feat_dim).
        """
        # Get embedding from the backbone.
        embed = self.embedding(x)
        # Apply non-linearity before the classifier.
        x = F.relu(embed)
        # Compute class logits.
        x = self.classifier(x)
        return x, embed
