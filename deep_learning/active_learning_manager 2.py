from .engine import Engine
import numpy as np
import torch


class ActiveLearningManager(object):
    """Helper class to manage pools and embeddings for active learning.

    Responsibilities:
      * Keep track of:
          - default_pool: all dataset indices (initial unlabeled pool)
          - active_pool: indices that have been selected / labeled
          - current_pool: which pool is currently "active" ('default' or 'active')
      * Maintain and update an embedding model using the Engine class.
      * Provide convenient accessors for train/test sets in embedding space.
      * Fine-tune the embedding model on the actively labeled pool.
    """

    # constructor
    def __init__(self, dataset, embedding_model, device, criterion, normalize):
        """Initialize the manager with dataset and embedding model.

        Args:
          dataset: object that implements getSingleLoader and getBalancedLoader
                   and exposes .samples and .indices.
          embedding_model: torch.nn.Module used to compute embeddings.
          device: torch device where the model and tensors live (e.g., 'cuda').
          criterion: loss function used during fine-tuning.
          normalize: bool flag passed to Engine.embedding to normalize outputs.
        """
        self.dataset = dataset
        # Default pool initially contains all indices from the dataset.
        self.default_pool = list(range(len(dataset)))
        # Active pool will store indices that have been selected / labeled.
        self.active_pool = []
        # current_pool points to whichever pool we want to operate on.
        self.current_pool = self.default_pool
        # Embedding model (typically a deep network).
        self.model = embedding_model
        # Cached embeddings for all dataset items (filled by updateEmbedding).
        self.embedding = None
        self.device = device
        self.criterion = criterion

        # Optimizer for fine-tuning the embedding model.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

        # Engine wraps training, validation, and embedding extraction logic.
        self.engine = Engine(self.device, self.model, criterion, optimizer)

        # Whether to L2-normalize embeddings when they are extracted.
        self.normalize = normalize

    # update embedding values after a finetuning
    def updateEmbedding(self, batch_size=128, num_workers=4):
        """Recompute and cache embeddings for all items in the dataset.

        This should be called after the embedding model is fine-tuned, so that
        self.embedding reflects the current model state.
        """
        print('Extracting embedding from the provided model ...')
        # Build a DataLoader over the whole dataset (no shuffling, val transform).
        loader = self.dataset.getSingleLoader(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            sub_indices=list(range(len(self.dataset))),
            transfm="val"
        )
        # Compute embeddings for all samples and store them.
        self.embedding = self.engine.embedding(
            loader,
            normalize=self.normalize
        )

    # select either the default or active pools
    def setPool(self, pool):
        """Switch current_pool between 'default' and 'active'."""
        assert pool in ["default", "active"]
        if pool == 'default':
            self.current_pool = self.default_pool
        else:
            self.current_pool = self.active_pool

    # gather test set
    def getTestSet(self):
        """Return embedding and labels for the default pool (often used as test).

        Returns:
          (X_test, y_test):
            - X_test: embeddings for indices in default_pool
            - y_test: corresponding integer labels from the dataset
        """
        return (
            self.embedding[self.default_pool],
            np.asarray([
                self.dataset.samples[self.dataset.indices[x]][1]
                for x in self.default_pool
            ])
        )

    # gather train set
    def getTrainSet(self):
        """Return embedding and labels for the active (labeled) pool.

        Returns:
          (X_train, y_train):
            - X_train: embeddings for indices in active_pool
            - y_train: corresponding integer labels from the dataset
        """
        return (
            self.embedding[self.active_pool],
            np.asarray([
                self.dataset.samples[self.dataset.indices[x]][1]
                for x in self.active_pool
            ])
        )

    # finetune the embedding model over the labeled pool
    def finetune_embedding(self, epochs, P, K, lr, num_workers=10):
        """Fine-tune the embedding model using only the active pool.

        Args:
          epochs: number of fine-tuning epochs.
          P: number of classes per batch (for P×K sampling).
          K: number of samples per class in each batch.
          lr: learning rate (currently not used; optimizer is fixed in __init__).
          num_workers: DataLoader worker processes.
        """
        # Build a balanced loader from the currently labeled active pool.
        train_loader = self.dataset.getBalancedLoader(
            P=P,
            K=K,
            num_workers=num_workers,
            sub_indices=self.active_pool
        )
        # Run several training epochs over the labeled pool.
        for epoch in range(epochs):
            self.engine.train_one_epoch(train_loader, epoch, False)

    # a utility function for saving the snapshot
    def get_pools(self):
        """Return a snapshot of pools and embeddings for logging or saving.

        Returns:
          A dict containing:
            - 'embedding': current embedding matrix for all samples.
            - 'active_indices': list of indices in active_pool.
            - 'default_indices': list of indices in default_pool.
            - 'active_pool': list of (path, label) tuples for active_pool.
            - 'default_pool': list of (path, label) tuples for default_pool.
        """
        return {
            "embedding": self.embedding,
            "active_indices": self.active_pool,
            "default_indices": self.default_pool,
            "active_pool": [
                self.dataset.samples[self.dataset.indices[x]]
                for x in self.active_pool
            ],
            "default_pool": [
                self.dataset.samples[self.dataset.indices[x]]
                for x in self.default_pool
            ]
        }
