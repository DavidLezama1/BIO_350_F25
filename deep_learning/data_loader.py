from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (RandomCrop, RandomErasing, 
CenterCrop, ColorJitter, RandomRotation, RandomHorizontalFlip, RandomOrder,
Normalize, Resize, Compose, ToTensor, RandomGrayscale)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import numpy as np
import os
import random
from PIL import ImageStat
from .engine import Engine


class ExtendedImageFolder(ImageFolder):
    """Extension of torchvision.datasets.ImageFolder with:

      * Cached dataset mean/std computation for normalization.
      * Support for using only a subset of indices.
      * Built-in train/val data augmentations and transforms.
      * Convenience loaders for balanced and single-batch sampling.
    """

    def __init__(self, base_folder, indices = None):
        """Initialize dataset from a base folder and optional subset of indices.

        Args:
          base_folder: path to the root directory (ImageFolder-style structure).
          indices: controls which samples to include:
            - None: use all samples.
            - list: use exactly these indices.
            - int: randomly sample that many indices from the full dataset.
        """
        super().__init__(base_folder)
        self.base_folder = base_folder
        # Compute or load cached per-channel mean and std for normalization.
        self.mean, self.std = self.calc_mean_std()

        # Determine which subset of the full dataset to use.
        if indices is None:
            # Use all available indices.
            self.indices = list(range(len(self.samples)))
        elif isinstance(indices, list):
            # Use only the given subset.
            self.indices = indices
        elif isinstance(indices, int):
            # Randomly sample 'indices' many items from the full dataset.
            assert indices <= len(self.samples)
            self.indices = random.sample(range(len(self.samples)), indices)
        else:
            raise TypeError('Invalid type for indices')

        # Pre-build transformations for train and validation modes.
        self.trnsfm = {}
        self.trnsfm['train'] = self.get_transform('train')
        self.trnsfm['val'] = self.get_transform('val')

    def setTransform(self, transform):
        """Select which transform ('train' or 'val') to apply in __getitem__."""
        assert transform in ["train", "val"]
        self.transform = self.trnsfm[transform]

    def __len__(self):
        """Return the number of samples in the active subset."""
        return len(self.indices)

    def __getitem__(self, ind):
        """
        Fetch a single sample from the subset.

        Args:
          ind (int): index into self.indices (not the original samples list).

        Returns:
          tuple: (sample_tensor, target_label, path)
        """
        # Map subset index back to the original ImageFolder sample index.
        index = self.indices[ind]
        path, target = self.samples[index]
        # Load the image using ImageFolder's loader.
        sample = self.loader(path)
        # Apply input transform (train or val) if defined.
        if self.transform is not None:
            sample = self.transform(sample)
        # Apply optional target transform.
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Return image tensor, numeric label, and original file path.
        return sample, target, path

    def get_transform(self, trns_mode):
        """Build a torchvision transform pipeline for train/val modes.

        Pipeline:
          * Resize to 256x256.
          * Train:
              - RandomCrop(224x224), RandomGrayscale, random order of
                flip/jitter/rotation, RandomErasing.
            Val:
              - CenterCrop(224x224).
          * Convert to tensor.
          * Normalize with dataset mean and std.

        Args:
          trns_mode: 'train' or 'val'.

        Returns:
          A composed transform (torchvision.transforms.Compose).
        """
        transform_list = []
        # Resize all images to a canonical size first.
        transform_list.append(Resize((256, 256)))
        if trns_mode == 'train':
            # Train-time augmentations to improve generalization.
            transform_list.append(RandomCrop((224, 224)))
            transform_list.append(RandomGrayscale())
            transform_list.append(RandomOrder(
                [RandomHorizontalFlip(), ColorJitter(), RandomRotation(20)]))
        else:
            # Validation: deterministic center crop.
            transform_list.append(CenterCrop((224, 224)))
        # Convert PIL image to tensor.
        transform_list.append(ToTensor())
        # Normalize using dataset-specific mean and std.
        transform_list.append(Normalize(self.mean, self.std))
        if trns_mode == 'train':
            # Further regularization: randomly erase parts of the image.
            transform_list.append(RandomErasing(value='random'))

        return Compose(transform_list)

    def calc_mean_std(self):
        """Compute or load per-channel mean and std for normalization.

        The result is cached in a file ".meanstd.cache" under base_folder
        so subsequent runs can load it quickly without recomputing.

        Returns:
          (means, stds): two 1D NumPy arrays of length 3 (RGB channels).
        """
        cache_file = os.path.join(self.base_folder, ".meanstd.cache")
        if not os.path.exists(cache_file):
            print("Calculating Mean and Std")
            means = np.zeros((3))
            stds = np.zeros((3))
            # To limit runtime, only sample up to 10,000 images.
            sample_size = min(len(self.samples), 10000)
            for i in range(sample_size):
                # Randomly pick an image path from the dataset.
                img = self.loader(random.choice(self.samples)[0])
                # Compute basic statistics (mean and stddev of pixel values).
                stat = ImageStat.Stat(img)
                means += np.array(stat.mean)/255.0
                stds += np.array(stat.stddev)/255.0
            # Average over the sampled images.
            means = means/sample_size
            stds = stds/sample_size
            # Save to cache for reuse.
            np.savetxt(cache_file, np.vstack((means, stds)))
        else:
            print("Load Mean and Std from "+cache_file)
            # Load cached mean/std from disk.
            contents = np.loadtxt(cache_file)
            means = contents[0, :]
            stds = contents[1, :]

        return means, stds

    def getClassesInfo(self):
        """Return class names and class-to-index mapping."""
        return self.classes, self.class_to_idx

    def getBalancedLoader(self, P= 10, K= 10, num_workers = 4, sub_indices= None, transfm = 'train'):
        """Get a DataLoader that yields class-balanced batches (P×K sampling).

        Args:
          P: number of distinct classes per batch.
          K: number of samples per class per batch.
          num_workers: number of worker processes for DataLoader.
          sub_indices: optional subset of indices to restrict the sampler to.
          transfm: 'train' or 'val' transform pipeline.

        Returns:
          DataLoader that uses BalancedBatchSampler to generate batches of
          size P*K where each batch is balanced across P different classes.
        """
        # Choose the appropriate transform (train or val).
        self.setTransform(transfm)
        if sub_indices is not None:
            # Use only a subset of the dataset (e.g., active pool).
            subset = Subset(self, sub_indices)
            train_batch_sampler = BalancedBatchSampler(
                subset, n_classes=P, n_samples=K
            )
            return DataLoader(subset, batch_sampler=train_batch_sampler,
                              num_workers=num_workers)
        # Otherwise, use the full dataset.
        train_batch_sampler = BalancedBatchSampler(self, n_classes=P, n_samples=K)
        return DataLoader(self, batch_sampler=train_batch_sampler,
                          num_workers=num_workers)

    def getSingleLoader(self, batch_size = 128, shuffle = True, num_workers = 4, sub_indices= None, transfm = 'train'):
        """Get a standard DataLoader over the dataset or a subset.

        Args:
          batch_size: number of samples per batch.
          shuffle: whether to shuffle data at each epoch.
          num_workers: number of worker processes for DataLoader.
          sub_indices: optional subset to restrict the loader to.
          transfm: 'train' or 'val' transform pipeline.

        Returns:
          A standard torch.utils.data.DataLoader.
        """
        # Choose transform mode.
        self.setTransform(transfm)
        if sub_indices is not None:
            # DataLoader over a subset of indices.
            return DataLoader(Subset(self, sub_indices),
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers)
        # DataLoader over the entire ExtendedImageFolder.
        return DataLoader(self,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers)


class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples.

    This is typically used for metric learning / deep embedding training:
      * Each batch contains P classes, with K examples per class.
      * Over time, class indices are reshuffled to avoid always using the same
        items for a given class.
    """

    def __init__(self, underlying_dataset, n_classes, n_samples):
        """Initialize balanced sampler from an underlying dataset.

        Args:
          underlying_dataset: dataset or Subset-like object that has
            .samples and .indices attributes in the ImageFolder style.
          n_classes: number of classes to sample per batch (P).
          n_samples: number of samples per class in each batch (K).
        """
        # If the dataset is a Subset, we need to look through underlying_dataset.dataset
        # to get the original samples and indices.
        if hasattr(underlying_dataset, "dataset"):
            self.labels = [
                underlying_dataset.dataset.samples[underlying_dataset.dataset.indices[i]][1]
                for i in underlying_dataset.indices
            ]
        else:
            # Otherwise, we are directly working with something like ExtendedImageFolder.
            self.labels = [underlying_dataset.samples[i][1] for i in underlying_dataset.indices]

        # Unique set of label IDs present in the subset.
        self.labels_set = set(self.labels)
        # Map each label to the array of positions (within the subset) where it appears.
        self.label_to_indices = {
            label: np.where(np.array(self.labels) == label)[0]
            for label in self.labels_set
        }
        # Shuffle the order for each label initially to randomize sampling.
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        # Keep track of how many samples we've used per label so far.
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        # Counter to track how many samples have been yielded overall.
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        # Keep a reference to the underlying dataset for len() computations.
        self.dataset = underlying_dataset
        # Effective batch size: P classes × K samples.
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        """Yield lists of indices forming class-balanced batches."""
        self.count = 0
        # Continue until we cannot form another full batch of size batch_size.
        while self.count + self.batch_size < len(self.dataset):
            #print(self.labels_set, self.n_classes)
            # Randomly choose n_classes distinct labels for this batch.
            classes = np.random.choice(
                list(self.labels_set), self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                # Take the next n_samples indices for this class.
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                   class_] + self.n_samples])
                # Update the pointer for how many indices we've used for this class.
                self.used_label_indices_count[class_] += self.n_samples
                # If we've exhausted this class's indices for another full block
                # of n_samples, reshuffle and reset the counter to reuse them.
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            # Yield one balanced batch of indices.
            yield indices
            # Update overall sample counter.
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        """Number of full balanced batches that can be drawn from the dataset."""
        return len(self.dataset) // (self.n_samples*self.n_classes)
