# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Informative and diverse batch sampler that samples points with small margin
while maintaining same distribution over clusters as entire training data.

Batch is created by sorting datapoints by increasing margin and then growing
the batch greedily.  A point is added to the batch if the result batch still
respects the constraint that the cluster distribution of the batch will
match the cluster distribution of the entire training set.

High-level idea:
  * Cluster the dataset into k clusters (k = number of classes).
  * Compute an uncertainty score for each point using the model margin.
  * Rank points from most uncertain (smallest margin) to least.
  * Greedily build a batch by taking uncertain points while ensuring
    that the batch's cluster proportions roughly match the full data.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from active_learning_methods.sampling_def import SamplingMethod


class InformativeClusterDiverseSampler(SamplingMethod):
  """Selects batch based on informative and diverse criteria.

    Returns highest uncertainty lowest margin points while maintaining
    same distribution over clusters as entire dataset.
  """

  def __init__(self, X, y, seed):
    # Name used in the active learning framework.
    self.name = 'informative_and_diverse'
    # Store original feature matrix.
    self.X = X
    # Flatten features into 2D array if needed for clustering.
    self.flat_X = self.flatten_X()
    # y only used for determining how many clusters there should be
    # probably not practical to assume we know # of classes before hand
    # should also probably scale with dimensionality of data
    self.y = y
    # Number of clusters = number of unique labels in y.
    self.n_clusters = len(list(set(y)))
    # MiniBatchKMeans is used for efficient clustering on larger datasets.
    self.cluster_model = MiniBatchKMeans(n_clusters=self.n_clusters)
    # Perform initial clustering and compute cluster statistics.
    self.cluster_data()

  def cluster_data(self):
    """Cluster the data and compute per-cluster proportions."""
    # Probably okay to always use MiniBatchKMeans
    # Should standardize data before clustering
    # Can cluster on standardized data but train on raw features if desired
    self.cluster_model.fit(self.flat_X)
    # unique: cluster ids; counts: number of points in each cluster.
    unique, counts = np.unique(self.cluster_model.labels_, return_counts=True)
    # Empirical probability of each cluster in the full dataset.
    self.cluster_prob = counts/sum(counts)
    # For each point, store its assigned cluster.
    self.cluster_labels = self.cluster_model.labels_

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns a batch of size N using informative and diverse selection.

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """
    # TODO(lishal): have MarginSampler and this share margin function
    try:
      # For many linear models, decision_function gives distance to hyperplane.
      distances = model.decision_function(self.X)
    except:
      # For models without decision_function, fall back to class probabilities.
      distances = model.predict_proba(self.X)
    # If distances is 1D, it's a binary problem; margin is |distance|.
    if len(distances.shape) < 2:
      min_margin = abs(distances)
    else:
      # For multiclass: sort scores and take top two classes.
      sort_distances = np.sort(distances, 1)[:, -2:]
      # Margin = difference between best and second-best scores.
      min_margin = sort_distances[:, 1] - sort_distances[:, 0]
    # Rank indices by margin (ascending): smallest margin = most uncertain.
    rank_ind = np.argsort(min_margin)
    # Remove points that have already been selected in previous iterations.
    rank_ind = [i for i in rank_ind if i not in already_selected]
    # Track how many points from each cluster are in the new batch so far.
    new_batch_cluster_counts = [0 for _ in range(self.n_clusters)]
    # The batch we will construct.
    new_batch = []
    # Greedily grow the batch by adding uncertain points that keep
    # the cluster distribution close to the global distribution.
    for i in rank_ind:
      if len(new_batch) == N:
        break
      label = self.cluster_labels[i]
      # Check whether adding this point would keep this cluster's fraction
      # in the batch below its global probability.
      if new_batch_cluster_counts[label] / N < self.cluster_prob[label]:
        new_batch.append(i)
        new_batch_cluster_counts[label] += 1
    # If the batch is still too small (because of cluster constraints),
    # fill remaining slots with the next most uncertain points.
    n_slot_remaining = N - len(new_batch)
    batch_filler = list(set(rank_ind) - set(already_selected) - set(new_batch))
    new_batch.extend(batch_filler[0:n_slot_remaining])
    return new_batch

  def to_dict(self):
    """Return clustering information for analysis/visualization."""
    output = {}
    # Cluster assignment of each datapoint.
    output['cluster_membership'] = self.cluster_labels
    return output
