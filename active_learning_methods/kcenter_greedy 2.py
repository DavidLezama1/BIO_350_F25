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

"""Returns points that minimizes the maximum distance of any point to a center.

Implements the k-Center-Greedy method in
Ozan Sener and Silvio Savarese.  A Geometric Approach to Active Learning for
Convolutional Neural Networks. https://arxiv.org/abs/1708.00489 2017

Distance metric defaults to l2 distance.  Features used to calculate distance
are either raw features or if a model has transform method then uses the output
of model.transform(X).

Can be extended to a robust k centers algorithm that ignores a certain number of
outlier datapoints.  Resulting centers are solution to multiple integer program.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.metrics import pairwise_distances
from active_learning_methods.sampling_def import SamplingMethod


class kCenterGreedy(SamplingMethod):
  """k-Center Greedy diversity sampler.

  This sampler chooses points so that the maximum distance between any
  data point and its nearest selected point (center) is minimized.
  It approximates the k-center problem using a greedy algorithm, which
  tends to pick well-separated points and therefore promotes diversity.
  """

  def __init__(self, X, y, seed, metric='euclidean'):
    # Store original feature matrix and labels.
    self.X = X
    self.y = y
    # Flatten X if needed so distance computation works on 2D arrays.
    self.flat_X = self.flatten_X()
    # Name used by the active learning framework.
    self.name = 'kcenter'
    # Features used for distance computation; may later be replaced by
    # model-transformed features in select_batch_.
    self.features = self.flat_X
    # Distance metric used for pairwise_distances (default: Euclidean).
    self.metric = metric
    # min_distances[i] = distance from point i to its closest selected center.
    self.min_distances = None
    # Total number of observations in the pool.
    self.n_obs = self.X.shape[0]
    # List of indices that have already been selected as part of previous batches.
    self.already_selected = []

  def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
    """Update min distances given cluster centers.

    Args:
      cluster_centers: indices of cluster centers
      only_new: only calculate distance for newly selected points and update
        min_distances.
      rest_dist: whether to reset min_distances.
    """

    # Optionally clear all stored distances before recomputing.
    if reset_dist:
      self.min_distances = None
    # If only_new is True, filter out centers that were already used before.
    if only_new:
      cluster_centers = [d for d in cluster_centers
                         if d not in self.already_selected]
    if cluster_centers:
      # Update min_distances for all examples given new cluster center.
      # x contains the feature vectors of the current centers.
      x = self.features[cluster_centers]
      # dist[i, j] = distance between point i and new center j.
      dist = pairwise_distances(self.features, x, metric=self.metric)

      if self.min_distances is None:
        # First time: min distance is distance to the closest new center.
        self.min_distances = np.min(dist, axis=1).reshape(-1,1)
      else:
        # Subsequent times: take the elementwise minimum with existing distances.
        self.min_distances = np.minimum(self.min_distances, dist)

  def select_batch_(self, model, already_selected, N, **kwargs):
    """
    Diversity promoting active learning method that greedily forms a batch
    to minimize the maximum distance to a cluster center among all unlabeled
    datapoints.

    Args:
      model: model with scikit-like API with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to minimize distance to cluster centers
    """

    try:
      # Assumes that the transform function takes in original data and not
      # flattened data.
      # If the model provides a transform method (e.g., deep features),
      # use those features to measure distances in a learned feature space.
      print('Getting transformed features...')
      self.features = model.transform(self.X)
      print('Calculating distances...')
      # Recompute distances from all points to the already selected centers.
      self.update_distances(already_selected, only_new=False, reset_dist=True)
    except:
      # If the model has no transform method, fall back to raw (flattened) features.
      print('Using flat_X as features.')
      self.update_distances(already_selected, only_new=True, reset_dist=False)

    new_batch = []

    # Greedily select N new points.
    for _ in range(N):
      if self.already_selected is None:
        # Initialize centers with a randomly selected datapoint
        ind = np.random.choice(np.arange(self.n_obs))
      else:
        # Choose the point that is currently farthest from any center,
        # i.e., the one with the largest min distance.
        ind = np.argmax(self.min_distances)
      # New examples should not be in already selected since those points
      # should have min_distance of zero to a cluster center.
      assert ind not in already_selected

      # After choosing a new center, update min_distances to include it.
      self.update_distances([ind], only_new=True, reset_dist=False)
      new_batch.append(ind)
    # Report the maximum remaining distance after selecting the batch,
    # which indicates how well the centers cover the space.
    print('Maximum distance from cluster centers is %0.2f'
            % max(self.min_distances))

    # Record which points have been selected so far (for future calls).
    self.already_selected = already_selected

    return new_batch
