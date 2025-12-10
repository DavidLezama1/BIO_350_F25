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

"""Diversity promoting sampling method that uses graph density to determine
 most representative points.

This is an implementation of the method described in
https://www.mpi-inf.mpg.de/fileadmin/inf/d2/Research_projects_files/EbertCVPR2012.pdf

High-level idea:
  * Build a k-nearest-neighbor graph over all data points.
  * Convert the graph into a weighted graph using a Gaussian kernel on
    distances between neighbors.
  * Define a "graph density" score for each point as the average weight
    of edges to its neighbors.
  * Select points with high density, while down-weighting neighbors of
    already selected points to promote diversity.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
import numpy as np
from active_learning_methods.sampling_def import SamplingMethod


class GraphDensitySampler(SamplingMethod):
  """Diversity promoting sampling method that uses graph density to determine
  most representative points.
  """

  def __init__(self, X, y, seed):
    """Initialize the sampler with data and compute initial graph densities.

    Args:
      X: Feature matrix of shape (n_samples, n_features).
      y: Labels (not directly used here but kept for API consistency).
      seed: Random seed (not used directly in this method).
    """
    # Name used to identify this sampler in the active learning framework.
    self.name = 'graph_density'
    # Store the original feature matrix.
    self.X = X
    # Flatten possibly structured inputs into a 2D array for distance calculations.
    self.flat_X = self.flatten_X()
    # Set gamma for Gaussian kernel to be equal to 1 / n_features
    # This controls how quickly the weight decays with distance.
    self.gamma = 1. / self.X.shape[1]
    # Build the k-NN graph and compute initial graph density scores.
    self.compute_graph_density()

  def compute_graph_density(self, n_neighbor=10):
    """Construct the k-NN graph and compute graph-density scores.

    Args:
      n_neighbor: Number of nearest neighbors to use for the k-NN graph.
    """
    # kneighbors graph is constructed using k=10
    # p=1 specifies the Manhattan (L1) distance for neighborhood computation.
    connect = kneighbors_graph(self.flat_X, n_neighbor, p=1)
    # Make connectivity matrix symmetric, if a point is a k nearest neighbor of
    # another point, make it vice versa
    neighbors = connect.nonzero()
    # Pair up row and column indices of non-zero entries to iterate over edges.
    inds = zip(neighbors[0], neighbors[1])
    # Graph edges are weighted by applying gaussian kernel to manhattan dist.
    # By default, gamma for rbf kernel is equal to 1/n_features but may
    # get better results if gamma is tuned.
    for entry in inds:
      i = entry[0]
      j = entry[1]
      # Compute the Manhattan distance between points i and j.
      distance = pairwise_distances(self.flat_X[[i]], self.flat_X[[j]], metric='manhattan')
      distance = distance[0, 0]
      # Convert distance into a similarity via a Gaussian (RBF) kernel.
      weight = np.exp(-distance * self.gamma)
      # Set symmetric weights for the undirected edge (i, j).
      connect[i, j] = weight
      connect[j, i] = weight
    # Store the weighted connectivity matrix for later use in selection.
    self.connect = connect
    # Define graph density for an observation to be sum of weights for all
    # edges to the node representing the datapoint.  Normalize sum weights
    # by total number of neighbors.
    self.graph_density = np.zeros(self.X.shape[0])
    for i in np.arange(self.X.shape[0]):
      # Sum all edge weights for node i and divide by the number of neighbors.
      self.graph_density[i] = connect[i, :].sum() / (connect[i, :] > 0).sum()
    # Keep a copy of the initial graph densities for analysis/visualization.
    self.starting_density = copy.deepcopy(self.graph_density)

  def select_batch_(self, N, already_selected, **kwargs):
    """Select a batch of points using graph density with diversity adjustment.

    Args:
      N: Number of points to select in this batch.
      already_selected: Indices of points that have already been selected.
      **kwargs: Additional arguments for compatibility (unused here).

    Returns:
      A list of indices corresponding to the selected batch.
    """
    # If a neighbor has already been sampled, reduce the graph density
    # for its direct neighbors to promote diversity.
    batch = set()
    # Force graph density of already-selected points to be lower than all others
    # so they are never selected again.
    self.graph_density[already_selected] = min(self.graph_density) - 1
    while len(batch) < N:
      # Select the point with the highest current graph density.
      selected = np.argmax(self.graph_density)
      # Find neighbors of the selected point in the graph.
      neighbors = (self.connect[selected, :] > 0).nonzero()[1]
      # Reduce density of neighbors to discourage picking similar points.
      self.graph_density[neighbors] = self.graph_density[neighbors] - self.graph_density[selected]
      # Add selected point to the batch.
      batch.add(selected)
      # Ensure already-selected and current batch members are never picked again.
      self.graph_density[already_selected] = min(self.graph_density) - 1
      self.graph_density[list(batch)] = min(self.graph_density) - 1
    # Return the selected indices as a list.
    return list(batch)

  def to_dict(self):
    """Return diagnostic information about the graph and initial densities.

    Returns:
      A dictionary with:
        - 'connectivity': the weighted adjacency matrix of the graph.
        - 'graph_density': the original (unmodified) graph density scores.
    """
    output = {}
    output['connectivity'] = self.connect
    output['graph_density'] = self.starting_density
    return output
