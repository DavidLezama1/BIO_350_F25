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

"""Another informative and diverse sampler that mirrors the algorithm described
in Xu, et. al., Representative Sampling for Text Classification Using 
Support Vector Machines, 2003

Batch is created by clustering points within the margin of the classifier and 
choosing points closest to the k centroids.

High-level idea:
  * First, identify unlabeled points that lie within the classifier's margin
    (i.e., where the current model is relatively uncertain).
  * Then, run k-means on just those “within-margin” points.
  * Finally, for each cluster, select the point closest to the cluster center.
  * This yields an informative (uncertain) and diverse (spread-out) batch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.cluster import MiniBatchKMeans
import numpy as np
from active_learning_methods.sampling_def import SamplingMethod


class RepresentativeClusterMeanSampling(SamplingMethod):
  """Selects batch based on informative and diverse criteria.

    Returns points within the margin of the classifier that are closest to the
    k-means centers of those points.  
  """

  def __init__(self, X, y, seed):
    # Name used by the active learning framework.
    self.name = 'cluster_mean'
    # Store the original feature matrix.
    self.X = X
    # Flatten features to a 2D array for clustering and distance operations.
    self.flat_X = self.flatten_X()
    # Store labels (not directly used in selection, but kept for API consistency).
    self.y = y
    # Random seed (not used directly here, but may be used by callers).
    self.seed = seed

  def select_batch_(self, model, N, already_selected, **kwargs):
    """Select a batch of size N using margin-based clustering.

    Steps:
      1. Compute per-sample margins (uncertainty) from the model.
      2. Identify unlabeled points that lie within the margin region.
      3. Cluster those points using k-means into N clusters.
      4. Select the point closest to each cluster center.
      5. If there are not enough within-margin points, fall back to
         simple uncertainty sampling (just smallest margins).
    """
    # Probably okay to always use MiniBatchKMeans
    # Should standardize data before clustering
    # Can cluster on standardized data but train on raw features if desired

    # Get model distances:
    #   - For many models, decision_function gives signed scores or distances.
    #   - If not available, use predicted probabilities instead.
    try:
      distances = model.decision_function(self.X)
    except:
      distances = model.predict_proba(self.X)

    # Compute margin for each point.
    # Binary: margin is absolute distance to the decision boundary.
    # Multiclass: margin is difference between best and second-best scores.
    if len(distances.shape) < 2:
      min_margin = abs(distances)
    else:
      sort_distances = np.sort(distances, 1)[:, -2:]
      min_margin = sort_distances[:, 1] - sort_distances[:, 0]

    # Rank all indices by margin (ascending: smallest margin = most uncertain).
    rank_ind = np.argsort(min_margin)
    # Remove points that were already selected in previous iterations.
    rank_ind = [i for i in rank_ind if i not in already_selected]

    # Take absolute value of distances so we can compare magnitudes.
    distances = abs(distances)
    # Compute, for each class, the minimum margin among already-selected points.
    # This defines a per-class margin threshold.
    min_margin_by_class = np.min(abs(distances[already_selected]),axis=0)

    # Identify unlabeled points that are "within the margin" of the classifier:
    # i.e., whose distance to at least one class is smaller than the threshold.
    unlabeled_in_margin = np.array([i for i in range(len(self.y))
                                    if i not in already_selected and
                                    any(distances[i]<min_margin_by_class)])

    # If there are fewer within-margin points than N, we can't form N clusters.
    # In that case, fall back to plain uncertainty sampling.
    if len(unlabeled_in_margin) < N:
      print("Not enough points within margin of classifier, using simple uncertainty sampling")
      return rank_ind[0:N]

    # Otherwise, cluster the within-margin points into N clusters.
    clustering_model = MiniBatchKMeans(n_clusters=N)
    # dist_to_centroid[i, k] = distance from point i to centroid k.
    dist_to_centroid = clustering_model.fit_transform(self.flat_X[unlabeled_in_margin])

    # For each centroid k, find the index of the nearest point (medoid).
    medoids = np.argmin(dist_to_centroid,axis=0)
    # Ensure unique medoid indices in case of ties or duplicates.
    medoids = list(set(medoids))
    # Map medoid positions back to original dataset indices.
    selected_indices = unlabeled_in_margin[medoids]

    # Sort selected indices by margin (most uncertain first).
    selected_indices = sorted(selected_indices,key=lambda x: min_margin[x])

    # If we selected fewer than N medoids (due to duplicates), fill the
    # remainder of the batch with the next most uncertain points.
    remaining = [i for i in rank_ind if i not in selected_indices]
    selected_indices.extend(remaining[0:N-len(selected_indices)])

    # Return final batch of representative, informative points.
    return selected_indices
