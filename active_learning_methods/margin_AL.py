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

"""Margin based AL method.

Samples in batches based on margin scores.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from active_learning_methods.sampling_def import SamplingMethod


class MarginAL(SamplingMethod):
  """Active learning sampler based on the margin between class scores.

  Idea:
    * For each unlabeled example, compute its distance to the decision boundary.
    * In binary classification: use |decision_function(x)|.
    * In multiclass: use the difference between the top two class scores.
    * Smaller margin => higher model uncertainty => more informative sample.
  """

  def __init__(self, X, y, seed):
    # Store feature matrix and labels (labels are not directly used here,
    # but kept for compatibility with the SamplingMethod interface).
    self.X = X
    self.y = y
    # Name by which this strategy will be referenced in the AL framework.
    self.name = 'margin'

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with smallest margin/highest uncertainty.

    For binary classification, can just take the absolute distance to decision
    boundary for each point.
    For multiclass classification, must consider the margin between distance for
    top two most likely classes.

    Args:
      model: scikit learn model with decision_function implemented
             (or predict_proba as a fallback)
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """

    try:
      # Preferred: use decision_function, which gives signed distances or
      # unnormalized scores before the final prediction step.
      distances = model.decision_function(self.X)
    except:
      # Fallback: if decision_function is not available, use class probabilities.
      distances = model.predict_proba(self.X)

    # If distances is 1D (binary case), margin is simply the absolute value.
    if len(distances.shape) < 2:
      min_margin = abs(distances)
    else:
      # Multiclass case:
      #   1. Sort scores for each example.
      #   2. Take the two largest scores (last two columns after sorting).
      sort_distances = np.sort(distances, 1)[:, -2:]
      # Margin = top score - second-best score.
      min_margin = sort_distances[:, 1] - sort_distances[:, 0]

    # Rank indices by margin in ascending order:
    # lowest margin => highest uncertainty.
    rank_ind = np.argsort(min_margin)
    # Remove samples that have already been selected in earlier AL rounds.
    rank_ind = [i for i in rank_ind if i not in already_selected]
    # Take the first N most uncertain samples as the active learning batch.
    active_samples = rank_ind[0:N]
    return active_samples
