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
from scipy import stats
from active_learning_methods.sampling_def import SamplingMethod


class EntropyAL(SamplingMethod):
  # Active learning sampler that uses prediction entropy as the
  # uncertainty measure. Higher entropy = more uncertain predictions.

  def __init__(self, X, y, seed):
    # Store the full feature matrix.
    self.X = X
    # Store labels (not directly used in selection, but kept for compatibility).
    self.y = y
    # Name used by the active learning framework to identify this sampler.
    self.name = 'entropy'

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with smallest margin/highest uncertainty.

    For binary classification, can just take the absolute distance to decision
    boundary for each point.
    For multiclass classification, must consider the margin between distance for
    top two most likely classes.

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """

    # Get predicted class probabilities for every example.
    # Shape: (n_samples, n_classes).
    probs = model.predict_proba(self.X) 
    # Compute entropy along the class axis for each sample.
    # Higher entropy means the model is more uncertain.
    entropy= np.apply_along_axis(stats.entropy, 1, probs)
    # Sort indices by entropy in descending order so that
    # the most uncertain points come first.
    srt = np.argsort(entropy)[::-1]
    # List of indices that will be selected this round.
    indices = []
    # Pointer into the sorted array.
    i = 0
    # Keep adding the most uncertain points until we have N new samples.
    while len(indices) < N:
        # Skip points that were already selected in previous AL rounds.
        if srt[i] not in already_selected: 
            indices.append(srt[i])
        # Move to the next most uncertain point.
        i += 1
    # Return indices of selected points.
    return indices
