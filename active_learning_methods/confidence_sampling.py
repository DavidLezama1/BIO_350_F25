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


class ConfidenceAL(SamplingMethod):
  # Active learning sampler that selects the least confident points.
  # It uses predicted class probabilities from a trained model and
  # prefers points where the model's maximum class probability is small
  # (i.e., where the model is most uncertain).

  def __init__(self, X, y, seed):
    # Store the full feature matrix. X is typically a 2D array of shape
    # (n_samples, n_features).
    self.X = X
    # Store labels if needed by the parent class or other methods.
    self.y = y
    # Name used by the broader active learning framework to identify this method.
    self.name = 'confidence'

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
    # Use the trained model to get predicted class probabilities for all points.
    # Shape: (n_samples, n_classes).
    probs = model.predict_proba(self.X) 
    # For each sample, take the maximum probability across classes.
    # A value close to 1 means the model is confident; lower values mean higher uncertainty.
    uncertainty= probs.max(axis=1)
    # Sort indices by this "confidence" score in ascending order so that
    # the least confident (most uncertain) samples come first.
    srt = np.argsort(uncertainty)
    # This list will store the selected sample indices.
    indices = []
    # Pointer into the sorted index list.
    i = 0
    # Continue until we have selected N new points.
    while len(indices) < N:
        # Only add a point if it has not already been selected previously.
        if srt[i] not in already_selected: 
            indices.append(srt[i])
        # Move to the next candidate in the sorted list.
        i += 1
    # Return the final list of indices to be labeled next.
    return indices
