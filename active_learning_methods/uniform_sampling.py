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

"""Uniform sampling method.

Samples in batches.

High-level idea:
  * This is a baseline active-learning strategy.
  * It ignores model uncertainty and simply picks random points
    from the unlabeled pool.
  * Useful as a control to compare against more sophisticated
    methods like margin-based or entropy-based sampling.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from active_learning_methods.sampling_def import SamplingMethod


class UniformSampling(SamplingMethod):
  """Random (uniform) batch sampler over the unlabeled pool."""

  def __init__(self, X, y, seed):
    # Store data and labels (labels are not used to choose points here).
    self.X = X
    self.y = y
    # Name used in the active-learning framework to refer to this strategy.
    self.name = 'uniform'
    # Fix NumPy random seed for reproducible random batches.
    np.random.seed(seed)

  def select_batch_(self, already_selected, N, **kwargs):
    """Returns batch of randomly sampled datapoints.

    Assumes that data has already been shuffled.

    Args:
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to label
    """

    # This is uniform given the remaining pool but biased wrt the entire pool.

    # Old simple (commented) version that just picks first N from remaining:
    # sample = [i for i in range(self.X.shape[0]) if i not in already_selected]
    # return sample[0:N]

    # New version: randomly permute all indices, then walk through until
    # we have N indices that are not already selected.
    shuffled_indices = np.random.permutation(self.X.shape[0])
    indices = []
    i = 0
    # Keep adding new indices until the batch reaches size N.
    while len(indices) < N:
        # Only include indices that are not in already_selected
        # (so we never re-query the same example).
        if shuffled_indices[i] not in already_selected: 
            indices.append(shuffled_indices[i])
        i += 1
    return indices
