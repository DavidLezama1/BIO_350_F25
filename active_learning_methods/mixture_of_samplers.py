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

"""Mixture of base sampling strategies

This module defines a sampler that combines several base Active Learning
strategies (e.g., margin, uniform, entropy) according to specified mixture
weights. It calls each base strategy to propose points and then merges the
results into a single batch.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from active_learning_methods.sampling_def import SamplingMethod
from active_learning_methods.constants import AL_MAPPING, get_base_AL_mapping

# Populate AL_MAPPING with all base samplers so we can construct them by name.
get_base_AL_mapping()


class MixtureOfSamplers(SamplingMethod):
  """Samples according to mixture of base sampling methods.

  If duplicate points are selected by the mixed strategies when forming the batch
  then the remaining slots are divided according to mixture weights and
  another partial batch is requested until the batch is full.
  """
  def __init__(self,
               X,
               y,
               seed,
               mixture={'methods': ('margin', 'uniform'),
                        'weight': (0.5, 0.5)},
               samplers=None):
    # Store feature matrix and labels.
    self.X = X
    self.y = y
    # Name used in the AL framework to identify this sampler.
    self.name = 'mixture_of_samplers'
    # List of base sampler names to mix (e.g., ['margin', 'uniform']).
    self.sampling_methods = mixture['methods']
    # Map from sampler name to its mixture weight.
    # The 'weights' entry in `mixture` specifies the relative proportions.
    self.sampling_weights = dict(zip(mixture['methods'], mixture['weights']))
    # Random seed (passed down to base samplers).
    self.seed = seed
    # A list/dict of initialized samplers is allowed as an input because
    # for AL_methods that search over different mixtures, may want mixtures to
    # have shared AL_methods so that initialization is only performed once for
    # computation intensive methods like HierarchicalClusteringAL and
    # states are shared between mixtures.
    # If initialized samplers are not provided, initialize them ourselves.
    if samplers is None:
      # Dictionary mapping sampler name -> sampler instance.
      self.samplers = {}
      self.initialize(self.sampling_methods)
    else:
      # Use provided samplers (sharing state between mixtures).
      self.samplers = samplers
    # History of which points each sampler suggested across calls.
    self.history = []

  def initialize(self, samplers):
    """Instantiate base samplers listed in `samplers` using AL_MAPPING.

    Args:
      samplers: iterable of sampler names (keys in AL_MAPPING).
    """
    self.samplers = {}
    for s in samplers:
      # Construct each sampler with the same (X, y, seed).
      self.samplers[s] = AL_MAPPING[s](self.X, self.y, self.seed)

  def select_batch_(self, already_selected, N, **kwargs):
    """Returns batch of datapoints selected according to mixture weights.

    The algorithm:
      * Keeps track of the set of selected indices `inds`.
      * Repeatedly:
          - Computes an "effective_N" that grows as we fail to fill the batch.
          - For each sampler, computes a sub-batch size proportional to its
            mixture weight and effective_N.
          - Calls sampler.select_batch(...) for that sampler.
          - Adds any novel (non-duplicate) indices to `inds`.
      * Continues until we have N unique indices.

    Args:
      already_included: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """
    # Make sure already_selected does not get modified by downstream calls.
    kwargs['already_selected'] = copy.copy(already_selected)
    # Set of final indices to be returned (avoid duplicates).
    inds = set()
    # Track which indices each sampler selected in this iteration.
    self.selected_by_sampler = {}
    for s in self.sampling_methods:
      self.selected_by_sampler[s] = []
    # effective_N increases if we still do not have enough unique points.
    effective_N = 0
    while len(inds) < N:
      # Each loop iteration we increase the "budget" effective_N by how many
      # points are still missing from the final batch.
      effective_N += N - len(inds)
      for s in self.sampling_methods:
        if len(inds) < N:
          # Compute batch size for this sampler based on its weight.
          # Enforce at least 1 and at most N.
          batch_size = min(max(int(self.sampling_weights[s] * effective_N), 1), N)
          sampler = self.samplers[s]
          # Pass requested batch size into the underlying sampler.
          kwargs['N'] = batch_size
          # Ask this base sampler for candidate indices.
          s_inds = sampler.select_batch(**kwargs)
          # Record all indices that this sampler ever selected (per iteration).
          for ind in s_inds:
            if ind not in self.selected_by_sampler[s]:
              self.selected_by_sampler[s].append(ind)
          # Filter out indices that are already in the combined set `inds`.
          s_inds = [d for d in s_inds if d not in inds]
          # Truncate to the number of remaining slots in the final batch.
          s_inds = s_inds[0 : min(len(s_inds), N-len(inds))]
          # Add these new unique indices to the global set.
          inds.update(s_inds)
    # Store a deep copy of the per-sampler selections for analysis later.
    self.history.append(copy.deepcopy(self.selected_by_sampler))
    # Return final list of selected indices.
    return list(inds)

  def to_dict(self):
    """Export mixture configuration and per-sampler details as a dictionary.

    Returns:
      A dictionary containing:
        - 'history': list of per-sampler selections over time.
        - 'samplers': list of sampler names in the mixture.
        - 'mixture_weights': dict of sampler_name -> weight.
        - For each sampler name s: s_output = samplers[s].to_dict().
    """
    output = {}
    output['history'] = self.history
    output['samplers'] = self.sampling_methods
    output['mixture_weights'] = self.sampling_weights
    # Include diagnostic info from each base sampler (if provided).
    for s in self.samplers:
      s_output = self.samplers[s].to_dict()
      output[s] = s_output
    return output
