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

"""Abstract class for wrapper sampling methods that call base sampling methods.

Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.

WrapperSamplingMethod is designed for meta-strategies that:
  * Use several base sampling methods (e.g., margin, uniform, mixtures).
  * Combine their suggested points according to some higher-level logic
    (e.g., EXP3 bandit, simulated batch trajectories).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

from active_learning_methods.constants import AL_MAPPING
from active_learning_methods.constants import get_all_possible_arms
from active_learning_methods.sampling_def import SamplingMethod

# Populate AL_MAPPING with all tools/“arms” (including mixture_of_samplers,
# bandit-based wrappers, etc.), so wrapper classes can instantiate them by name.
get_all_possible_arms()


class WrapperSamplingMethod(SamplingMethod):
  """Base class for wrapper-style active learning methods.

  Inherits from SamplingMethod so that wrappers share the same interface
  (e.g., select_batch) as base samplers, but they are responsible for
  creating and coordinating several underlying samplers.
  """
  __metaclass__ = abc.ABCMeta

  def initialize_samplers(self, mixtures):
    """Create base samplers and mixture samplers for a given set of mixtures.

    Args:
      mixtures: iterable of mixture specs, each a dict with key 'methods'
        (a list of sampler names, e.g., ['margin', 'uniform']) and typically
        'weights' (their mixture proportions). Wrapper subclasses (like
        BanditDiscreteSampler or SimulateBatchSampler) pass in mixture configs.

    This method:
      1. Finds the unique base method names across all mixtures.
      2. Instantiates a shared base sampler for each method name.
      3. For each mixture, creates a MixtureOfSamplers instance that uses
         these shared base samplers.
    """
    methods = []
    # Collect all method names referenced in the mixture configs.
    for m in mixtures:
      methods += m['methods']
    # Use a set to keep only unique method names.
    methods = set(methods)
    # base_samplers: one instance per base method (e.g., margin, uniform).
    self.base_samplers = {}
    for s in methods:
      # Instantiate each base AL sampler using AL_MAPPING and share them
      # across all mixtures that reference this method.
      self.base_samplers[s] = AL_MAPPING[s](self.X, self.y, self.seed)
    # self.samplers: list of MixtureOfSamplers, one per mixture config.
    self.samplers = []
    for m in mixtures:
      # AL_MAPPING['mixture_of_samplers'] is a constructor for MixtureOfSamplers.
      # We pass the shared base_samplers to allow state sharing (e.g., for
      # expensive samplers like hierarchical clustering).
      self.samplers.append(
          AL_MAPPING['mixture_of_samplers'](self.X, self.y, self.seed, m,
                                            self.base_samplers))
