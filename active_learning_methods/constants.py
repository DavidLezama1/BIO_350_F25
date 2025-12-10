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

"""Controls imports to fill up dictionary of different sampling methods.

This module defines a global dictionary, AL_MAPPING, that maps string
names (like 'margin', 'entropy', 'bandit_discrete') to the corresponding
Active Learning sampler classes or partially-initialized constructors.

The functions here:
  - get_base_AL_mapping():    register basic AL strategies.
  - get_all_possible_arms():  register mixture-of-samplers strategy.
  - get_wrapper_AL_mapping(): register wrapper strategies such as bandits
                              and simulation-based samplers.
  - get_mixture_of_samplers(): parse mixture names and build a partial
                               constructor for MixtureOfSamplers.
  - get_AL_sampler():         main lookup function used by the rest of
                              the code to get a sampler by name.
"""

from functools import partial

# Global mapping from sampler name (str) to class or partial() constructor.
AL_MAPPING = {}


def get_base_AL_mapping():
  """Populate AL_MAPPING with the base (non-wrapper) AL samplers.

  Each import brings in a specific active learning strategy. The mapping
  connects a short string key to the corresponding class, so other parts
  of the code can request samplers by name.
  """
  from active_learning_methods.margin_AL import MarginAL
  from active_learning_methods.informative_diverse import InformativeClusterDiverseSampler
  from active_learning_methods.hierarchical_clustering_AL import HierarchicalClusterAL
  from active_learning_methods.uniform_sampling import UniformSampling
  from active_learning_methods.represent_cluster_centers import RepresentativeClusterMeanSampling
  from active_learning_methods.graph_density import GraphDensitySampler
  from active_learning_methods.kcenter_greedy import kCenterGreedy
  from active_learning_methods.entropy_sampling import EntropyAL
  from active_learning_methods.confidence_sampling import ConfidenceAL

  # Uncertainty-based margin sampler.
  AL_MAPPING['margin'] = MarginAL
  # Hybrid informative + diverse sampler.
  AL_MAPPING['informative_diverse'] = InformativeClusterDiverseSampler
  # Hierarchical clustering-based sampler.
  AL_MAPPING['hierarchical'] = HierarchicalClusterAL
  # Purely random sampling baseline.
  AL_MAPPING['uniform'] = UniformSampling
  # Representative cluster center sampling (based on means).
  AL_MAPPING['margin_cluster_mean'] = RepresentativeClusterMeanSampling
  # Graph density-based diversity sampler.
  AL_MAPPING['graph_density'] = GraphDensitySampler
  # k-Center greedy diversity sampler.
  AL_MAPPING['kcenter'] = kCenterGreedy
  # Entropy-based uncertainty sampler.
  AL_MAPPING['entropy'] = EntropyAL
  # Confidence-based (least confident) sampler.
  AL_MAPPING['confidence'] = ConfidenceAL


def get_all_possible_arms():
  """Register the MixtureOfSamplers arm into AL_MAPPING.

  This method makes the 'mixture_of_samplers' key available so that
  we can construct samplers that combine multiple base AL strategies
  with specified weights.
  """
  from active_learning_methods.mixture_of_samplers import MixtureOfSamplers
  AL_MAPPING['mixture_of_samplers'] = MixtureOfSamplers


def get_wrapper_AL_mapping():
  """Populate AL_MAPPING with wrapper samplers (bandits, simulators).

  These methods do not select individual points directly. Instead, they
  wrap around base samplers and:
    - choose which sampler to use at each step (bandit),
    - simulate multiple AL trajectories and pick the best one.
  """
  from active_learning_methods.bandit_discrete import BanditDiscreteSampler
  from active_learning_methods.simulate_batch import SimulateBatchSampler

  # Bandit mixture strategy: several arms each with different margin/uniform
  # proportions. partial() creates constructors with fixed 'samplers' argument.
  AL_MAPPING['bandit_mixture'] = partial(
      BanditDiscreteSampler,
      samplers=[{
          'methods': ['margin', 'uniform'],
          'weights': [0, 1]
      }, {
          'methods': ['margin', 'uniform'],
          'weights': [0.25, 0.75]
      }, {
          'methods': ['margin', 'uniform'],
          'weights': [0.5, 0.5]
      }, {
          'methods': ['margin', 'uniform'],
          'weights': [0.75, 0.25]
      }, {
          'methods': ['margin', 'uniform'],
          'weights': [1, 0]
      }])

  # Simpler bandit setup with just two arms: all uniform vs all margin.
  AL_MAPPING['bandit_discrete'] = partial(
      BanditDiscreteSampler,
      samplers=[{
          'methods': ['margin', 'uniform'],
          'weights': [0, 1]
      }, {
          'methods': ['margin', 'uniform'],
          'weights': [1, 0]
      }])

  # Simulation-based sampler that runs multiple AL simulations using
  # different mixtures and returns a combined result (not necessarily the
  # best single run).
  AL_MAPPING['simulate_batch_mixture'] = partial(
      SimulateBatchSampler,
      samplers=({
          'methods': ['margin', 'uniform'],
          'weights': [1, 0]
      }, {
          'methods': ['margin', 'uniform'],
          'weights': [0.5, 0.5]
      }, {
          'methods': ['margin', 'uniform'],
          'weights': [0, 1]
      }),
      n_sims=5,
      train_per_sim=10,
      return_best_sim=False)

  # Simulation-based sampler that returns the best-performing simulation
  # (according to some metric) from multiple trials.
  AL_MAPPING['simulate_batch_best_sim'] = partial(
      SimulateBatchSampler,
      samplers=[{
          'methods': ['margin', 'uniform'],
          'weights': [1, 0]
      }],
      n_sims=10,
      train_per_sim=10,
      return_type='best_sim')

  # Simulation-based sampler that aggregates arm usage frequencies across
  # simulations and returns a selection based on those frequencies.
  AL_MAPPING['simulate_batch_frequency'] = partial(
      SimulateBatchSampler,
      samplers=[{
          'methods': ['margin', 'uniform'],
          'weights': [1, 0]
      }],
      n_sims=10,
      train_per_sim=10,
      return_type='frequency')


def get_mixture_of_samplers(name):
  """Parse a mixture name and return a partial MixtureOfSamplers constructor.

  Expected name format:
    'mixture_of_samplers-sampler1-weight1-sampler2-weight2-...'

  Example:
    'mixture_of_samplers-margin-0.7-uniform-0.3'

  This function:
    - validates that 'mixture_of_samplers' is loaded,
    - parses sampler names and their weights from the string,
    - checks that weights sum to 1,
    - returns a partial() for MixtureOfSamplers with the given mixture.
  """
  assert 'mixture_of_samplers' in name
  if 'mixture_of_samplers' not in AL_MAPPING:
    raise KeyError('Mixture of Samplers not yet loaded.')
  # Split the name into tokens, skipping the first one ('mixture_of_samplers').
  args = name.split('-')[1:]
  # Sampler names are at even positions, weights at odd positions.
  samplers = args[0::2]
  weights = args[1::2]
  # Convert weights from strings to floats.
  weights = [float(w) for w in weights]
  # Ensure weights define a valid probability distribution.
  assert sum(weights) == 1
  # Bundle methods and weights into a configuration dictionary.
  mixture = {'methods': samplers, 'weights': weights}
  # Optional debugging output to visualize the parsed mixture.
  print(mixture)
  # Return a partial constructor for MixtureOfSamplers using this mixture.
  return partial(AL_MAPPING['mixture_of_samplers'], mixture=mixture)


def get_AL_sampler(name):
  """Return the sampler (class or constructor) corresponding to the given name.

  Behavior:
    - If 'name' is in AL_MAPPING and is not 'mixture_of_samplers', return it.
    - If 'name' encodes a mixture (contains 'mixture_of_samplers'), parse it
      and return a partial MixtureOfSamplers.
    - Otherwise, raise NotImplementedError.

  This is the main entry point used by the rest of the code to retrieve
  a sampling strategy by name.
  """
  # Directly return base or wrapper samplers if they have been registered.
  if name in AL_MAPPING and name != 'mixture_of_samplers':
    return AL_MAPPING[name]
  # If this is a mixture-of-samplers string, construct it on the fly.
  if 'mixture_of_samplers' in name:
    return get_mixture_of_samplers(name)
  # If we reach here, the requested sampler is not available.
  raise NotImplementedError('The specified sampler is not available.')
