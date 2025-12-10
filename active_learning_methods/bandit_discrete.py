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

"""Bandit wrapper around base AL sampling methods.

Assumes adversarial multi-armed bandit setting where arms correspond to 
mixtures of different AL methods.

Uses EXP3 algorithm to decide which AL method to use to create the next batch.
Similar to Hsu & Lin 2015, Active Learning by Learning.
https://www.csie.ntu.edu.tw/~htlin/paper/doc/aaai15albl.pdf

In this implementation:
* Each "arm" is a mixture of underlying active learning (AL) strategies.
* After each batch, a reward is computed from the model accuracy.
* The EXP3 update adjusts the probability of choosing each arm.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from active_learning_methods.wrapper_sampler_def import AL_MAPPING, WrapperSamplingMethod


class BanditDiscreteSampler(WrapperSamplingMethod):
  """Wraps EXP3 around mixtures of indicated methods.

  Uses EXP3 mult-armed bandit algorithm to select sampler methods.
  """

  def __init__(self,
               X,
               y,
               seed,
               reward_function = lambda AL_acc: AL_acc[-1],
               gamma=0.5,
               samplers=[{'methods':('margin','uniform'),'weights':(0,1)},
                         {'methods':('margin','uniform'),'weights':(1,0)}]):
    """Initializes sampler with indicated gamma and arms.

    Args:
      X: training data
      y: labels, may need to be input into base samplers
      seed: seed to use for random sampling
      reward_function: reward based on previously observed accuracies.  Assumes
        that the input is a sequence of observed accuracies.  Will ultimately be
        a class method and may need access to other class properties.
      gamma: weight on uniform mixture.  Arm probability updates are a weighted
        mixture of uniform and an exponentially weighted distribution.
        Lower gamma more aggressively updates based on observed rewards.
      samplers: list of dicts with two fields
        'samplers': list of named samplers
        'weights': percentage of batch to allocate to each sampler
    """
    # Name used to identify this sampler in the larger AL framework.
    self.name = 'bandit_discrete'
    # Fix the random seed so that arm selection is reproducible.
    np.random.seed(seed)
    # Store training data and labels (passed to underlying samplers).
    self.X = X
    self.y = y
    self.seed = seed
    # Initialize underlying samplers (arms) using parent-class helper.
    # This sets up self.samplers and self.base_samplers based on the configs.
    self.initialize_samplers(samplers)

    # Bandit hyperparameter controlling exploration vs exploitation.
    self.gamma = gamma
    # Number of arms (i.e., different sampler mixtures).
    self.n_arms = len(samplers)
    # Function that maps the accuracy history to a scalar reward.
    self.reward_function = reward_function

    # History of which arms were selected at each iteration.
    self.pull_history = []
    # History of observed accuracies (used to compute rewards).
    self.acc_history = []
    # Weight vector w for EXP3; initialized to 1 for all arms.
    self.w = np.ones(self.n_arms)
    # Intermediate reward estimate vector x used in the EXP3 update.
    self.x = np.zeros(self.n_arms)
    # Initial arm-selection probabilities (uniform at the start).
    self.p = self.w / (1.0 * self.n_arms)
    # List to store probability distributions over time for later analysis.
    self.probs = []

  def update_vars(self, arm_pulled):
    """Update EXP3 weights and probabilities after pulling an arm.

    Args:
      arm_pulled: Index of the arm that was selected in the previous round.
    """
    # Compute the reward from the accuracy history using the reward function.
    reward = self.reward_function(self.acc_history)
    # Reset reward estimate vector.
    self.x = np.zeros(self.n_arms)
    # Importance-weighted reward estimate for the pulled arm:
    # reward divided by the probability with which it was chosen.
    self.x[arm_pulled] = reward / self.p[arm_pulled]
    # Update the weight vector w according to the EXP3 rule.
    self.w = self.w * np.exp(self.gamma * self.x / self.n_arms)
    # Update the probability distribution p as a mixture of:
    #  - normalized weights (exploitation),
    #  - uniform distribution over arms (exploration).
    self.p = ((1.0 - self.gamma) * self.w / sum(self.w)
              + self.gamma / self.n_arms)
    # Print current probabilities (useful for debugging / logging).
    print(self.p)
    # Store the probability distribution for analysis.
    self.probs.append(self.p)

  def select_batch_(self, already_selected, N, eval_acc, **kwargs):
    """Returns batch of datapoints sampled using mixture of AL_methods.

    Assumes that data has already been shuffled.

    Args:
      already_selected: index of datapoints already selected
      N: batch size
      eval_acc: accuracy of model trained after incorporating datapoints from
        last recommended batch

    Returns:
      indices of points selected to label
    """
    # Record the most recent model accuracy.
    self.acc_history.append(eval_acc)
    # If we have pulled at least one arm before, update bandit variables
    # based on the most recently pulled arm.
    if len(self.pull_history) > 0:
      self.update_vars(self.pull_history[-1])
    # Sample an arm index according to the current probability distribution p.
    arm = np.random.choice(range(self.n_arms), p=self.p)
    # Record which arm was pulled this round.
    self.pull_history.append(arm)
    # Add batch size and already-selected indices to kwargs for the base sampler.
    kwargs['N'] = N
    kwargs['already_selected'] = already_selected
    # Delegate the actual point selection to the chosen base sampler.
    sample = self.samplers[arm].select_batch(**kwargs)
    return sample

  def to_dict(self):
    """Return a dictionary summarizing the bandit's state and history.

    This is useful for logging, saving, or later analysis of bandit behavior.
    """
    output = {}
    # Configurations of the base samplers (arms).
    output['samplers'] = self.base_samplers
    # Probability distributions over arms at each round.
    output['arm_probs'] = self.probs
    # Sequence of arm indices chosen over time.
    output['pull_history'] = self.pull_history
    # Observed rewards (accuracies) at each round.
    output['rewards'] = self.acc_history
    return output
