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

""" Select a new batch based on results of simulated trajectories."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np

from active_learning_methods.wrapper_sampler_def import AL_MAPPING
from active_learning_methods.wrapper_sampler_def import WrapperSamplingMethod


class SimulateBatchSampler(WrapperSamplingMethod):
  """Creates batch based on trajectories simulated using smaller batch sizes.

  Current support use case: simulate smaller batches than the batch size
  actually indicated to emulate which points would be selected in a
  smaller batch setting. This method can do better than just selecting
  a batch straight out if smaller batches perform better and the simulations
  are informative enough and are not hurt too much by labeling noise.
  """

  def __init__(self,
               X,
               y,
               seed,
               samplers=[{'methods': ('margin', 'uniform'),'weight': (1, 0)}],
               n_sims=10,
               train_per_sim=10,
               return_type='best_sim'):
    """ Initialize sampler with options.

    Args:
      X: training data
      y: labels may be used by base sampling methods
      seed: seed for np.random
      samplers: list of dicts with two fields
        'samplers': list of named samplers
        'weights': percentage of batch to allocate to each sampler
      n_sims: number of total trajectories to simulate
      train_per_sim: number of minibatches to split the batch into
      return_type: two return types supported right now
        best_sim: return points selected by the best trajectory
        frequency: returns points selected the most over all trajectories
    """
    # Name of this wrapper sampler (used in logging / configs).
    self.name = 'simulate_batch'
    # Store full training data and labels.
    self.X = X
    self.y = y
    # Random seed for reproducibility of simulations and hallucinated labels.
    self.seed = seed
    # Number of simulated trajectories per sampler.
    self.n_sims = n_sims
    # How many mini-batches we conceptually split one batch into when simulating.
    self.train_per_sim = train_per_sim
    # How we choose the final batch from simulation results (best_sim/frequency).
    self.return_type = return_type
    # List of sampler configurations (each describing a mixture of base methods).
    self.samplers_list = samplers
    # Initialize underlying samplers based on provided specifications.
    self.initialize_samplers(self.samplers_list)
    # Stores detailed information about each simulation trajectory.
    self.trace = []
    # Stores all indices that have ever been selected during simulation.
    self.selected = []
    # Set the numpy random seed to keep sampling behavior reproducible.
    np.random.seed(seed)

  def simulate_batch(self, sampler, N, already_selected, y, model, X_test,
                     y_test, **kwargs):
    """Simulates smaller batches by using hallucinated y to select next batch.

    Assumes that select_batch is only dependent on already_selected and not on
    any other states internal to the sampler.  i.e. this would not work with
    BanditDiscreteSampler but will work with margin, hierarchical, and uniform.

    Args:
      sampler: dict with two fields
        'samplers': list of named samplers
        'weights': percentage of batch to allocate to each sampler
      N: batch size
      already_selected: indices already labeled
      y: y to use for training
      model: model to use for margin calc
      X_test: validaiton data
      y_test: validation labels

    Returns:
      - mean accuracy
      - indices selected by best hallucinated trajectory
      - best accuracy achieved by one of the trajectories
    """
    # Size of each simulated minibatch (how many points to add per training step).
    minibatch = max(int(math.ceil(N / self.train_per_sim)), 1)
    # Will store [hallucinated_acc, true_acc] for each simulation.
    results = []
    # Track the best hallucinated accuracy across simulations.
    best_acc = 0
    # Indices selected by the best-performing simulated trajectory.
    best_inds = []
    # Reset list of all selected indices across *all* simulations.
    self.selected = []
    # Number of minibatches to cover N total points (ceil division).
    n_minibatch = int(N/minibatch) + (N % minibatch > 0)

    # Run n_sims independent simulated trajectories.
    for _ in range(self.n_sims):
      # For this trajectory, store the indices selected and hallucinated labels.
      inds = []
      hallucinated_y = []

      # Copy these objects to make sure they are not modified while simulating
      # trajectories as they are used later by the main run_experiment script.
      kwargs['already_selected'] = copy.copy(already_selected)
      kwargs['y'] = copy.copy(y)
      # Assumes that model has already by fit using all labeled data so
      # the probabilities can be used immediately to hallucinate labels
      kwargs['model'] = copy.deepcopy(model)

      # Simulate adding data in minibatch chunks.
      for _ in range(n_minibatch):
        # Ensure we never exceed the requested batch size N in this sim.
        batch_size = min(minibatch, N-len(inds))
        if batch_size > 0:
          # Ask the sampler to propose the next minibatch.
          kwargs['N'] = batch_size
          new_inds = sampler.select_batch(**kwargs)
          inds.extend(new_inds)

          # All models need to have predict_proba method
          probs = kwargs['model'].predict_proba(self.X[new_inds])
          # Hallucinate labels for selected datapoints to be label
          # using class probabilities from model
          try:
            # Some models are wrapped in GridSearchCV or similar; use best_estimator_.
            classes = kwargs['model'].best_estimator_.classes_
          except:
            # Otherwise, use the model's classes_ attribute directly.
            classes = kwargs['model'].classes_
          # Sample a label for each selected point according to its predicted probabilities.
          new_y = ([
              np.random.choice(classes, p=probs[i, :])
              for i in range(batch_size)
          ])
          hallucinated_y.extend(new_y)
          # Not saving already_selected here, if saving then should sort
          # only for the input to fit but preserve ordering of indices in
          # already_selected
          # Update the list of labeled indices with the new points.
          kwargs['already_selected'] = sorted(kwargs['already_selected']
                                              + new_inds)
          # Insert hallucinated labels into the label vector at selected indices.
          kwargs['y'][new_inds] = new_y
          # Re-train the model on the expanded labeled dataset (with hallucinated labels).
          kwargs['model'].fit(self.X[kwargs['already_selected']],
                              kwargs['y'][kwargs['already_selected']])
      # Evaluate model performance on validation set under hallucinated labels.
      acc_hallucinated = kwargs['model'].score(X_test, y_test)
      # Track the best (highest) hallucinated accuracy and its selected indices.
      if acc_hallucinated > best_acc:
        best_acc = acc_hallucinated
        best_inds = inds
      # Retrain model on the same indices but with the *true* labels to compare.
      kwargs['model'].fit(self.X[kwargs['already_selected']],
                          y[kwargs['already_selected']])
      # Useful to know how accuracy compares for model trained on hallucinated
      # labels vs trained on true labels.  But can remove this train to speed
      # up simulations.  Won't speed up significantly since many more models
      # are being trained inside the loop above.
      acc_true = kwargs['model'].score(X_test, y_test)
      results.append([acc_hallucinated, acc_true])
      print('Hallucinated acc: %.3f, Actual Acc: %.3f' % (acc_hallucinated,
                                                          acc_true))

      # Save trajectory for reference
      t = {}
      t['arm'] = sampler                 # which sampler configuration was used
      t['data_size'] = len(kwargs['already_selected'])
      t['inds'] = inds                   # indices selected in this trajectory
      t['y_hal'] = hallucinated_y        # hallucinated labels
      t['acc_hal'] = acc_hallucinated    # validation acc with hallucinated labels
      t['acc_true'] = acc_true           # validation acc with true labels
      self.trace.append(t)
      # Keep track of all selected indices across all simulations (for frequency mode).
      self.selected.extend(inds)
      # Delete created copies
      del kwargs['model']
      del kwargs['already_selected']
    # Convert results to NumPy array for convenience (n_sims x 2).
    results = np.array(results)
    # Return mean [hallucinated_acc, true_acc], best indices, and best hallucinated accuracy.
    return np.mean(results, axis=0), best_inds, best_acc

  def sampler_select_batch(self, sampler, N, already_selected, y, model, X_test, y_test, **kwargs):
    """Calculate the performance of the model if the batch had been selected using the base method without simulation.

    Args:
      sampler: dict with two fields
        'samplers': list of named samplers
        'weights': percentage of batch to allocate to each sampler
      N: batch size
      already_selected: indices already selected
      y: labels to use for training
      model: model to use for training
      X_test, y_test: validation set

    Returns:
      - indices selected by base method
      - validation accuracy of model trained on new batch
    """
    # Work on a copy of the model so we don't overwrite the caller's model.
    m = copy.deepcopy(model)
    # Pass labels and copied model into the sampler.
    kwargs['y'] = y
    kwargs['model'] = m
    # Use a copy of already_selected so we don't modify original list.
    kwargs['already_selected'] = copy.copy(already_selected)
    inds = []
    # Request a full batch of size N from the base sampler (no hallucination here).
    kwargs['N'] = N
    inds.extend(sampler.select_batch(**kwargs))
    # Update labeled set to include the newly chosen points.
    kwargs['already_selected'] = sorted(kwargs['already_selected'] + inds)

    # Fit model on real labels for these points and compute validation accuracy.
    m.fit(self.X[kwargs['already_selected']], y[kwargs['already_selected']])
    acc = m.score(X_test, y_test)
    # Cleanup temporary objects.
    del m
    del kwargs['already_selected']
    return inds, acc

  def select_batch_(self, N, already_selected, y, model,
                    X_test, y_test, **kwargs):
    """ Returns a batch of size N selected by using the best sampler in simulation

    Args:
      samplers: list of sampling methods represented by dict with two fields
        'samplers': list of named samplers
        'weights': percentage of batch to allocate to each sampler
      N: batch size
      already_selected: indices of datapoints already labeled
      y: actual labels, used to compare simulation with actual
      model: training model to use to evaluate different samplers.  Model must
        have a predict_proba method with same signature as that in sklearn
      n_sims: the number of simulations to perform for each sampler
      minibatch: batch size to use for simulation
    """

    # Will store, for each sampler:
    # [mean_result_vector, best_sim_inds, base_inds, base_acc]
    results = []

    # THE INPUTS CANNOT BE MODIFIED SO WE MAKE COPIES FOR THE CHECK LATER
    # Should check model but kernel_svm does not have coef_ so need better
    # handling here
    copy_selected = copy.copy(already_selected)
    copy_y = copy.copy(y)

    # Loop over all sampler configurations in this wrapper.
    for s in self.samplers:
      # Run the simulated trajectories (hallucinated labels) for this sampler.
      sim_results, sim_inds, sim_acc = self.simulate_batch(
          s, N, already_selected, y, model, X_test, y_test, **kwargs)
      # Also compute the performance of using the sampler "as is" (no simulation).
      real_inds, acc = self.sampler_select_batch(
          s, N, already_selected, y, model, X_test, y_test, **kwargs)
      print('Best simulated acc: %.3f, Actual acc: %.3f' % (sim_acc, acc))
      # Store both simulated and real results for this sampler configuration.
      results.append([sim_results, sim_inds, real_inds, acc])
    # Choose the sampler whose *mean hallucinated accuracy* (index 0 of sim_results)
    # is the highest.
    best_s = np.argmax([r[0][0] for r in results])

    # Make sure that model object fed in did not change during simulations
    assert all(y == copy_y)
    assert all([copy_selected[i] == already_selected[i]
                for i in range(len(already_selected))])

    # Return indices based on return type specified
    if self.return_type == 'best_sim':
      # Use the indices from the best simulated trajectory.
      return results[best_s][1]
    elif self.return_type == 'frequency':
      # Use the points that appear most frequently across all simulated trajectories.
      unique, counts = np.unique(self.selected, return_counts=True)
      argcount = np.argsort(-counts)
      return list(unique[argcount[0:N]])
    # Default: return the indices from the base (non-simulated) selection
    # for the best sampler.
    return results[best_s][2]

  def to_dict(self):
    """Return a dictionary with full simulation trace information."""
    output = {}
    # Each entry in 'simulated_trajectories' is a dict describing one trajectory.
    output['simulated_trajectories'] = self.trace
    return output
