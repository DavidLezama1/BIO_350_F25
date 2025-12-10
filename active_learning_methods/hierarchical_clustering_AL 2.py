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

"""Hierarchical cluster AL method.

Implements algorithm described in Dasgupta, S and Hsu, D,
"Hierarchical Sampling for Active Learning, 2008
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph
from active_learning_methods.sampling_def import SamplingMethod
from active_learning_methods.utils.tree import Tree


class HierarchicalClusterAL(SamplingMethod):
  """Implements hierarchical cluster AL based method.

  All methods are internal.  select_batch_ is called via abstract classes
  outward facing method select_batch.

  Default affininity is euclidean and default linkage is ward which links
  cluster based on variance reduction.  Hence, good results depend on
  having normalized and standardized data.
  """

  def __init__(self, X, y, seed, beta=2, affinity='euclidean', linkage='ward',
               clustering=None, max_features=None):
    """Initializes AL method and fits hierarchical cluster to data.

    Args:
      X: data
      y: labels for determinining number of clusters as an input to
        AgglomerativeClustering
      seed: random seed used for sampling datapoints for batch
      beta: width of error used to decide admissble labels, higher value of beta
        corresponds to wider confidence and less stringent definition of
        admissibility
        See scikit Aggloerative clustering method for more info
      affinity: distance metric used for hierarchical clustering
      linkage: linkage method used to determine when to join clusters
      clustering: can provide an AgglomerativeClustering that is already fit
      max_features: limit number of features used to construct hierarchical
        cluster.  If specified, PCA is used to perform feature reduction and
        the hierarchical clustering is performed using transformed features.
    """
    # Name used to identify this AL strategy.
    self.name = 'hierarchical'
    # Save random seed and set numpy RNG for reproducibility.
    self.seed = seed
    np.random.seed(seed)
    # Variables for the hierarchical cluster
    self.already_clustered = False
    # If a pre-fit clustering model is passed, store it and mark as clustered.
    if clustering is not None:
      self.model = clustering
      self.already_clustered = True
    # Attributes describing the hierarchical tree structure from sklearn.
    self.n_leaves = None
    self.n_components = None
    self.children_list = None
    self.node_dict = None
    self.root = None  # Node name, all node instances access through self.tree
    self.tree = None
    # Variables for the AL algorithm
    self.initialized = False
    # Beta controls how strict admissibility bounds are.
    self.beta = beta
    # Dictionary mapping node id -> assigned label (best_label).
    self.labels = {}
    # Current pruning of the hierarchical tree (list of node ids).
    self.pruning = []
    # Dictionary mapping node id -> set of admissible labels for that node.
    self.admissible = {}
    # Nodes selected in the last iteration (internal nodes that were expanded).
    self.selected_nodes = None
    # Data variables
    self.classes = None
    self.X = X

    # The original label set from the dataset (used to define class space).
    classes = list(set(y))
    self.n_classes = len(classes)
    # If max_features is given, use PCA to reduce dimensionality before
    # clustering (often helpful for high-dimensional data).
    if max_features is not None:
      transformer = PCA(n_components=max_features)
      transformer.fit(X)
      self.transformed_X = transformer.fit_transform(X)
      #connectivity = kneighbors_graph(self.transformed_X,max_features)
      self.model = AgglomerativeClustering(
          affinity=affinity, linkage=linkage, n_clusters=len(classes))
      # Fit clustering on PCA-transformed features.
      self.fit_cluster(self.transformed_X)
    else:
      # Otherwise, cluster directly on the original feature space.
      self.model = AgglomerativeClustering(
          affinity=affinity, linkage=linkage, n_clusters=len(classes))
      self.fit_cluster(self.X)
    # Store labels as a numpy array for consistency.
    self.y = y

    # Dictionary storing labels observed during active learning:
    # key = index of datapoint, value = observed class.
    self.y_labels = {}
    # Fit cluster and update cluster variables

    # Build a Tree object from the hierarchical clustering structure.
    self.create_tree()
    print('Finished creating hierarchical cluster')

  def fit_cluster(self, X):
    """Fit AgglomerativeClustering model and cache structure info."""
    if not self.already_clustered:
      self.model.fit(X)
      self.already_clustered = True
    # Number of leaf nodes in the hierarchical tree.
    self.n_leaves = self.model.n_leaves_
    # Number of connected components (used internally by sklearn).
    self.n_components = self.model.n_components_
    # Children relationships: each row [left_child, right_child].
    self.children_list = self.model.children_

  def create_tree(self):
    """Create a Tree object from the clustering children_list.

    Builds a mapping from node ids to their children and uses it to
    construct a binary Tree. Also initializes the admissible label sets.
    """
    node_dict = {}
    # Leaves: indices 0 .. n_leaves-1, each with no children.
    for i in range(self.n_leaves):
      node_dict[i] = [None, None]
    # Internal nodes: indices n_leaves .. n_leaves+len(children_list)-1
    for i in range(len(self.children_list)):
      node_dict[self.n_leaves + i] = self.children_list[i]
    self.node_dict = node_dict
    # The sklearn hierarchical clustering algo numbers leaves which correspond
    # to actual datapoints 0 to n_points - 1 and all internal nodes have
    # ids greater than n_points - 1 with the root having the highest node id
    self.root = max(self.node_dict.keys())
    # Create the Tree structure from this node dictionary.
    self.tree = Tree(self.root, self.node_dict)
    # Populate mapping from each node to its set of descendant leaves.
    self.tree.create_child_leaves_mapping(range(self.n_leaves))
    # Initialize admissible label sets for all nodes.
    for v in node_dict:
      self.admissible[v] = set()

  def get_child_leaves(self, node):
    """Convenience wrapper: return leaf indices under given node id."""
    return self.tree.get_child_leaves(node)

  def get_node_leaf_counts(self, node_list):
    """Return an array of leaf counts for each node in node_list."""
    node_counts = []
    for v in node_list:
      node_counts.append(len(self.get_child_leaves(v)))
    return np.array(node_counts)

  def get_class_counts(self, y):
    """Gets the count of all classes in a sample.

    Args:
      y: sample vector for which to perform the count
    Returns:
      count of classes for the sample vector y, the class order for count will
      be the same as that of self.classes
    """
    # unique: unique class values in y; counts: their respective counts.
    unique, counts = np.unique(y, return_counts=True)
    complete_counts = []
    # Build a complete count vector aligned with self.classes.
    for c in self.classes:
      if c not in unique:
        complete_counts.append(0)
      else:
        index = np.where(unique == c)[0][0]
        complete_counts.append(counts[index])
    return np.array(complete_counts)

  def observe_labels(self, labeled):
    """Update observed labels dictionary based on newly labeled points.

    Args:
      labeled: dict mapping datapoint index -> observed label
    """
    # Incorporate all new labels into y_labels.
    for i in labeled:
      self.y_labels[i] = labeled[i]
    # Update the set of classes actually observed so far.
    self.classes = np.array(
        sorted(list(set([self.y_labels[k] for k in self.y_labels]))))
    # Update number of classes based on observed labels.
    self.n_classes = len(self.classes)

  def initialize_algo(self):
    """Initialize pruning and labels for the first iteration.

    Start with a pruning consisting only of the root node, and assign
    an initial (random) label to the root.
    """
    # Start with the root as the only node in the pruning.
    self.pruning = [self.root]
    # Assign root a random label from the observed class set.
    self.labels[self.root] = np.random.choice(self.classes)
    # Update the corresponding Tree node's best_label.
    node = self.tree.get_node(self.root)
    node.best_label = self.labels[self.root]
    # Track which nodes were selected (initially just the root).
    self.selected_nodes = [self.root]

  def get_node_class_probabilities(self, node, y=None):
    """Estimate class probabilities for a node based on observed labels.

    Args:
      node: node id in the tree.
      y: optional label array to use instead of self.y_labels.

    Returns:
      (n_v, p_v):
        n_v: number of labeled datapoints in this node's subtree.
        p_v: vector of estimated class probabilities over self.classes.
    """
    # Get indices of all leaves under this node.
    children = self.get_child_leaves(node)
    # Choose which label mapping to use.
    if y is None:
      y_dict = self.y_labels
    else:
      y_dict = dict(zip(range(len(y)), y))
    # Collect labels for all children that have been observed.
    labels = [y_dict[c] for c in children if c in y_dict]
    # If no labels have been observed, simply return uniform distribution
    if len(labels) == 0:
      return 0, np.ones(self.n_classes)/self.n_classes
    # Otherwise, build normalized class-count vector.
    return len(labels), self.get_class_counts(labels) / (len(labels) * 1.0)

  def get_node_upper_lower_bounds(self, node):
    """Compute upper and lower confidence bounds for class probabilities.

    Uses a simple concentration bound around the empirical probabilities
    to create an interval [p_lb, p_up] for each class.

    Returns:
      (p_lb, p_up): arrays of same length as self.classes with lower/upper
                    bounds on class probabilities.
    """
    n_v, p_v = self.get_node_class_probabilities(node)
    # If no observations, return worst possible upper lower bounds
    if n_v == 0:
      return np.zeros(len(p_v)), np.ones(len(p_v))
    # Width of confidence interval for each class (per Dasgupta & Hsu).
    delta = 1. / n_v + np.sqrt(p_v * (1 - p_v) / (1. * n_v))
    # Clamp bounds to [0, 1].
    return (np.maximum(p_v - delta, np.zeros(len(p_v))),
            np.minimum(p_v + delta, np.ones(len(p_v))))

  def get_node_admissibility(self, node):
    """Determine which labels are admissible for this node.

    A label is admissible if its error is within a beta-scaled tolerance
    relative to the best alternative. Uses lower/upper bounds to account
    for uncertainty.

    Returns:
      Boolean array (length = n_classes) where True indicates admissible.
    """
    # Lower/upper bounds on class probabilities for this node.
    p_lb, p_up = self.get_node_upper_lower_bounds(node)
    # For each candidate label i, compute minimum possible error of any
    # alternative label (1 - p_up[c]) where c != i.
    all_other_min = np.vectorize(
        lambda i:min([1 - p_up[c] for c in range(len(self.classes)) if c != i]))
    # Lowest alternative error scaled by beta.
    lowest_alternative_error = self.beta * all_other_min(
        np.arange(len(self.classes)))
    # A label is admissible if its worst-case error (1 - p_lb) is still
    # below the beta-scaled alternative error.
    return 1 - p_lb < lowest_alternative_error

  def get_adjusted_error(self, node):
    """Compute adjusted error for each class at this node.

    Any label that is not admissible has its error set to 1 (worst case),
    so only admissible labels can be chosen as best_label.
    """
    # Empirical error (1 - probability) for each class.
    _, prob = self.get_node_class_probabilities(node)
    error = 1 - prob
    # Admissibility mask for this node.
    admissible = self.get_node_admissibility(node)
    # Set error = 1 for all non-admissible labels.
    not_admissible = np.where(admissible != True)[0]
    error[not_admissible] = 1.0
    return error

  def get_class_probability_pruning(self, method='lower'):
    """Get per-node probabilities for the currently assigned labels (pruning).

    Args:
      method: 'empirical', 'lower', or 'upper' for the probability estimate.

    Returns:
      Array of probabilities corresponding to each node in self.pruning.
    """
    prob_pruning = []
    for v in self.pruning:
      # Label currently assigned to this node.
      label = self.labels[v]
      label_ind = np.where(self.classes == label)[0][0]
      if method == 'empirical':
        # Use empirical class probabilities.
        _, v_prob = self.get_node_class_probabilities(v)
      else:
        # Use lower/upper confidence bounds.
        lower, upper = self.get_node_upper_lower_bounds(v)
        if method == 'lower':
          v_prob = lower
        elif method == 'upper':
          v_prob = upper
        else:
          raise NotImplementedError
      # Probability associated with node's current label.
      prob = v_prob[label_ind]
      prob_pruning.append(prob)
    return np.array(prob_pruning)

  def get_pruning_impurity(self, y):
    """Compute impurity of the current pruning given labels y.

    Impurity is defined as (1 - max class probability) for each node,
    and then averaged using weights based on how many leaves each node
    contains.
    """
    impurity = []
    # Compute impurity per node.
    for v in self.pruning:
      _, prob = self.get_node_class_probabilities(v, y)
      impurity.append(1-max(prob))
    impurity = np.array(impurity)
    # Weight by number of leaves under each node.
    weights = self.get_node_leaf_counts(self.pruning)
    weights = weights / sum(weights)
    return sum(impurity*weights)

  def update_scores(self):
    """Update scores and split decisions bottom-up for all nodes.

    For each node, we:
      - Compute admissible labels and adjusted error.
      - Decide whether node should be split (internal nodes only).
      - Update node.best_label and node.score accordingly.
    """
    # Start with all leaf nodes.
    node_list = set(range(self.n_leaves))
    # Loop through generations from bottom to top
    while len(node_list) > 0:
      parents = set()
      for v in node_list:
        node = self.tree.get_node(v)
        # Update admissible labels for node
        admissible = self.get_node_admissibility(v)
        admissable_indices = np.where(admissible)[0]
        # Add any admissible labels to this node's admissible set.
        for l in self.classes[admissable_indices]:
          self.admissible[v].add(l)
        # Calculate error per class and choose best label.
        v_error = self.get_adjusted_error(v)
        best_label_ind = np.argmin(v_error)
        # If best label is admissible, store as node.best_label.
        if admissible[best_label_ind]:
          node.best_label = self.classes[best_label_ind]
        score = v_error[best_label_ind]
        # By default, no split at this node.
        node.split = False

        # Determine if node should be split
        if v >= self.n_leaves:  # v is not a leaf
          if len(admissable_indices) > 0:  # There exists an admissible label
            # Make sure label set for node so that we can flow to children
            # if necessary
            assert node.best_label is not None
            # Only split if all ancestors are admissible nodes
            # This is part  of definition of admissible pruning
            admissible_ancestors = [len(self.admissible[a]) > 0 for a in
                                    self.tree.get_ancestor(node)]
            if all(admissible_ancestors):
              # Children of this internal node.
              left = self.node_dict[v][0]
              left_node = self.tree.get_node(left)
              right = self.node_dict[v][1]
              right_node = self.tree.get_node(right)
              # Leaf counts for [parent, left child, right child].
              node_counts = self.get_node_leaf_counts([v, left, right])
              # Score if we split this node: weighted combination of children.
              split_score = (node_counts[1] / node_counts[0] *
                             left_node.score + node_counts[2] /
                             node_counts[0] * right_node.score)
              # If splitting improves score, choose to split.
              if split_score < score:
                score = split_score
                node.split = True
        # Store final score for this node.
        node.score = score
        # Add parent of this node to the set we'll process in the next layer up.
        if node.parent:
          parents.add(node.parent.name)
        # Move up one level in the tree.
        node_list = parents

  def update_pruning_labels(self):
    """Update pruning and labels based on nodes selected for splitting.

    This function:
      - Replaces each selected internal node v in pruning with its children
        (according to tree.get_pruning).
      - Ensures that the pruning covers all leaves exactly once.
      - Propagates labels down: if a node has no best_label, inherit from parent.
    """
    # For each node that was selected for expansion in the last iteration:
    for v in self.selected_nodes:
      node = self.tree.get_node(v)
      # Get the set of nodes that replace v in the pruning.
      pruning = self.tree.get_pruning(node)
      self.pruning.remove(v)
      self.pruning.extend(pruning)
    # Check that pruning covers all leave nodes
    node_counts = self.get_node_leaf_counts(self.pruning)
    assert sum(node_counts) == self.n_leaves
    # Fill in labels for all nodes in the pruning.
    for v in self.pruning:
      node = self.tree.get_node(v)
      # If node has no best_label, inherit parent's label.
      if node.best_label  is None:
        node.best_label = node.parent.best_label
      self.labels[v] = node.best_label

  def get_fake_labels(self):
    """Create a full pseudo-label vector from current pruning labels.

    Each datapoint (leaf) gets the label of the pruning node that contains it.
    """
    fake_y = np.zeros(self.X.shape[0])
    # For each node in the pruning, assign its label to all its leaf children.
    for p in self.pruning:
      indices = self.get_child_leaves(p)
      fake_y[indices] = self.labels[p]
    return fake_y

  def train_using_fake_labels(self, model, X_test, y_test):
    """Train model on pseudo-labels and evaluate on test set.

    This is sometimes used to evaluate how good the current pruning is
    at predicting labels.

    Returns:
      test accuracy if pruning covers all classes; 0 otherwise.
    """
    # Distinct labels that appear in the current pruning.
    classes_labeled = set([self.labels[p] for p in self.pruning])
    # Only proceed if all classes are represented in the pruning.
    if len(classes_labeled) == self.n_classes:
      fake_y = self.get_fake_labels()
      model.fit(self.X, fake_y)
      test_acc = model.score(X_test, y_test)
      return test_acc
    return 0

  def select_batch_(self, N, already_selected, labeled, y, **kwargs):
    """Select a batch of datapoints to label using hierarchical sampling.

    Args:
      N: batch size (number of points to select)
      already_selected: indices of datapoints already chosen before
      labeled: dict mapping indices -> labels for newly labeled points
      y: ground-truth labels (used to compute true impurity for reporting)

    Returns:
      List of datapoint indices to query next.
    """
    # Observe labels for previously recommended batches
    self.observe_labels(labeled)

    # Initialize algorithm the first time this is called.
    if not self.initialized:
      self.initialize_algo()
      self.initialized = True
      print('Initialized algo')

    print('Updating scores and pruning for labels from last batch')
    # Update scores and split decisions for all nodes.
    self.update_scores()
    # Update pruning structure and labels after new splits.
    self.update_pruning_labels()
    print('Nodes in pruning: %d' % (len(self.pruning)))
    print('Actual impurity for pruning is: %.2f' %
          (self.get_pruning_impurity(y)))

    # TODO(lishal): implement multiple selection methods
    # selected_nodes: will store which pruning nodes are used in this batch.
    selected_nodes = set()
    # Get number of leaves under each pruning node (used for weighting).
    weights = self.get_node_leaf_counts(self.pruning)
    # Lower probability for nodes with higher class certainty.
    probs = 1 - self.get_class_probability_pruning()
    # Combine node size and uncertainty into a single weight.
    weights = weights * probs
    weights = weights / sum(weights)
    batch = []

    print('Sampling batch')
    # Sample datapoints from nodes in the pruning according to weights.
    while len(batch) < N:
      # Randomly select a pruning node, weighted by its "importance".
      node = np.random.choice(list(self.pruning), p=weights)
      children = self.get_child_leaves(node)
      # Choose only children not yet labeled and not already in current batch.
      children = [
          c for c in children if c not in self.y_labels and c not in batch
      ]
      if len(children) > 0:
        selected_nodes.add(node)
        # Randomly choose one child from the eligible set to add to batch.
        batch.append(np.random.choice(children))
    # Store which nodes were selected in this round for future pruning updates.
    self.selected_nodes = selected_nodes
    return batch

  def to_dict(self):
    """Return a simple dict containing the node_dict (tree structure).

    This is useful for logging, saving, or visualizing the hierarchy.
    """
    output = {}
    output['node_dict'] = self.node_dict
    return output
