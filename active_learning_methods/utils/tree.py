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

"""Node and Tree class to support hierarchical clustering AL method.

Assumed to be binary tree.

Node class is used to represent each node in a hierarchical clustering.
Each node has certain properties that are used in the AL method.

Tree class is used to traverse a hierarchical clustering.

In this context, the Tree typically represents the output of a hierarchical
clustering algorithm. Each Node can be an internal node (cluster) or a leaf
(actual data point). The active learning method uses:

* node weights (relative size of subtrees),
* leaf mappings (which leaves belong to each internal node),
* pruning (choosing a subset of nodes that summarize the tree).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy


class Node(object):
  """Node class for hierarchical clustering.

  Initialized with name and left right children.

  Attributes:
    name: Identifier of the node (e.g., cluster ID or data point index).
    left: Left child Node in the binary tree (or None for leaf).
    right: Right child Node in the binary tree (or None for leaf).
    is_leaf: Boolean flag indicating whether the node is a leaf.
    parent: Parent Node in the tree (set later by Tree.fill_parents()).
    score: Score used in hierarchical clustering active learning (default 1.0).
    split: Boolean flag indicating whether this node has been "split" or
      expanded into its children for pruning purposes.
    best_label: Placeholder for storing the best label associated with this
      node (used by some AL strategies).
    weight: Fraction of total leaves that fall under this node (set later).
  """

  def __init__(self, name, left=None, right=None):
    # Name uniquely identifies the node.
    self.name = name
    # Pointers to left and right children (can be None for leaves).
    self.left = left
    self.right = right
    # A node is a leaf if it has no children.
    self.is_leaf = left is None and right is None
    # Parent will be assigned by Tree.fill_parents().
    self.parent = None
    # Fields for hierarchical clustering AL
    self.score = 1.0
    # Whether this node has been split/expanded in the pruning process.
    self.split = False
    # Best label (e.g., predicted class) associated with this node.
    self.best_label = None
    # Weight of this node, defined as (# leaves under node / total # leaves).
    self.weight = None

  def set_parent(self, parent):
    """Set the parent pointer for this node."""
    self.parent = parent


class Tree(object):
  """Tree object for traversing a binary tree.

  Most methods apply to trees in general with the exception of get_pruning
  which is specific to the hierarchical clustering AL method.

  Attributes:
    node_dict: Dictionary mapping node IDs to [left_child_id, right_child_id].
    root: Root Node of the tree.
    nodes: Dictionary mapping node IDs to Node instances (for fast lookup).
    leaves_mapping: Mapping from node ID to list of leaf IDs in its subtree.
    n_leaves: Total number of leaf nodes in the tree.
  """

  def __init__(self, root, node_dict):
    """Initializes tree and creates all nodes in node_dict.

    Args:
      root: id of the root node
      node_dict: dictionary with node_id as keys and entries indicating
        left and right child of node respectively.

    The constructor first builds the tree structure (Node objects) using
    make_tree(), then creates a fast-access mapping from node IDs to Node
    instances, and fills in parent pointers via fill_parents().
    """
    # Store the raw node structure (child relationships).
    self.node_dict = node_dict
    # Recursively build the tree of Node objects starting from the root ID.
    self.root = self.make_tree(root)
    # Dictionary for fast ID -> Node lookup, filled in fill_parents().
    self.nodes = {}
    # Mapping from node ID to its descendant leaves (filled later).
    self.leaves_mapping = {}
    # Populate self.nodes and parent pointers for all nodes.
    self.fill_parents()
    # Total number of leaves; set when child-leaf mapping is created.
    self.n_leaves = None

  def print_tree(self, node, max_depth):
    """Helper function to print out tree for debugging.

    Args:
      node: ID of the node from which to start printing.
      max_depth: Maximum depth (number of levels) to traverse downward.

    For each level up to max_depth, this function prints:
      node <id>: score <score>, weight <weight>
    for all nodes on that level.
    """
    # Start with the given node ID as the only node in the current level.
    node_list = [node]
    output = ""
    level = 0
    # Continue until we reach max_depth or there are no more nodes to traverse.
    while level < max_depth and len(node_list):
      # Use a set to collect children for the next level (avoid duplicates).
      children = set()
      for n in node_list:
        # Look up the Node object from its ID.
        node = self.get_node(n)
        # Append information about this node to the output string.
        output += ("\t"*level+"node %d: score %.2f, weight %.2f" %
                   (node.name, node.score, node.weight)+"\n")
        # Add children (if they exist) to the next level.
        if node.left:
          children.add(node.left.name)
        if node.right:
          children.add(node.right.name)
      # Move down one level in the tree.
      level += 1
      node_list = children
    # Print the full textual representation of the tree up to max_depth.
    return print(output)

  def make_tree(self, node_id):
    """Recursively construct the Node-based tree from node_dict.

    Args:
      node_id: ID of the current node to construct.

    Returns:
      A Node instance whose left and right children are recursively built
      using the node_dict structure, or None if node_id is None.
    """
    if node_id is not None:
      # Create current node and recursively create left and right subtrees.
      return Node(node_id,
                  self.make_tree(self.node_dict[node_id][0]),
                  self.make_tree(self.node_dict[node_id][1]))

  def fill_parents(self):
    """Populate parent pointers and build the ID->Node mapping (self.nodes)."""

    # Inner recursive helper to traverse the tree.
    def rec(pointer, parent):
      if pointer is not None:
        # Register this node in the dictionary for fast lookup.
        self.nodes[pointer.name] = pointer
        # Set the parent pointer.
        pointer.set_parent(parent)
        # Recurse on the left and right children.
        rec(pointer.left, pointer)
        rec(pointer.right, pointer)

    # Start recursion from the root; root has no parent (None).
    rec(self.root, None)

  def get_node(self, node_id):
    """Return the Node object associated with a given node ID."""
    return self.nodes[node_id]

  def get_ancestor(self, node):
    """Return a list of ancestor IDs from a node up to (but not including) root.

    Args:
      node: Either a Node object or an integer node ID.

    Returns:
      A list of node IDs representing the ancestors on the path from the given
      node up to the root (excluding the root itself).
    """
    ancestors = []
    # If an integer is given, look up the corresponding Node.
    if isinstance(node, int):
      node = self.get_node(node)
    # Walk upward through parent pointers until we reach the root.
    while node.name != self.root.name:
      node = node.parent
      ancestors.append(node.name)
    return ancestors

  def fill_weights(self):
    """Compute node weights based on the proportion of leaves in each subtree.

    For each node v in node_dict, we define:
      node.weight = (# leaves in subtree rooted at v) / (total # leaves).

    The leaves_mapping and n_leaves must already have been set by
    create_child_leaves_mapping().
    """
    for v in self.node_dict:
      node = self.get_node(v)
      # Number of leaves under this node divided by total number of leaves.
      node.weight = len(self.leaves_mapping[v]) / (1.0 * self.n_leaves)

  def create_child_leaves_mapping(self, leaves):
    """DP for creating child leaves mapping.
    
    Storing in dict to save recompute.

    Args:
      leaves: List of leaf node IDs in the tree.

    This method builds a mapping from every node ID (internal or leaf) to the
    list of leaf IDs contained in its subtree. It proceeds bottom-up using a
    dynamic programming-style approach: start with leaves and then iteratively
    fill parents once both children have been processed.
    """
    # Total number of leaves in the tree.
    self.n_leaves = len(leaves)
    # Initialize mapping for leaf nodes: each leaf maps to a list containing
    # only itself.
    for v in leaves:
      self.leaves_mapping[v] = [v]
    # Start from the parents of all leaves.
    node_list = set([self.get_node(v).parent for v in leaves])
    # Repeatedly process nodes whose children already have leaf mappings.
    while node_list:
      # Copy current set of nodes to fill this iteration.
      to_fill = copy.copy(node_list)
      for v in node_list:
        # We can fill v only if both children have leaf mappings.
        if (v.left.name in self.leaves_mapping
            and v.right.name in self.leaves_mapping):
          # Remove v from the set of nodes we still need to fill.
          to_fill.remove(v)
          # The leaves under v are the union of the leaves under its children.
          self.leaves_mapping[v.name] = (self.leaves_mapping[v.left.name] +
                                         self.leaves_mapping[v.right.name])
          # After filling v, we may need to fill its parent (if it exists).
          if v.parent is not None:
            to_fill.add(v.parent)
      # Move up the tree for the next iteration.
      node_list = to_fill
    # Once the leaves_mapping is complete, compute node weights.
    self.fill_weights()

  def get_child_leaves(self, node):
    """Return the list of leaf IDs under a given node ID."""
    return self.leaves_mapping[node]

  def get_pruning(self, node):
    """Return a pruning (list of node IDs) starting from the given node.

    The pruning is defined recursively:
      * If node.split is True, recurse into both children and concatenate
        their prunings.
      * Otherwise, stop at this node and return its ID.

    Intuitively, nodes with split == True are "expanded" to show more detail in
    the tree, while nodes with split == False are treated as leaf-like
    representatives in the pruning.
    """
    if node.split:
      # If node is split, prune both children and combine the results.
      return self.get_pruning(node.left) + self.get_pruning(node.right)
    else:
      # If node is not split, it is a terminal node in the pruning.
      return [node.name]
