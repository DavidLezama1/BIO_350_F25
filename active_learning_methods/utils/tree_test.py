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

"""Tests for sampling_methods.utils.tree.

This file contains unit tests for the Tree and Node classes used in
the active learning code. The tests verify that:
* the tree structure is created correctly,
* nodes can be accessed and have the correct parents/children,
* ancestor and leaf queries work as expected,
* weights and pruning operations behave correctly.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from active_learning__methods.utils import tree


class TreeTest(unittest.TestCase):
  """Test case for the Tree data structure used in sampling methods."""

  def setUp(self):
    """Build a simple binary tree that is reused across all tests.

    The tree layout is:

            1
           / \
          2   3
         / \ / \
        4  5 6  7

    - Internal nodes: 1, 2, 3
    - Leaf nodes: 4, 5, 6, 7

    We also mark nodes 1 and 2 as split, which simulates a partially
    grown decision tree used in active learning.
    """
    # node_dict maps each node ID to its left and right children.
    # Leaf nodes have [None, None] to indicate no children.
    node_dict = {
        1: (2, 3),
        2: (4, 5),
        3: (6, 7),
        4: [None, None],
        5: [None, None],
        6: [None, None],
        7: [None, None]
    }
    # Initialize the Tree with root node 1 and the dictionary above.
    self.tree = tree.Tree(1, node_dict)
    # Create a mapping from each internal node to its descendant leaf nodes.
    self.tree.create_child_leaves_mapping([4, 5, 6, 7])
    # Mark node 1 as a split (expanded) node.
    node = self.tree.get_node(1)
    node.split = True
    # Mark node 2 as a split (expanded) node.
    node = self.tree.get_node(2)
    node.split = True

  def assertNode(self, node, name, left, right):
    """Helper method to assert basic node properties.

    Args:
      node: The node object returned by the Tree.
      name: Expected name (ID) of the node.
      left: Expected name (ID) of the left child.
      right: Expected name (ID) of the right child.
    """
    # Check that the node has the correct ID and children.
    self.assertEqual(node.name, name)
    self.assertEqual(node.left.name, left)
    self.assertEqual(node.right.name, right)

  def testTreeRootSetCorrectly(self):
    """Test that the root node is set correctly with its children."""
    # The root should be node 1 with children 2 (left) and 3 (right).
    self.assertNode(self.tree.root, 1, 2, 3)

  def testGetNode(self):
    """Test that get_node returns the correct Node instance."""
    # Retrieve node 1 and confirm it is a Node with the right name.
    node = self.tree.get_node(1)
    assert isinstance(node, tree.Node)
    self.assertEqual(node.name, 1)

  def testFillParent(self):
    """Test that parent links for nodes are filled correctly."""
    # Node 3 should have node 1 as its parent.
    node = self.tree.get_node(3)
    self.assertEqual(node.parent.name, 1)

  def testGetAncestors(self):
    """Test that get_ancestor returns all ancestors of a node.

    For leaf node 5, the ancestors should be nodes 2 and 1.
    """
    ancestors = self.tree.get_ancestor(5)
    # Verify that both 1 and 2 appear in the ancestor list for node 5.
    self.assertTrue(all([a in ancestors for a in [1, 2]]))

  def testChildLeaves(self):
    """Test that get_child_leaves returns the correct descendant leaves.

    For internal node 3, the child leaves should be leaf nodes 6 and 7.
    """
    leaves = self.tree.get_child_leaves(3)
    # Check that 6 and 7 are present in the returned leaf list.
    self.assertTrue(all([c in leaves for c in [6, 7]]))

  def testFillWeights(self):
    """Test that node weights are filled or computed correctly.

    In this configuration, node 3 is expected to have a weight of 0.5,
    according to the weighting scheme used in the Tree implementation.
    """
    node = self.tree.get_node(3)
    self.assertEqual(node.weight, 0.5)

  def testGetPruning(self):
    """Test that get_pruning returns a valid pruning of the tree.

    Starting from the root node (1), the pruning should include nodes
    that represent the current structure of the tree. Here, we expect
    nodes 3, 4, and 5 to be part of the pruning set.
    """
    node = self.tree.get_node(1)
    pruning = self.tree.get_pruning(node)
    # Check that all expected nodes are in the pruning result.
    self.assertTrue(all([n in pruning for n in [3, 4, 5]]))

if __name__ == '__main__':
  # Run all unit tests in this module when executed as a script.
  unittest.main()
