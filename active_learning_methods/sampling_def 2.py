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

"""Abstract class for sampling methods.

Provides interface to sampling methods that allow same signature
for select_batch.  Each subclass implements select_batch_ with the desired
signature for readability.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np

class SamplingMethod(object):
  """Base/abstract class for all active-learning sampling methods.

  Subclasses must:
    * Implement __init__(self, X, y, seed, **kwargs) to store data and config.
    * Implement select_batch_(...) that performs the actual selection logic.
  The public interface is select_batch(...), which simply forwards to
  select_batch_(...); this lets different subclasses have slightly different
  parameter lists while keeping a common entry-point.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    # X: full pool of data points (features).
    # y: labels (may be partially known or only used for configuration).
    # seed: random seed for reproducibility if subclasses need randomness.
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    """Flatten high-dimensional inputs into 2D (n_samples, n_features).

    Many sampling methods assume a 2D feature matrix. If X has more than
    2 dimensions (e.g., images with shape [N, H, W, C]), this method
    collapses all non-batch dimensions into a single feature vector.
    """
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    """Abstract internal selection method.

    Each subclass defines its own signature and logic for choosing
    which indices to query next. The public wrapper select_batch(...)
    simply forwards keyword arguments to this function.
    """
    return

  def select_batch(self, **kwargs):
    """User-facing selection method.

    This keeps a consistent entry-point for all samplers while allowing
    subclasses to customize their internal select_batch_ signature.
    """
    return self.select_batch_(**kwargs)

  def to_dict(self):
    """Optional hook for exporting diagnostic information.

    Subclasses can override this to return stats, internal state,
    or anything else that might be useful for analysis or logging.
    """
    return None
