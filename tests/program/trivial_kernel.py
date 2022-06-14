# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# RUN: %PYTHON %s | FileCheck %s

from collections import namedtuple
import logging

import jax
import jax.numpy as jnp
import numpy as np

from iree.jax import kernel, like, Program

logging.basicConfig(level=logging.DEBUG)

_x = jnp.ones((4, 1024), jnp.float32)


class TrivialKernel(Program):
  def test_topk(self, x=like(_x)):
    return self._topk(x)

  @kernel
  def _topk(x):
    return jax.lax.top_k(x, 10)


# CHECK: module @trivial_kernel
m = TrivialKernel()
print(Program.get_mlir_module(m))

x = np.random.random(_x.shape).astype(_x.dtype)
print("Run:", [a.to_host() for a in m.test_topk(x)])

