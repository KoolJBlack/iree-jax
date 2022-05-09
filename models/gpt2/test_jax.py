"""
Runs the GPT-2 model end-to-end using the normal JAX api

This is primarily for testing / validating step by step against the normal XLA
backend.
"""

from absl.testing import absltest
from absl.testing import parameterized
from os import path

import absl
import builtins
import itertools
import jax
import jax.numpy as jnp
import numpy
import pathlib
import transformers

# Local import.
import config
import model

FLAGS = absl.flags.FLAGS

def load_gpt2_tokenizer():
  builtins.open, tmp_open = open, builtins.open
  gpt2_dir = FLAGS.assets_path
  tokenizer = transformers.GPT2Tokenizer(
      vocab_file=path.join(gpt2_dir, 'vocab.json'),
      merges_file=path.join(gpt2_dir, 'merges.txt'))
  builtins.open = tmp_open
  return tokenizer

class GPT2RealWeightsTest(parameterized.TestCase):

  def setUp(self):
    gpt2_dir = FLAGS.assets_path
    self.tokenizer = load_gpt2_tokenizer()
    self.model_name = 'gpt2'
    self.params = model.load_gpt2_model(self.model_name, gpt2_dir)
    super().setUp()

  @parameterized.parameters(*itertools.product(
    ["cpu"]))
  def test_batch_one(self, backend):
    dtype = jnp.float32
    S = 64
    if dtype == jnp.bfloat16 and jax.devices()[0].device_kind.lower() == 'cpu':
      self.skipTest('bf16 decoding on CPU is broken')
    B, T = 1, 1  # T is one decode step; S is encoding/cache length
    W = 4 # W is the beam search width
    L, _, _, Q, H, _ = model.model_sizes[self.model_name]
    params = jax.tree_map(lambda x: jnp.asarray(x, dtype=dtype), self.params)
    kv = model.init_kv(B, W, S, L, Q, H, dtype=dtype)  # assume same dtype for now

    encode = jax.jit(model.encode, backend=backend)
    decode = jax.jit(model.decode, backend=backend)

    # string = 'zero one two three four'
    string = 'when in the course'
    prompt = jnp.array(self.tokenizer(string)['input_ids'])[None, :]
    t = jnp.array([prompt.shape[1]], dtype=jnp.int32)
    t0 = jnp.zeros((B,), dtype=jnp.int32)
    t0 = t0.at[0].set(prompt.shape[1])

    kv, x0, y0, score = encode(params, kv, prompt, 0, t)
    # x0 = numpy.asarray(x0)
    print("Encode Results")
    print(y0)
    print(", ".join([f'"{self.tokenizer.decode(int(x))}"' for x in numpy.nditer(x0)]))
    print(score)

    kv, x1, y1, score = decode(params, kv, x0, y0, score, t0 + 0)
    print("Decode Results")
    print(y1)
    print(", ".join([f'"{self.tokenizer.decode(int(x))}"' for x in numpy.nditer(x1)]))
    print(score)

    kv, x2, y2, score = decode(params, kv, x1, y1, score, t0 + 1)
    print("Decode Results")
    print(y2)
    print(", ".join([f'"{self.tokenizer.decode(int(x))}"' for x in numpy.nditer(x2)]))
    print(score)

if __name__ == '__main__':
  absltest.main()
