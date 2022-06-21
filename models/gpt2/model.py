from os import path
import functools

import h5py
import jax
import jax.numpy as jnp
from jax._src import abstract_arrays
import numpy as np


# pylint: disable=invalid-name
def load_gpt2_model(name, gpt2_dir):
  """Load GPT-2 pretrained weights into JAX DeviceArrays."""
  with open(path.join(gpt2_dir, 'tf_model.h5'), 'rb') as backend:
    with h5py.File(backend, 'r') as f:
      root = f['transformer']['tfgp_t2lm_head_model']['transformer']
      params = []
      params.append(np.asarray(root['wte']['weight:0']))  # text embeddings
      params.append(np.asarray(root['wpe']['embeddings:0']))  # pos embeddings
      layer_params = []
      for i in range(1000):
        try:
          layer_root = root[f'h_._{i}']
        except KeyError:
          break
        xnorm_scale = np.asarray(layer_root['ln_1']['gamma:0'])
        xnorm_bias = np.asarray(layer_root['ln_1']['beta:0'])
        wqkv = np.asarray(layer_root['attn']['c_attn']['weight:0'])
        # hardcoded
        Q = 64
        wqkv = wqkv.reshape((wqkv.shape[0], 3, -1, Q)).transpose((1, 2, 3, 0))
        wqkv_bias = np.asarray(layer_root['attn']['c_attn']['bias:0']).reshape(
            (3, -1, Q))
        wo = np.asarray(layer_root['attn']['c_proj']['weight:0'])
        wo = wo.reshape((-1, Q, wo.shape[0]))
        wo_bias = np.asarray(layer_root['attn']['c_proj']['bias:0'])
        ynorm_scale = np.asarray(layer_root['ln_2']['gamma:0'])
        ynorm_bias = np.asarray(layer_root['ln_2']['beta:0'])
        w_i = np.asarray(layer_root['mlp']['c_fc']['weight:0'])
        w_i_bias = np.asarray(layer_root['mlp']['c_fc']['bias:0'])
        w_o = np.asarray(layer_root['mlp']['c_proj']['weight:0'])
        w_o_bias = np.asarray(layer_root['mlp']['c_proj']['bias:0'])
        layer_params.append(((xnorm_scale, xnorm_bias),
                             (wqkv, wqkv_bias),
                             (wo, wo_bias),
                             (ynorm_scale, ynorm_bias),
                             (w_i, w_i_bias),
                             (w_o, w_o_bias)))
      params.append(layer_params)
      fnorm_scale = np.asarray(root['ln_f']['gamma:0'])
      fnorm_bias = np.asarray(root['ln_f']['beta:0'])
      params.append((fnorm_scale, fnorm_bias))
  return jax.tree_map(jnp.asarray, params)


def init_layer(E, F, Q, H, dtype):
  """Initialize parameters with all ones."""
  xnorm_scale = jnp.ones((E,), dtype=dtype)
  xnorm_bias = jnp.ones((E,), dtype=dtype)
  wqkv = jnp.ones((3, H, Q, E), dtype=dtype)
  wqkv_bias = jnp.ones((3, H, Q), dtype=dtype)
  wo = jnp.ones((H, Q, E), dtype=dtype)
  wo_bias = jnp.ones((E,), dtype=dtype)
  ynorm_scale = jnp.ones((E,), dtype=dtype)
  ynorm_bias = jnp.ones((E,), dtype=dtype)
  w_i = jnp.ones((E, F), dtype=dtype)
  w_i_bias = jnp.ones((F,), dtype=dtype)
  w_o = jnp.ones((F, E), dtype=dtype)
  w_o_bias = jnp.ones((E,), dtype=dtype)
  return ((xnorm_scale, xnorm_bias),
          (wqkv, wqkv_bias),
          (wo, wo_bias),
          (ynorm_scale, ynorm_bias),
          (w_i, w_i_bias),
          (w_o, w_o_bias))


def init(L, E, F, Q, H, V, dtype):
  wte = jnp.ones((V, E), dtype=dtype)
  wpe = jnp.ones((1024, E), dtype=dtype)
  layer_params = [init_layer(E, F, Q, H, dtype) for l in range(L)]
  fnorm_scale = jnp.ones((E,), dtype=dtype)
  fnorm_bias = jnp.ones((E,), dtype=dtype)
  return (wte, wpe, layer_params, (fnorm_scale, fnorm_bias))


def init_kv(B, W, S, L, Q, H, dtype, abstract=False):
  if abstract:
    return [abstract_arrays.ShapedArray((2, B, W, S, H, Q), dtype=dtype) for l in range(L)]
  return [jnp.zeros((2, B, W, S, H, Q), dtype=dtype) for l in range(L)]


def fprop_layer(params, kv, x, t0, i, mask):
  """Run a single transformer layer."""
  ((xnorm_scale, xnorm_bias),
   (wqkv, wqkv_bias),
   (wo, wo_bias),
   (ynorm_scale, ynorm_bias),
   (w_i, w_i_bias),
   (w_o, w_o_bias)) = params
  # x = with_sharding_constraint(x, x_sharding)
  xnorm = jax.nn.normalize(x) * xnorm_scale + xnorm_bias
  qkv = jnp.einsum('bwte,ihqe->ibwthq', xnorm, wqkv)
  qkv = qkv + wqkv_bias[:, None, None, None]
  q, new_kv = qkv[0], qkv[1:]
  if i is not None:
    # "encoding" a single prefix
    if mask is not None:
      new_kv = mask[None, :, :, :, None, None] * new_kv
      q = q * mask[:, :, :, None, None]
    kv = jax.lax.dynamic_update_slice(kv, new_kv, [0, i, 0, 0, 0, 0])
    k, v = jax.lax.dynamic_slice(kv, [0, i, 0, 0, 0, 0], [2, new_kv.shape[1], new_kv.shape[2], *kv.shape[3:]])
  elif t0 is not None:
    # "decoding" a single timestep
    kv = jax.vmap(jax.lax.dynamic_update_slice,
                  (1, 1, [None, None, 0, None, None]),
                  1)(kv, new_kv, [0, 0, t0, 0, 0])
    k, v = kv
  else:
    # running a whole batch
    k, v = new_kv
  outer = jnp.einsum('bwthq,bwshq->bwtsh', q, k) / jnp.asarray(
      jnp.sqrt(v.shape[-1]), dtype=x.dtype)
  # s refers to timestep attended to; t refers to timestep attending
  s = jnp.arange(outer.shape[3])[None, None, None, :]
  t = (0 if t0 is None else t0[:, None, None, None]
       ) + jnp.arange(outer.shape[2])[None, None, :, None]
  if i is not None or t0 is not None:
    invalid = t < s
    outer = outer - jnp.asarray(
        jnp.inf, dtype=x.dtype) * invalid[:, :, :, :, None]
  alpha = jax.nn.softmax(outer, 3)
  inner = jnp.einsum('bwtsh,bwshq->bwthq', alpha, v)
  y = jnp.einsum('bwthq,hqe->bwte', inner, wo) + wo_bias + x
  # y = with_sharding_constraint(y, x_sharding)
  ynorm = jax.nn.normalize(y) * ynorm_scale + ynorm_bias
  act = jax.nn.gelu(jnp.einsum('bwte,ef->bwtf', ynorm, w_i) + w_i_bias)
  z = jnp.einsum('bwtf,fe->bwte', act, w_o) + w_o_bias + y
  # z = with_sharding_constraint(z, x_sharding)
  return kv, z


def embed(embedding, x):
  return jax.vmap(jax.vmap(
      lambda emb, inputs: emb[inputs],
      (None, 0), 0), (None, 0), 0)(embedding, x)


def fprop(params, kv, x, t0, i, mask):
  (wte, wpe, layer_params, (fnorm_scale, fnorm_bias)) = params
  x = embed(wte, x) + embed(wpe, (0 if t0 is None else t0[:, None, None]
                                  ) + jnp.arange(
                                      x.shape[2], dtype=x.dtype)[None, None, :])
  if mask is not None:
    x = jnp.where(mask[:, :, :, None], x, 0)
  for l in range(len(layer_params)):
    kv[l], x = jax.named_call(fprop_layer, name=f'L_{l}')(
        layer_params[l], kv[l], x, t0, i, mask)
  x = jax.nn.normalize(x) * fnorm_scale + fnorm_bias
  return kv, x

def beamsearch(x, wte, beams, old_score=None):
  logits = jnp.einsum('bwte,ve->bwtv', x, wte)

  if old_score is not None:
    logits = logits + old_score[:, :, :, None]

  w = logits.shape[1]
  t = logits.shape[2]
  v = logits.shape[3]

  logits = jnp.transpose(logits, (0, 2, 1, 3)) # btwv
  logits = jax.lax.collapse(logits, 2, 4) # bt[wv]
  print(f"logits shape (pre pad): {logits.shape} and bemas: {beams}")
  # logits shape: (1, 1, 50257) and bemas: 4
  blocking_size = 64
  pad_width = (blocking_size - (logits.shape[2] % blocking_size)) % logits.shape[2]
  # pad_width= 47 #Padding added at the end of logits to make it divisible by some nice number
  pad_width_list = [[0,0], [0,0], [0, pad_width]]
  padded_logits = jax.numpy.pad(logits, pad_width_list, mode="constant", constant_values=np.NINF)
  print(f"logits shape (after pad): {padded_logits.shape} and bemas: {beams}")
  score, ind = jax.lax.top_k(padded_logits, beams) # btw

  score = jnp.transpose(score, (0, 2, 1)) # bwt
  ind = jnp.transpose(ind, (0, 2, 1)) # bwt

  # Split the indices between which beam was selected and what the next token is.
  prev = ind // v
  nxt = ind % v
  return nxt, prev, score

def greedy(x, wte):
  logits = jnp.einsum('bwte,ve->bwtv', x, wte)
  assert logits.shape[1] == 1
  ind = jnp.argmax(logits, -1)
  score = logits[np.arange(logits.shape[0]), :, -1, ind]
  return ind, np.zeros(ind.shape, jnp.int32), score


@functools.partial(jax.jit, donate_argnums=1)
def encode(params, kv, prompt, i, t):
  iota = jnp.arange(prompt.shape[1])[None, None, :]
  length = t[:, None, None]
  mask = jnp.where(iota < length, 1, 0)
  
  # Initially we only have one beam to populate.
  prompt = prompt[:, None, :]

  kv, y = fprop(params, kv, prompt, jnp.array([0], dtype=jnp.int32), i, mask)
  y = y [jnp.arange(t.shape[0]), :, t-1, :]
  y = y[:, :, None, :]

  return kv, *beamsearch(y, params[0], kv[0].shape[2])
  # return kv, *greedy(y, params[0])

@functools.partial(jax.jit)
def decode(params, kv, x, y, s, t):
  _, _, T = x.shape
  assert T == 1

  # Select the appropriate beams
  kv = [m[:, jnp.arange(m.shape[1]), y[:, :, 0]] for m in kv]

  kv, y = fprop(params, kv, x, t, None, None)
  return kv, *beamsearch(y, params[0], kv[0].shape[2], s)


# order is (L=length, E=embed, F=ffn, Q=qkv, H=heads, V=vocab)
model_sizes = {
    'gpt2': (12, 768, 4*768, 64, 768//64, 50257),
    '355m': (24, 1024, 4*1024, 64, 1024//64, 51200),
    'gpt2-xl': (48, 1600, 4*1600, 64, 1600//64, 50257),
    '52b': (64, 8192, 4*8192, 64, 8192//64, 51200),
}
