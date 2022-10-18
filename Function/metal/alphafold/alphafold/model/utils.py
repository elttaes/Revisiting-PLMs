# Copyright 2021 DeepMind Technologies Limited
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
"""A collection of JAX utility functions for use in protein folding."""

import collections
import numbers
from typing import Mapping

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def final_init(config):
    if config.zero_init:
        return 'zeros'
    else:
        return 'linear'


def batched_gather(params, indices, axis=0, batch_dims=0):
    """Implements a JAX equivalent of `tf.gather` with `axis` and `batch_dims`."""
    take_fn = lambda p, i: jnp.take(p, i, axis=axis)
    for _ in range(batch_dims):
        take_fn = jax.vmap(take_fn)
    return take_fn(params, indices)


def mask_mean(mask, value, axis=None, drop_mask_channel=False, eps=1e-10):
    """Masked mean."""
    if drop_mask_channel:
        mask = mask[..., 0]

    mask_shape = mask.shape
    value_shape = value.shape

    assert len(mask_shape) == len(value_shape)

    if isinstance(axis, numbers.Integral):
        axis = [axis]
    elif axis is None:
        axis = list(range(len(mask_shape)))
    assert isinstance(axis, collections.Iterable), (
        'axis needs to be either an iterable, integer or "None"')

    broadcast_factor = 1.
    for axis_ in axis:
        value_size = value_shape[axis_]
        mask_size = mask_shape[axis_]
        if mask_size == 1:
            broadcast_factor *= value_size
        else:
            assert mask_size == value_size

    return (jnp.sum(mask * value, axis=axis) /
            (jnp.sum(mask, axis=axis) * broadcast_factor + eps))


def flat_params_to_haiku(params: Mapping[str, np.ndarray]) -> hk.Params:
    """Convert a dictionary of NumPy arrays to Haiku parameters."""
    hk_params = {}
    count = 0
    for path, array in params.items():
        scope, name = path.split('//')
        if scope not in hk_params:
            hk_params[scope] = {}
        #这里因为evoformer是48层，所以如果遇见48层的就取出其中一层
        #但这里维度会降低，使用np.newaxis对其升个维度
        #因为param是通过npz进行保存的，numpy数据很容易处理
        #if (len(array) == 48):
            #hk_params[scope][name] = jnp.array((array[0])[np.newaxis,:])
            #continue
        hk_params[scope][name] = jnp.array(array)
        #print(array[0].shape)
        #print(len(array))
    return hk_params


def softmax(x):
    if x.dtype != jnp.float32:
        return jax.nn.softmax(x.astype(jnp.float32)).astype(x.dtype)
    else:
        return jax.nn.softmax(x)