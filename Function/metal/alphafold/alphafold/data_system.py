# Copyright 2021 Beijing DP Technology Co., Ltd.
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

"""Data system used to load training datasets."""

from absl import logging
import glob
import jax
import jax.numpy as jnp
import jax.random as jrand
import json
from multiprocessing import Process, Queue
import numpy as np
import os



FEATNAME_DICT = set(['aatype', 'residue_index', 'seq_length', 'template_aatype', 'template_all_atom_masks', 'template_all_atom_positions', 'template_sum_probs', 'is_distillation', 'seq_mask', 'msa_mask', 'msa_row_mask', 'random_crop_to_size_seed', 'template_mask', 'template_pseudo_beta', 'template_pseudo_beta_mask', 'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'extra_msa', 'extra_msa_mask', 'extra_msa_row_mask', 'bert_mask', 'true_msa', 'extra_has_deletion', 'extra_deletion_value', 'msa_feat', 'target_feat'])

def cast_to_precision(batch, precision):
  # the input batch is asserted of precision fp32.
  if precision == 'bf16':
    dtype = jnp.bfloat16
  elif precision == 'fp16':
    dtype = jnp.float16
  else:   # assert fp32 specified
    return batch
  for key in batch:
    # skip int type
    if batch[key].dtype in [np.int32, np.int64, jnp.int32, jnp.int64]:
      continue
    if 'feat' in key or 'mask' in key or key in FEATNAME_DICT:
      batch[key] = jnp.asarray(batch[key], dtype=dtype)
  return batch
