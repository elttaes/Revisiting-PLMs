from jax._src.api import T
from alphafold.model import model
from alphafold.model import data as param_loader
from alphafold.model import config
from absl import logging
from absl import flags
from absl import app
import optax
#import sys
#import random
import pickle
import os
import jax
#import jax.profiler
import numpy as np
import jax.numpy as jnp
#from alphafold.train.data_system import cast_to_precision
import time
#import torch
#import scipy
from typing import List, Tuple, Any
from typing import Any, Mapping
import functools
import shutil
import tensorflow as tf
from process_path_casp14 import process_path
file_save = './tmp/MLP'
model_params_dir='/home/public/bigdata/af2data/af2data'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
#batch_sizes = jax.local_device_count()
batch_sizes = jax.lib.xla_bridge.device_count()
#flags.DEFINE_integer('batch_size', 8, 'Train batch size per core')
flags.DEFINE_float('learning_rate', 1e-5, 'Max learning-rate')
flags.DEFINE_float('grad_clip_value', 1, 'Gradient norm clip value')
flags.DEFINE_string('checkpoint_dir', file_save,
                    'Directory to store checkpoints.')

FLAGS = flags.FLAGS
LOG_EVERY = 256
MAX_STEPS = 10**6
# Set up the model, loss, and updater.
model_name = 'model_1'
model_config = config.model_config(model_name)
model_runner = model.RunModel(model_config)

def lm_loss_fn_predict(forward_fn,
                       params,
                       rng,
                       data: Mapping[str, jnp.ndarray]
                       ) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data)  # ['logits']
    logits = (logits + jnp.transpose(logits, [1, 0, 2])) / 2
    targets = data['ContactMap']
    # print(data['metal'])
    # logits=jnp.squeeze(logits)
    onehot_targets = jax.nn.one_hot(targets.astype(jnp.int32), 2)
    # print(logits.shape,targets.shape)
    #assert logits.shape == targets.shape
    loss = jnp.mean(optax.softmax_cross_entropy(
        logits=logits, labels=onehot_targets))
    L_accuracy = compute_precision_at_l5(logits, targets)
    return loss, L_accuracy

def compute_precision_at_l5(prediction, labels):
    sequence_lengths = prediction.shape[0]
    valid_mask = labels != -1
    seqpos = jnp.arange(sequence_lengths)
    y_ind, x_ind = jnp.meshgrid(seqpos, seqpos)
    valid_mask &= jnp.squeeze(((y_ind - x_ind) >= 6))  # .unsqueeze(0)
    correct = 0
    total = 0
    prediction_1 = jax.nn.softmax(prediction)[:, :, 1]
    #jnp.save("7WED_prediction_1.npy", prediction_1)
    masked_prob = (prediction_1 * valid_mask)  # .reshape(-1)
    #jnp.save("7WED_masked_prob.npy", masked_prob)
    masked_prob = masked_prob.reshape(-1)
    #labels_1=labels.reshape(-1)
    most_likely, selected_position_L = jax.lax.top_k(
        masked_prob, sequence_lengths)
    selected_L = labels.reshape(-1).take(selected_position_L)
    correct = jnp.sum(selected_L)
    total = selected_L.shape[0]
    print("L:",correct / total)

    most_likely, selected_position_L2 = jax.lax.top_k(
        masked_prob, sequence_lengths//2)
    selected_L2 = labels.reshape(-1).take(selected_position_L2)
    correct = jnp.sum(selected_L2)
    total = selected_L2.shape[0]
    print("L2:",correct / total)
    
    most_likely, selected_position_L5 = jax.lax.top_k(
        masked_prob, sequence_lengths//5)
    selected_L5 = labels.reshape(-1).take(selected_position_L5)
    correct = jnp.sum(selected_L5)
    total = selected_L5.shape[0]
    print("L5:",correct / total)
    return(correct / total)


class Updater:
    """A stateless abstraction around an init_fn/update_fn pair.

    This extracts some common boilerplate from the training loop.
    """
    def __init__(self, net_init, loss_fn,
                 optimizer: optax.GradientTransformation):
        self._net_init = net_init
        self._loss_fn = loss_fn
        self._opt = optimizer

    def init(self, rng, data):
        """Initializes state of the updater."""
        out_rng, init_rng = jax.random.split(rng)
        params = self._net_init(feat=data)
        pretrain_params = param_loader.get_model_haiku_params(
            model_name=model_name, data_dir=model_params_dir)
        for z in pretrain_params:
            if z in params:
                print("this param in model ", z)
                params[z] = pretrain_params[z]
            else:
                print("this param not in model ", z)
        opt_state = self._opt.init(params)
        out = dict(
            step=np.array(0),
            rng=out_rng,
            opt_state=opt_state,
            params=params,
        )
        return out

    def update(self, state: Mapping[str, Any], data: Mapping[str, jnp.ndarray]):
        """Updates the state using some data and returns metrics."""
        rng, new_rng = jax.random.split(state['rng'])
        # print(rng,new_rng)
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)
        g = jax.lax.pmean(g, axis_name='p')
        updates, opt_state = self._opt.update(g, state['opt_state'], params)
        params = optax.apply_updates(params, updates)

        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            'loss': loss,
        }
        return new_state, metrics


class CheckpointingUpdater:
    """A didactic checkpointing wrapper around an Updater.

    A more mature checkpointing implementation might:
        - Use np.savez() to store the core data instead of pickle.
        - Not block JAX async dispatch.
        - Automatically garbage collect old checkpoints.
    """

    def __init__(self,
                 inner: Updater,
                 checkpoint_dir: str,
                 checkpoint_every_n: int = 20000):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n

    def _checkpoint_paths(self):
        return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint_' in p]

    def init(self):
        checkpoint = os.path.join(self._checkpoint_dir,
                                    max(self._checkpoint_paths()))
        logging.info('Loading checkpoint from %s', checkpoint)
        with open(checkpoint, 'rb') as f:
            state = pickle.load(f)
        return state

def datasets_test():
    features_dir = './pdb/6UF2'
    dataset = tf.data.Dataset.list_files(
        os.path.join(features_dir, '*.pkl'))
    dataset = dataset.shuffle(buffer_size=1 * 1)
    dataset = dataset.batch(1)
    return iter(dataset.as_numpy_iterator())

def main(_):
    loss_fn_predict = functools.partial(
        lm_loss_fn_predict, model_runner.predict)

    optimizer = optax.chain(
        optax.adamw(FLAGS.learning_rate, b1=0.9, b2=0.99)
    )
    updater = Updater(model_runner.init_params, loss_fn_predict, optimizer)
    updater = CheckpointingUpdater(updater, FLAGS.checkpoint_dir)
    
    logging.info('loading parameters...')
    rng = jax.random.PRNGKey(428)
    state = updater.init()
    print('start test')
    test_data = datasets_test()
    acc_pool = []
    param = jax.tree_map(lambda x: x[0], (state['params']))
    for idx, batch in enumerate(test_data):
        data = process_path(batch)
        data = jax.tree_map(lambda x: x[0], (data))
        loss, acc = loss_fn_predict(param, rng, data)
        acc_pool.append(acc)


if __name__ == '__main__':
    app.run(main)
