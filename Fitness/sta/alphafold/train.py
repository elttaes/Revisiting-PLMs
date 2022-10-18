from jax._src.api import T
from alphafold.model import model
from alphafold.model import data as param_loader
from alphafold.model import config
from absl import logging
from absl import flags
from absl import app
import optax
import sys
import random
import pickle
import haiku as hk
import os
import jax
import jax.profiler
import numpy as np
import jax.numpy as jnp
import time
import scipy
from typing import List, Tuple, Any
from typing import Any, Mapping
import functools
import tensorflow as tf
from process_path import process_path
file_save='./tmp/sta'
writer = tf.summary.create_file_writer(file_save)
model_params_dir='./af2data'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '4'
batch_sizes = jax.local_device_count()
#batch_sizes = jax.lib.xla_bridge.device_count()
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


def lm_loss_fn(forward_fn,
                params1,params2,
                rng,
                data: Mapping[str, jnp.ndarray],
                is_training: bool = True) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params1,params2, rng, data)
    targets = data['sta']
    logits=jnp.squeeze(logits)
    loss=optax.l2_loss(logits,targets)
    loss = jnp.sum(loss)
    return loss

def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.mstats.spearmanr(target_array, prediction_array).correlation

def lm_loss_fn_predict(forward_fn,
                params,
                rng,
                data: Mapping[str, jnp.ndarray]
                ) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data)
    targets = data['sta']
    logits=jnp.squeeze(logits)
    loss=optax.l2_loss(logits,targets)
    loss = jnp.sum(loss)
    #spear_value=spearmanr(logits,targets)
    return loss,logits,targets

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
                print("this param in model ",z)
                params[z] = pretrain_params[z]
            else:
                print("this param not in model ",z)
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
        params = state['params']
        loss, g = jax.value_and_grad(self._loss_fn)(params, rng, data)
        g = jax.lax.pmean(g, axis_name='p')
        updates, opt_state = self._opt.update(g, state['opt_state'],params)
        params = optax.apply_updates(params, updates)
        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
            #'loss': loss,
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
                checkpoint_every_n: int = 512):
        self._inner = inner
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_every_n = checkpoint_every_n

    def _checkpoint_paths(self):
        return [p for p in os.listdir(self._checkpoint_dir) if 'checkpoint_' in p]

    def init(self, rng, data):
        """Initialize experiment state."""
        if not os.path.exists(self._checkpoint_dir) or not self._checkpoint_paths():
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            return self._inner.init(rng, data)
        else:
            checkpoint = os.path.join(self._checkpoint_dir,
                                    max(self._checkpoint_paths()))
            logging.info('Loading checkpoint from %s', checkpoint)
            with open(checkpoint, 'rb') as f:
                state = pickle.load(f)
            return state

    def update(self, state, data):
        """Update experiment state."""
        step = np.array(state['step'])[0]
        #print(step)
        if step % self._checkpoint_every_n == 0:
            path = os.path.join(self._checkpoint_dir,
                                'checkpoint_{:07d}.pkl'.format(step))
            checkpoint_state = jax.device_get(state)
            logging.info('Serializing experiment state to %s', path)
            with open(path, 'wb') as f:
                pickle.dump(checkpoint_state, f)

        state, out = jax.pmap(self._inner.update, axis_name='p')(state, data)
        return state, out
    
    def save_checkpoint(self, state):
        """Update experiment state."""
        # NOTE: This blocks until `state` is computed. If you want to use JAX async
        # dispatch, maintain state['step'] as a NumPy scalar instead of a JAX array.
        # Context: https://jax.readthedocs.io/en/latest/async_dispatch.html
        step = np.array(state['step'])[0]
        #print(step)
        #if step % self._checkpoint_every_n == 0:
        path = os.path.join(self._checkpoint_dir,
                            'checkpoint_{:07d}.pkl'.format(step))
        checkpoint_state = jax.device_get(state)
        logging.info('Serializing experiment state to %s', path)
        with open(path, 'wb') as f:
            pickle.dump(checkpoint_state, f)

def datasets_train():
    features_dir='./fitness/sta/train/feature'
    dataset = tf.data.Dataset.list_files(
        os.path.join(features_dir, '*.pkl'))
    dataset = dataset.shuffle(buffer_size=1 * batch_sizes)
    dataset = dataset.batch(batch_sizes)
    return iter(dataset.as_numpy_iterator())

def datasets_test():
    features_dir='./fitness/sta/valid/feature'
    dataset = tf.data.Dataset.list_files(
        os.path.join(features_dir, '*.pkl'))
    dataset = dataset.shuffle(buffer_size=1 * 1)
    dataset = dataset.batch(1)
    return iter(dataset.as_numpy_iterator())

def main(_):
    loss_fn = functools.partial(lm_loss_fn, model_runner.predict)
    loss_fn_predict = functools.partial(lm_loss_fn_predict, model_runner.predict)
    
    train_set = datasets_train()
    path = next(train_set)
    data= process_path(path)
    
    optimizer = optax.chain(
        optax.adamw(FLAGS.learning_rate, b1=0.9, b2=0.99)
    )
    updater = Updater(model_runner.init_params, loss_fn, optimizer)
    updater = CheckpointingUpdater(updater, FLAGS.checkpoint_dir)

    logging.info('Initializing parameters...')
    rng = jax.random.PRNGKey(428)
    rng_pmap = jnp.broadcast_to(rng, (batch_sizes,) + rng.shape)

    state = jax.pmap(updater.init)(rng_pmap, data)

    train_step=3500
    logging.info('Starting train loop...')
    prev_time = time.time()
    best_spearmanr=0
    for step in range(30000):
        train_set = datasets_train()
        print('start train epoch ',step)
        for idx,batch in enumerate(train_set):
            data=process_path(batch)
            state, metrics = updater.update(state, data)
            if (step+1)*idx % LOG_EVERY == 0:
                metrics=jax.tree_map(lambda x: x[0], (metrics))
                steps_per_sec = LOG_EVERY / (time.time() - prev_time)
                prev_time = time.time()
                metrics.update({'epoch_step': metrics['step']%train_step})
                metrics.update({'steps_per_sec': steps_per_sec})
                logging.info({k: float(v) for k, v in metrics.items()})

        print('start test')
        test_data=datasets_test()
        pool_logits=[]
        pool_targets=[]
        for idx,batch in enumerate(test_data):
            data=process_path(batch)
            if(idx%256==0):
                print('test idx: {}'.format(idx))
            param,data=jax.tree_map(lambda x: x[0], (state['params'],data))
            loss,logits,targets=loss_fn_predict(param, rng, data)
            pool_logits.append(logits)
            pool_targets.append(targets)

        res = spearmanr(pool_logits,pool_targets)
        print("Spearman:",res)
        if(res > best_spearmanr):
            best_spearmanr=res
            updater.save_checkpoint(state)
        with writer.as_default():
            tf.summary.scalar("Spearman", res, step=step)

if __name__ == '__main__':
    app.run(main)