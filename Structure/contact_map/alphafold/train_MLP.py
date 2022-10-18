from jax._src.api import T
from alphafold.model import model
from alphafold.model import data as param_loader
from alphafold.model import config
from absl import logging
from absl import flags
from absl import app
import optax
import pickle
import os
import jax
import jax.profiler
import numpy as np
import jax.numpy as jnp
import haiku as hk
#from alphafold.train.data_system import cast_to_precision
import time
from typing import List, Tuple, Any
from typing import Any, Mapping
import functools
import shutil
import tensorflow as tf
from process_path import process_path
file_save='./tmp/MLP'
#shutil.rmtree(file_save,True)
writer = tf.summary.create_file_writer(file_save)
model_params_dir='/home/public/bigdata/af2data/af2data'
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
    logits = forward_fn(params1,params2, rng, data)
    logits = (logits + jnp.transpose(logits,[1,0,2])) / 2
    targets = data['ContactMap']
    targets=jax.nn.one_hot(targets.astype(jnp.int32),2)
    assert logits.shape == targets.shape
    loss=optax.softmax_cross_entropy(logits=logits,labels=targets)
    return jnp.mean(loss)

def lm_loss_fn_predict(forward_fn,
                params,
                rng,
                data: Mapping[str, jnp.ndarray]
                ) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data)
    logits = (logits + jnp.transpose(logits,[1,0,2])) / 2
    targets = data['ContactMap']
    #print(data['metal'])
    #logits=jnp.squeeze(logits)
    onehot_targets=jax.nn.one_hot(targets.astype(jnp.int32),2)
    #print(logits.shape,targets.shape)
    #assert logits.shape == targets.shape
    loss=jnp.mean(optax.softmax_cross_entropy(logits=logits,labels=onehot_targets))
    L5_accuracy=compute_precision_at_l5(logits, targets)

    return loss,L5_accuracy

def compute_precision_at_l5(prediction, labels):
    sequence_lengths=prediction.shape[0]
    valid_mask = labels != -1
    seqpos = jnp.arange(sequence_lengths)
    y_ind,x_ind = jnp.meshgrid(seqpos,seqpos)
    valid_mask &= jnp.squeeze(((y_ind - x_ind) >= 6))#.unsqueeze(0)
    correct = 0
    total = 0
    prediction_1=jax.nn.softmax(prediction)[:,:,1]
    masked_prob = (prediction_1 * valid_mask)#.reshape(-1)
    #print(masked_prob)
    masked_prob=masked_prob.reshape(-1)
    most_likely,selected_position=jax.lax.top_k(masked_prob,sequence_lengths)
    #most_likely = masked_prob.top_k(100 // 5, sorted=False)
    selected = labels.reshape(-1).take(selected_position)
    # print("most_likely",most_likely)
    # print("selected",selected)
    # print("selected.shape[0]",selected.shape[0])
    correct += jnp.sum(selected)
    total += selected.shape[0]
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
                print("this param in model ",z)
                params[z] = pretrain_params[z]
            else:
                print("this param not in model ",z)
        trainable_params, non_trainable_params = hk.data_structures.partition(lambda m, n, p: m == "alphafold/alphafold_iteration/contact_head/logits", params)
        opt_state = self._opt.init(trainable_params)
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
        #print(rng,new_rng)
        params = state['params']
        trainable_params, non_trainable_params = hk.data_structures.partition(lambda m, n, p: m == "alphafold/alphafold_iteration/contact_head/logits", params)
        g = jax.grad(self._loss_fn)(trainable_params,non_trainable_params, rng, data)
        g = jax.lax.pmean(g, axis_name='p')
        
        updates, opt_state = self._opt.update(g, state['opt_state'],trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        params = hk.data_structures.merge(trainable_params, non_trainable_params)
        new_state = {
            'step': state['step'] + 1,
            'rng': new_rng,
            'opt_state': opt_state,
            'params': params,
        }

        metrics = {
            'step': state['step'],
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
                checkpoint_every_n: int = 64):
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

def datasets_train():
    features_dir='./datasets/ssp/train'
    dataset = tf.data.Dataset.list_files(
        os.path.join(features_dir, '*.pkl'))
    dataset = dataset.shuffle(buffer_size=1 * batch_sizes)
    dataset = dataset.batch(batch_sizes)
    return iter(dataset.as_numpy_iterator())

def datasets_test():
    features_dir='./datasets/ssp/test'
    dataset = tf.data.Dataset.list_files(
        os.path.join(features_dir, '*.pkl'))
    dataset = dataset.shuffle(buffer_size=1 * 1)
    dataset = dataset.batch(1)
    return iter(dataset.as_numpy_iterator())

def main(_):
    loss_fn = functools.partial(lm_loss_fn, model_runner.predict_MLP)
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
    train_step=2920
    logging.info('Starting train loop...')
    prev_time = time.time()
    for step in range(30000):
        train_set = datasets_train()
        print('start train epoch ',step)
        for idx,batch in enumerate(train_set):
            
            data=process_path(batch)
            state, metrics = updater.update(state, data)
            if (step+1)*idx % 64 == 0:
                param,data=jax.tree_map(lambda x: x[0], (state['params'],data))
                loss,acc=loss_fn_predict(param, rng, data)
                steps_per_sec = LOG_EVERY / (time.time() - prev_time)
                prev_time = time.time()
                metrics.update({'steps_per_sec': steps_per_sec})
                with writer.as_default():
                    tf.summary.scalar("loss", loss, step=metrics['step'][0])
            # print('train ok')
            # break
        test_acc=0
        print('start test')
        test_data=datasets_test()
        for idx,batch in enumerate(test_data):
            data=process_path(batch)
            if(idx==512):
                print('test idx: {},acc: {}'.format(idx,test_acc/(idx+1)))
                break
            param,data=jax.tree_map(lambda x: x[0], (state['params'],data))
            loss,acc=loss_fn_predict(param, rng, data)
            test_acc+=acc
        print('test_acc:',test_acc/(idx+1))
        with writer.as_default():
            tf.summary.scalar("test_acc", test_acc/(idx+1), step=step)

if __name__ == '__main__':
    app.run(main)