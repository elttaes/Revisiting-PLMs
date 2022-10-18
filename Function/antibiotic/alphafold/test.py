from jax._src.api import T
from alphafold.model import model
from alphafold.model import config
from absl import flags
from absl import app
import pickle
import os
import jax
import jax.profiler
import jax.numpy as jnp
from typing import Any, Mapping
import functools
import tensorflow as tf
from process_path import process_path
file_save='./test'
writer = tf.summary.create_file_writer(file_save)
model_params_dir='/home/bigdata/af2data/af2data'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '4'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
batch_sizes = jax.local_device_count()
flags.DEFINE_float('learning_rate', 1e-5, 'Max learning-rate')
flags.DEFINE_float('grad_clip_value', 1, 'Gradient norm clip value')
flags.DEFINE_string('checkpoint_dir', file_save,
                    'Directory to store checkpoints.')

FLAGS = flags.FLAGS
LOG_EVERY = 50
MAX_STEPS = 10**6
model_name = 'model_1'
model_config = config.model_config(model_name)
model_runner = model.RunModel(model_config)


def lm_loss_fn_predict(forward_fn,
                params,
                rng,
                data: Mapping[str, jnp.ndarray]
                ) -> jnp.ndarray:
    """Compute the loss on data wrt params."""
    logits = forward_fn(params, rng, data)#['logits']
    targets = jax.nn.one_hot(data['druglabels'], 19)
    logits=jnp.squeeze(logits)
    targets=jnp.squeeze(targets)
    assert logits.shape == targets.shape
    loss = -jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
    labels=data['druglabels']
    accuracy=jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss,accuracy

def datasets_test():
    features_dir='/home/bigdata/my/datasets/drug/test'
    dataset = tf.data.Dataset.list_files(
        os.path.join(features_dir, '*.pkl'))
    dataset = dataset.shuffle(buffer_size=1 * 1)
    dataset = dataset.batch(1)
    return iter(dataset.as_numpy_iterator())

def main(_):
    loss_fn_predict = functools.partial(lm_loss_fn_predict, model_runner.predict)
    test_acc=0
    rng = jax.random.PRNGKey(428)
    checkpoint='/root/drug/tmp/checkpoint.pkl'
    with open(checkpoint, 'rb') as f:
        state = pickle.load(f)
    print('start test')
    param=jax.tree_map(lambda x: x[0], (state['params']))
    test_data=datasets_test()
    for idx,batch in enumerate(test_data):
        data=process_path(batch)
        if(idx%256==0):
            print('test idx: {},acc: {}'.format(idx,test_acc/(idx+1)))
        param,data=jax.tree_map(lambda x: x[0], (state['params'],data))
        loss,acc=loss_fn_predict(param, rng, data)
        test_acc+=acc
        with writer.as_default():
            tf.summary.scalar("test_acc", test_acc/(idx+1), step=idx)
    print('test_acc:',test_acc/(idx+1))


if __name__ == '__main__':
    app.run(main)