from jax._src.api import T
from alphafold.model import model
from alphafold.model import config
from alphafold.model import data
from absl import logging
from absl import flags
from absl import app
import optax
import sys
import random
import pickle
import os
import jax
import jax.profiler
import numpy as np
from jax.lib import xla_bridge
from jax.tree_util import tree_map
import functools
from functools import partial
from jax import lax, grad, pmap
import haiku as hk
import jmp
import tensorflow as tf
import jax.numpy as jnp
import checkpointSR
import time
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '4'
#os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = "platform"
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = "5,6,2,7"
batch_size=4
# 训练只需要修改这两个路径
# 1.数据集路径DOWNLOAD_DIR
# 2.feature里面的 .pkl保存路径features_dir
data_dir = '../drug'
features_dir = './train'
params_dir = 'PretrainParams0322_msafirstrow'
checkpoint_dir = './'+params_dir+'/checkpoint'
writer = tf.summary.create_file_writer(checkpoint_dir)

if not (os.path.exists(params_dir)):
    os.makedirs(params_dir)
train_from_zero = True
restore_num=1000
# 是否开启训练以及是否计算tm-score
model_name = 'model_1'
#batch_size = xla_bridge.device_count()

FLAGS = flags.FLAGS
LEARNING_RATE = 1e-5

def softmax_cross_entropy(logits, labels):
    """Computes softmax cross entropy given logits and one-hot class labels."""
    loss = -jnp.sum(labels * jax.nn.log_softmax(logits), axis=-1)
    return jnp.asarray(loss)

model_config = config.model_config(model_name)
model_runner = model.RunModel(model_config)

def replicate_array(x): return np.broadcast_to(x, (batch_size,) + x.shape)

@jax.checkpoint
def main(argv):
    opt = optax.adamw(LEARNING_RATE, b1=0.9, b2=0.99)

    model_config.data.eval.num_ensemble = 1
    if(train_from_zero==False):
        model_params = checkpointSR.load_checkpoint(checkpoint_dir+'_'+str(restore_num))
        replicated_params = tree_map(replicate_array, model_params)
        opt_state = opt.init(replicated_params)
    
    @partial(jax.pmap, axis_name='num_devices')
    def spmd_update(trainable_params, non_trainable_params,processed_feature_dict):
        trainable_params_grads = jax.grad(model_runner.train_classify,allow_int=True)(trainable_params, non_trainable_params, processed_feature_dict)
        grads = jax.lax.pmean(trainable_params_grads, axis_name='num_devices')
        return grads

    def datasets():
        dataset = tf.data.Dataset.list_files(
            os.path.join(features_dir, '*.pkl'))
        dataset = dataset.shuffle(buffer_size=1 * batch_size)
        dataset = dataset.batch(batch_size)
        return iter(dataset.as_numpy_iterator())

    def features2parallel(features_processed):
        data_res0 = dict()
        for key in features_processed[0].keys():
            features0 = []
            for i in range(len(features_processed)):
                features0.append(jnp.array(features_processed[i][key]))
            features0 = jnp.array(features0)
            data_res0[key] = features0
        return data_res0

    epoch = 10000
    for i in range(epoch):
        if(i % 2900 == 0):
            train_set = datasets()
        if(train_from_zero==False):
            if(i<restore_num+2):
                continue
        feature = []
        path = next(train_set)
        print(path)
        print("epoch ", i)
        for j, index in enumerate(path):
            start_time=time.perf_counter()
            feat_paths = np.char.decode(index, 'utf-8')
            fs = open(str(feat_paths), "rb")
            feature_dict = pickle.load(fs)
            feature_dict['ssp'] = jnp.array(feature_dict['ssp'])
            feature_dict['ssp'] = feature_dict['ssp'] + 1
            feature_dict['dist'][feature_dict['dist'] < 8] = 1
            feature_dict['dist'][feature_dict['dist'] >= 8] = 0
            feature_dict['dist'][jnp.isnan(feature_dict['dist'])] = 0
            feature.append(model_runner.process_features(
                feature_dict, random_seed=random.randrange(sys.maxsize)))

            if(i==0 & j==0):
                if(train_from_zero==True):
                    model_params = model_runner.init_params_retern(
                    feature[0])
                    model_params_finetune = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)
                    for z in model_params_finetune:
                        if z in model_params:
                            model_params[z] = model_params_finetune[z]
                    replicated_params = tree_map(replicate_array, model_params)
                    trainable_params, non_trainable_params = hk.data_structures.partition(
    lambda m, n, p: m == "alphafold/alphafold_iteration/ssp_msa_head/logits", replicated_params)
                    opt_state = opt.init(trainable_params)

        processed_feature_dict = features2parallel(feature)
        data_time=time.perf_counter()
        print("data_time",round(data_time-start_time,2))
        grads = spmd_update(trainable_params, non_trainable_params, processed_feature_dict)
        updates, opt_state = opt.update(grads, opt_state,trainable_params)
        trainable_params = optax.apply_updates(trainable_params, updates)
        train_time=time.perf_counter()
        print("train_time",round(train_time-data_time,2))
        if(i % 500 == 0):
            replicated_params = hk.data_structures.merge(trainable_params, non_trainable_params)
            params,processed_feature_dict_inference = jax.tree_map(lambda x: x[0], (replicated_params, processed_feature_dict))
            checkpointSR.save_checkpoint(params, checkpoint_dir+'_'+str(i))

if __name__ == '__main__':
    app.run(main)
