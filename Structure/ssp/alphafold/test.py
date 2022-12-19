
from alphafold.model import model
from alphafold.model import config
from absl import app
import sys
import random
import pickle
import os
import jax
import jax.profiler
import numpy as np
import tensorflow.compat.v1 as tf
import jax.numpy as jnp
import checkpointSR
import logging
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

features_dir = 'test'
param_epoch = 3000
checkpoint_dir = 'PretrainParams'
model_name = 'model_1'
batch_size =  1
model_config = config.model_config(model_name)
model_runner = model.RunModel(model_config)


@jax.checkpoint
def main(argv):
    def spmd_update_inference(params, processed_feature_dict):
        prediction_result = model_runner.inference(
            params, processed_feature_dict)
        labels = processed_feature_dict['ssp'][0]
        logits = prediction_result[0]['ssp']['logits']
        aa = tf.argmax(logits, 1)
        accuracys = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))
        return accuracys.numpy()

    def datasets():
        dataset = tf.data.Dataset.list_files(
            os.path.join(features_dir, '*.pkl'))
        dataset = dataset.shuffle(buffer_size=1 * batch_size)
        dataset = dataset.batch(batch_size)
        return iter(dataset.as_numpy_iterator())

    test_set = datasets()

    epoch = 5000
    accuracy = 0
    for i in range(epoch):
        feature = []
        path = next(test_set)
        logger.info(i)
        for j, index in enumerate(path):
            feat_paths = np.char.decode(index, 'utf-8')
            fs = open(str(feat_paths), "rb")
            feature_dict = pickle.load(fs)
            feature_dict['ssp'] = jnp.array(feature_dict['ssp'])
            feature_dict['ssp'] = feature_dict['ssp']+1
            feature_dict['dist'][feature_dict['dist'] < 8] = 1
            feature_dict['dist'][feature_dict['dist'] >= 8] = 0
            feature_dict['dist'][jnp.isnan(feature_dict['dist'])] = 0
            feature = model_runner.process_features(
                feature_dict, random_seed=random.randrange(sys.maxsize), is_inference=True)
            if(i == 0 & j == 0):
                model_paramTest = checkpointSR.load_checkpoint(
                    checkpoint_dir+'/checkpoint_'+str(param_epoch))

        accuracy += spmd_update_inference(model_paramTest, feature)
        logger.info(accuracy/(i+1))


if __name__ == '__main__':
    app.run(main)
