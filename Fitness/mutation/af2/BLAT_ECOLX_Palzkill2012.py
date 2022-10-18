import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
from jax._src.api import T
from alphafold.model import model
from alphafold.model import config
from absl import flags
from absl import app
import pickle
import jax
import jax.profiler
import numpy as np
import tensorflow as tf
from alphafold.model import data
import constants
import scipy
from pkl.BLAT_ECOLX_Palzkill2012 import seq,res
file_save='./tmp'
#shutil.rmtree(file_save,True)
writer = tf.summary.create_file_writer(file_save)
model_params_dir='/home/my/bigdata/my/af2data'
os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '4'
#os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
batch_sizes = jax.local_device_count()
#batch_sizes = jax.lib.xla_bridge.device_count()
#flags.DEFINE_integer('batch_size', 8, 'Train batch size per core')

FLAGS = flags.FLAGS
LOG_EVERY = 256
MAX_STEPS = 10**6
# Set up the model, loss, and updater.
model_name = 'model_1'
mixed_precision='fp32'
model_config = config.model_config(model_name)
model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir='/home/my/nips/mutation/alphafold')
model_runner = model.RunModel(model_config,model_params)

def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.mstats.spearmanr(target_array, prediction_array).correlation

def main(_):
    mutation_num="alphafold/pkl/BLAT_ECOLX_Palzkill2012"
    feat_paths=mutation_num+".pkl"
    #feat_paths="data/pkl/BLAT_ECOLX_Ostermeier2014.pkl"
    fs = open(str(feat_paths), "rb")
    feature_dict = pickle.load(fs)
    data =model_runner.process_features(
        feature_dict, random_seed=428)
    result=model_runner.predict(data,42)['masked_msa']['logits'][0]
    result=jax.nn.log_softmax(result)
    seq_res=[]
    exp_res=[]
    for i in range(len(seq)):
        #print(result[i][constants.HHBLITS_AA_TO_ID[seq[i][-1]]])
        count=int(seq[i][1:-1])-24
        seq_res.append(result[count][constants.HHBLITS_AA_TO_ID[seq[i][-1]]]-result[count][constants.HHBLITS_AA_TO_ID[seq[i][0]]])
    for i in range(len(res)):
        exp_res.append(float(res[i]))
    print(spearmanr(seq_res,exp_res))

if __name__ == '__main__':
    app.run(main)