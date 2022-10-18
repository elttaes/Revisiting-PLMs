import numpy as np
import jax.numpy as jnp
import pickle
import sys
import random
from alphafold.model import model
from alphafold.model import config
model_name = 'model_1'
model_config = config.model_config(model_name)
model_runner = model.RunModel(model_config)

def features2parallel(features_processed):
    data_res0 = dict()
    for key in features_processed[0].keys():
        features0 = []
        for i in range(len(features_processed)):
            features0.append(jnp.array(features_processed[i][key]))
        features0 = jnp.array(features0)
        data_res0[key] = features0
    return data_res0

def process_path(path):
    feature=[]
    labels=[]
    for j, index in enumerate(path):
        feat_paths = np.char.decode(index, 'utf-8')
        fs = open(str(feat_paths), "rb")
        feature_dict = pickle.load(fs)
        seq_length=feature_dict['ssp'].shape[0]
        # feature_dict['ssp'] = jnp.array(feature_dict['ssp'])
        # feature_dict['ssp'] = feature_dict['ssp']+1
        feature_dict['dist'][feature_dict['dist'] < 8] = 1
        feature_dict['dist'][feature_dict['dist'] >= 8] = 0
        feature_dict['dist'][jnp.isnan(feature_dict['dist'])] = 0
        # feature_dict['dist']=feature_dict['dist'][:256,:256]
        # if(seq_length<256):
        #     padxy=256-seq_length
        #     feature_dict['dist']=np.pad(feature_dict['dist'],((0,padxy),(0,padxy)),'constant', constant_values=(0,0)) 
        #     dist_mask=np.ones((seq_length,seq_length))
        #     dist_mask=np.pad(dist_mask,((0,padxy),(0,padxy)),'constant', constant_values=(0,0)) 
        labels.append(feature_dict['dist'])
        feature.append(model_runner.process_features_test(
            feature_dict, random_seed=random.randrange(sys.maxsize)))
    processed_feature_dict = features2parallel(feature)
    processed_feature_dict['ContactMap']=np.array(labels)
    #print(processed_feature_dict['druglabels'])
    return processed_feature_dict