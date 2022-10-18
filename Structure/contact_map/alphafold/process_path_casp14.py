import os
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
        print(index)
        feature_dict = pickle.load(fs)
        aa=os.path.split(index)[-1][:4]
        labels=np.load('pdb/'+aa.decode()[:4]+'/'+aa.decode()+'.npy')
        feature.append(model_runner.process_features_test(
            feature_dict, random_seed=random.randrange(sys.maxsize)))
    processed_feature_dict = features2parallel(feature)
    processed_feature_dict['ContactMap']=np.array(labels[np.newaxis,:])
    #sss=processed_feature_dict['ContactMap'].reshape(-1)
    #print(processed_feature_dict['druglabels'])
    return processed_feature_dict