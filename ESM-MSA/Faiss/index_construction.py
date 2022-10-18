import os.path

import faiss
import re
import pandas as pd
import torch
import esm
import numpy as np
from Bio import SeqIO
from math import ceil
from model import MSARetrieveModel
from vector_construction import fasta2vec
from utils import TimeCounter
from LoadingData.data_construction import construct_pointer


# 构建Faiss索引
def contruct_faiss_index(vectors, path, dim, measure, param):
	index = faiss.index_factory(dim, param, measure)
		
	if not index.is_trained:
		index.train(vectors)
	index.add(vectors)
	
	faiss.write_index(index, path)


if __name__ == '__main__':
	# model_name = "RetrieveModel_d1_m1_t4_I1211_B256_pku"
	model_list = ["RetrieveModel_t17_I430_B256_sim_p1_n10"]
	
	for model_name in model_list:
		params = {"device_ids": [0, 1, 2, 3],
		          "model_path": f"../PretrainedModels/{model_name}.pt",
		          "fasta_path": f"/sujin/TwinTowers/task_eval/trRosetta/trRosetta_test_db.faa",
		          "save_path":  None,
		          "batch_size": 64,
		          'type': 'database',
		          "cover": True}
		# vectors = fasta2vec(**params)
		dim, measure = 1280, faiss.METRIC_L2
		param = f'Flat'
		save_path = f"../task_eval/trRosetta/{model_name}.index"
		# with TimeCounter("Constructing index..."):
		# 	contruct_faiss_index(vectors, save_path, dim, measure, param)
	
	# fasta_path = f"/sujin/TwinTowers/task_eval/pku/pku_test_db.faa"
	# pointer_path = f"/sujin/TwinTowers/task_eval/pku/pku_pointer.tsv"
	#
	# with TimeCounter("Constructing pointer..."):
	# 	construct_pointer(fasta_path, pointer_path)
	#
	file_names = [f'{i}_{i+100}' for i in range(0, 1100, 100)]
	for name in file_names:
		for i in range(10):
			print(name, i, flush=True)
			params['fasta_path'] = f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}_{i}.faa"
			params['save_path'] = f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}_p{i}"
			if os.path.exists(f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}_p{i}.npy"):
				continue

			fasta2vec(**params)

			fasta_path = params['fasta_path']
			pointer_path = f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}_p{i}_pointer.tsv"
			if not os.path.exists(pointer_path):
				with TimeCounter("Constructing pointer..."):
					construct_pointer(fasta_path, pointer_path)

	#
	names = [f'{i}_{i+100}' for i in range(0, 1100, 100)]
	for name in names:
		for i in range(10):
			part_num = i
			path = "/sujin/dataset/uniclust30/UniRef30_2020_06"
			print(name, part_num, flush=True)
			vec_path = f"{path}_{name}/UniRef30_2020_06_{name}_p{part_num}.npy"
			if os.path.exists(vec_path):
				save_path = f"{path}_{name}/UniRef30_2020_06_{name}_p{part_num}_IVF_Flat.index"
				if os.path.exists(save_path):
					continue

				vectors = np.load(vec_path)
				print(vectors.shape[0], flush=True)
				vectors[np.isnan(vectors)] = 0
				n = ceil(vectors.shape[0] / 256)
				dim, measure = 1280, faiss.METRIC_L2
				param = f'IVF{n}, Flat'
				print(param, flush=True)
				with TimeCounter("Constructing index..."):
					contruct_faiss_index(vectors, save_path, dim, measure, param)
	
	# for i in [5, 6, 7, 8, 9]:
	# 	vectors = np.load(f"/sujin/dataset/uniclust30/UniRef30_2020_06_100_200/UniRef30_2020_06_100_200_p{i}.npy")
	# 	print(vectors.shape)
	# 	length = np.linalg.norm(vectors, axis=1, keepdims=True)
	# 	print(length.shape)
	# 	res = vectors / length
	# 	print(np.linalg.norm(res[0], 2))
	# 	np.save(f"/sujin/dataset/uniclust30/UniRef30_2020_06_100_200/UniRef30_2020_06_100_200_p{i}.npy", res)