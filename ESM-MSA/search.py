import sys

import faiss
import os
import numpy as np
import torch
from math import ceil
from model import MSARetrieveModel
from utils import TimeCounter
from vector_construction import fasta2vec
from Results.format import *


def search(q_vec_path, db_index_path, iter_num=2, t=1, nprobe=1, min_num=0, max_num=500, verbose=True):
	
	with TimeCounter(f"Loading {db_index_path}...", verbose):
		index = faiss.read_index(db_index_path, faiss.IO_FLAG_MMAP|faiss.IO_FLAG_READ_ONLY)
		
		if max_num == 0:
			max_num = index.ntotal
			
		index.nprobe = index.ntotal if nprobe == 0 else nprobe
			
	q_vec = np.load(q_vec_path)
	print(q_vec.shape)
	# t = q_vec[:, -1:]
	with TimeCounter("Searching index...", verbose):
		dist, indices = index.search(np.ascontiguousarray(q_vec), max_num)
		dist = np.sqrt(dist)
		print(dist)

	if iter_num > 1:
		db_vectors = index.reconstruct_n(0, index.ntotal)

		for times in range(iter_num-1):
			cnt = np.sum(dist <= t, axis=1)
			for i, n in enumerate(cnt):
				if n != 0:
					num = min(n, 100)
					homo_vecs = db_vectors[indices[i][:num]]
					q_vec[i] = (np.sum(homo_vecs, axis=0) + q_vec[i]) / (homo_vecs.shape[0] + 1)

			with TimeCounter(f"Iteratively searching index...{times+2}", verbose):
				dist, indices = index.search(q_vec, max_num)
				dist = np.sqrt(dist)
	
	cnt = np.sum(np.bitwise_and(dist <= t, dist >= 0), axis=1)
	# min num limitation
	cnt[cnt < min_num] = min_num
	
	homo_list = [indices[i, :n] for i, n in enumerate(cnt)]
	dist_list = [dist[i, :n] for i, n in enumerate(cnt)]

	return homo_list, dist_list


def pipeline(q_vec_path, db_index_path, q_info_path, db_pointer_path,
             db_path, save_path, iter_num=1, t=1, nprobe=1, min_num=0, max_num=500, verbose=True):

	homo_list, dist_list = search(q_vec_path, db_index_path, iter_num, t, nprobe, min_num, max_num, verbose)
	with TimeCounter("output results...", verbose):
		write_to_tsv(q_info_path, db_pointer_path, db_path, save_path, homo_list, dist_list, max_num)


if __name__ == '__main__':
	# 对查询序列构建索引库和向量库
	device_ids = [0, 1, 2, 3]
	model_path = "PretrainedModels/RetrieveModel_scope2.07_p1_n5_m4_I17100_B256.pt"
	q_seq_path = 'Data/test_scope2.07.faa'
	save_path = 'Faiss/test_scope2.07'
	# fasta2vec(device_ids, model_path, q_seq_path, save_path)

	q_vec_path = f'{save_path}.npy'
	q_info_path = f'{save_path}_index.tsv'
	db_index_path = "Faiss/uniclust30_2018_08_seed_Flat.index"
	db_info_path = "Faiss/uniclust30_2018_08_seed_index.tsv"

	# 查询向量
	save_path = 'Results/test_scope2.07_res_Flat.tsv'
	# pipeline(q_vec_path, db_index_path, q_info_path, db_info_path, save_path, iter_num=2, t=1, gpu=False)

	msa_path = "Results"
	name = "query"
	tsv_to_msa(save_path, msa_path, name)