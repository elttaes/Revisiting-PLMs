import faiss
import os
import numpy as np
import torch
import pandas as pd
from model import MSARetrieveModel
from utils import TimeCounter
from vector_construction import fasta2vec
from search import pipeline


def twintower_eval(q_res_path, db_test_path):
	res = pd.read_csv(q_res_path, sep='\t')
	db = pd.read_csv(db_test_path, sep='\t')

	info_dict = {}
	for _, row in db.iterrows():
		cls, seq = row
		info_dict[seq] = cls
		if cls in info_dict.keys():
			info_dict[cls] += 1
		else:
			info_dict[cls] = 1

	total_P = 0
	total_N = 0
	total_TP = 0

	for q_id, group in res.groupby('query_id'):
		TP = 0
		cls = info_dict[q_id]
		P = info_dict[cls]
		# N = min(group.shape[0], P)
		N = group.shape[0]
		# if P <= 100:
		# 	continue
		
		total_P += P
		total_N += N
		
		for i, data in enumerate(group.values):
			if i == N:
				break
			hit = data[2]
			if info_dict[q_id] == info_dict[hit]:
				total_TP += 1
				TP += 1

		recall = TP / P * 100
		precision = TP / N * 100 if N != 0 else 0

		print(f"query_ID:{q_id}\tSF_ID:{cls:<10}recall:{f'{TP}/{P}({recall:.2f}%)':<20}"
		      f"precision:{f'{TP}/{N}({precision:.2f}%)':<20}")

	print(f'recall: {total_TP}/{total_P}({total_TP/total_P*100:.2f}%)')
	print(f"precision: {total_TP}/{total_N}({total_TP/total_N*100:.2f}%)")


# makeblastdb -in test_db_scope2.08.faa -dbtype prot -input_type fasta -out /home/public/sj/data/scope2.08
# psiblast -query test_scope2.07.faa -db /home/public/sj/data/scope2.07 -out query_blast.tsv -outfmt "6 qseqid sseqid" -num_iterations 2
def blast_eval(res_path, db_path):
	data = pd.read_csv(res_path, sep='\t', header=None, names=['SCOP_ID', 'HOMO_ID'])
	db = pd.read_csv(db_path, sep='\t')
	data.dropna(inplace=True)

	total_TP = 0
	total_P = 0
	total_N = 0
	l = len(data.groupby('SCOP_ID').indices.keys())
	for query_id, homo_ids in data.groupby('SCOP_ID').indices.items():
		sf_id = db[db['SCOP_ID'] == query_id]['SF_ID'].values[0]
		pred_homo_ids = set(data['HOMO_ID'].iloc[homo_ids].values)
		true_homo_ids = set(db[db['SF_ID'] == sf_id]['SCOP_ID'].values)

		P = len(true_homo_ids)
		N = len(pred_homo_ids)
		TP = len(true_homo_ids & pred_homo_ids)

		recall = TP / P * 100
		precision = TP / N * 100

		total_TP += TP
		total_P += P
		total_N += N

		print(f"query_ID:{query_id}\tFOLD_ID:{sf_id:<10} recall:{f'{TP}/{P}({recall:.2f}%)':<20}"
		      f"precision:{f'{TP}/{N}({precision:.2f}%)':<20}")

	print(f'recall: {total_TP}/{total_P}({total_TP / total_P * 100:.2f}%)')
	print(f"precision: {total_TP}/{total_N}({total_TP / total_N * 100:.2f}%)")


# jackhmmer -N 3 -A res.ato query.faa query_0.faa && reformat.pl sto a3m res.ato res.a3m
# hmmpy query_hmmer -o query_hmmer.tsv
def hmmer_eval(hmmer_path, db_path):
	data = pd.read_csv(hmmer_path, sep='\t')
	db = pd.read_csv(db_path, sep='\t')

	total_TP = 0
	total_P = 0
	total_N = 0
	indices = data.groupby('tlen').indices

	for query_id, homo_ids in indices.items():
		sf_id = db[db['SCOP_ID'] == query_id]['SF_ID'].values[0]
		pred_homo_ids = set(data['target name'].iloc[homo_ids].values)
		true_homo_ids = set(db[db['SF_ID'] == sf_id]['SCOP_ID'].values)

		P = len(true_homo_ids)
		N = len(pred_homo_ids)
		TP = len(true_homo_ids & pred_homo_ids)
		# if P <= 100:
		# 	continue
		recall = TP / P * 100
		precision = TP / N * 100

		total_TP += TP
		total_P += P
		total_N += N

		print(f"query_ID:{query_id}\tFOLD_ID:{sf_id:<10} recall:{f'{TP}/{P}({recall:.2f}%)':<20}"
		      f"precision:{f'{TP}/{N}({precision:.2f}%)':<20}")

	print(f'recall: {total_TP}/{total_P}({total_TP / total_P * 100:.2f}%)')
	print(f"precision: {total_TP}/{total_N}({total_TP / total_N * 100:.2f}%)")

# /mnt/beegfs/inspurfs/user-fs/sujin/dataset/mmseqs2/tools
def mmseqs2_eval(path, db_path):
	data = pd.read_csv(path, sep='\t', header=None, names=['qseqid', 'sseqid', 'pident', 'length', 'mismatch', 'gapopen',
	                                                       'qstart', 'qend', 'sstart', 'send', 'evalue', 'bitscore'])

	db = pd.read_csv(db_path, sep='\t')
	total_TP = 0
	total_P = 0
	total_N = 0
	indices = data.groupby('qseqid').indices
	
	for query_id, homo_ids in indices.items():
		sf_id = db[db['SCOP_ID'] == query_id]['SF_ID'].values[0]
		pred_homo_ids = set(data['sseqid'].iloc[homo_ids].values)
		true_homo_ids = set(db[db['SF_ID'] == sf_id]['SCOP_ID'].values)
		
		P = len(true_homo_ids)
		N = len(pred_homo_ids)
		TP = len(true_homo_ids & pred_homo_ids)
		
		recall = TP / P * 100
		precision = TP / N * 100
		
		total_TP += TP
		total_P += P
		total_N += N
		
		print(f"query_ID:{query_id}\tFOLD_ID:{sf_id:<10} recall:{f'{TP}/{P}({recall:.2f}%)':<20}"
		      f"precision:{f'{TP}/{N}({precision:.2f}%)':<20}")
	
	print(f'recall: {total_TP}/{total_P}({total_TP / total_P * 100:.2f}%)')
	print(f"precision: {total_TP}/{total_N}({total_TP / total_N * 100:.2f}%)")
	

def hhblits_eval(hhblits_path, db_path):
	data = pd.read_csv(hhblits_path, sep='\t')
	db = pd.read_csv(db_path, sep='\t')

	total_TP = 0
	total_P = 0
	total_N = 0
	indices = data.groupby('query').indices
	l = len(indices.keys())

	for query_id, homo_ids in indices.items():
		sf_id = db[db['SCOP_ID'] == query_id]['SF_ID'].values[0]
		pred_homo_ids = set(data['target'].iloc[homo_ids].values)
		true_homo_ids = set(db[db['SF_ID'] == sf_id]['SCOP_ID'].values)

		P = len(true_homo_ids)
		N = len(pred_homo_ids)
		TP = len(true_homo_ids & pred_homo_ids)

		recall = TP / P * 100
		precision = TP / N * 100

		total_TP += TP
		total_P += P
		total_N += N

		print(f"query_ID:{query_id}\tFOLD_ID:{sf_id:<10} recall:{f'{TP}/{P}({recall:.2f}%)':<20}"
		      f"precision:{f'{TP}/{N}({precision:.2f}%)':<20}")

	print(f'recall: {total_TP}/{total_P}({total_TP / total_P * 100:.2f}%)')
	print(f"precision: {total_TP}/{total_N}({total_TP / total_N * 100:.2f}%)")


if __name__ == '__main__':
	device_ids = [0, 1, 2, 3]
	# model_name = "esm1b_t33_650M_UR50S"
	model_name = "RetrieveModel_d1_m1_t53_I59_B256_phase6"
	# model_name = 'RetrieveModel_scope2.08_d1_m1_t12_I663_B256’
	# model_name = 'RetrieveModel_scope2.08_d1_m1_t62_I663_B256_phase3'
	model_path = f"PretrainedModels/{model_name}.pt"
	q_seq_path = 'Data/test_scope2.08.faa'
	save_path = f'Faiss/test_{model_name}'
	fasta2vec(device_ids, model_path, q_seq_path, save_path, cover=False)
	
	q_vec_path = f'{save_path}.npy'
	q_info_path = f'{save_path}_info.tsv'
	name = '800_900'
	db_index_path = f"Faiss/test_db_{model_name}.index"
	db_pointer_path = f"Faiss/test_db_{model_name}_pointer.tsv"
	db_path = "Data/scope2.08_<=512.faa"
	
	# 查询向量
	save_path = f'Results/UniRef30_2020_06_{name}_index.tsv'
	pipeline(q_vec_path, db_index_path, q_info_path, db_pointer_path, db_path, save_path, iter_num=1, t=1, nprobe=1, max_num=0)
	
	# 评估双塔模型
	res_path = save_path
	q_test_path = "test_scope2.08.tsv"
	db_test_path = "Data/db_scope2.08.tsv"
	twintower_eval(res_path, db_test_path)
	
	db_path = 'Data/test_db_scope2.08.tsv'
	# 评估hmmer3
	hmmer_path = 'Data/query_hmmer.tsv'
	# hmmer_eval(hmmer_path, db_path)
	
	# evaluate mmseqs2
	mmseqs2_path = "Data/resultDB.m8"
	# mmseqs2_eval(mmseqs2_path, db_path)
	
	# evaluate hhblits
	hhblits_path = "Data/query_hhblits.tsv"
	# hhblits_eval(hhblits_path, db_path)
	
	# 评估psi-blast
	blast_path = 'Data/query_blast.tsv'
	# blast_eval(blast_path, db_path)