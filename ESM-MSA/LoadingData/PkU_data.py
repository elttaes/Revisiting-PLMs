import os
import random
import multiprocessing as mp
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
from utils import setup_seed, progress_bar
from math import ceil
from utils import truncate


def building(file_list, path, n_per_msa, st, ed):
	print("building training set...")
	pool = list(range(len(file_list)))
	with open(path, 'w') as f:
		f.write("ori\tpos\tneg\n")

		for i, file in enumerate(file_list[st: ed]):

			records = list(SeqIO.parse(file, 'fasta'))

			# 对正样本进行采样
			ori_seqs = []
			pos_seqs = []
			for _ in range(n_per_msa):
				ori, pos = random.sample(records, 2)
				ori_seqs.append(truncate(str(ori.seq).replace('-', '').upper(), 512))
				pos_seqs.append(truncate(str(pos.seq).replace('-', '').upper(), 512))

			# 对负样本进行采样
			neg_nums = []
			for _ in range(n_per_msa):
				while True:
					neg_num = random.choice(pool)
					if file != file_list[neg_num]:
						neg_nums.append(neg_num)
						break

			neg_seqs = []
			for num in neg_nums:
				neg_records = list(SeqIO.parse(file_list[num], 'fasta'))
				sample = random.choice(neg_records)
				neg_seqs.append(truncate(str(sample.seq).replace('-', '').upper(), 512))

			for j in range(len(ori_seqs)):
				f.write(f"{ori_seqs[j]}\t{pos_seqs[j]}\t{neg_seqs[j]}\n")

			progress_bar(i+1, ed-st, path)


def mp_setup(path, n, n_per_msa, out):
	print("loading data...")
	print(path)
	# dirs = os.listdir(path)
	# file_list = []
	# for dir in tqdm(dirs):
	# 	sub_dir = path + "/" + dir
	# 	for file in os.listdir(sub_dir):
	# 		file_path = sub_dir + '/' + file
	# 		file_list.append(file_path)
	file_list = [f"{path}/{file}" for file in os.listdir(path)]
	print("finished")

	# 将数据分割为多份
	num = ceil(len(file_list) / n)
	p_list = []
	for i in range(n):
		st = num * i
		ed = st + num
		path = f"{out}_{st}_{ed}.tsv"

		p = mp.Process(target=building, args=(file_list, path, n_per_msa, st, ed))
		p.start()
		p_list.append(p)

	# 等待所有进程执行完毕
	for p in p_list:
		p.join()

	# 合并数据
	count = 0
	with open(f"{out}", 'w') as f:
		f.write("ori\tpos\tneg\n")

		for i in range(n):
			st = num * i
			ed = st + num
			path = f"{out}_{st}_{ed}.tsv"
			with open(path, 'r') as rf:
				for line in rf.readlines()[1:]:
					f.write(line)

			os.remove(path)


if __name__ == '__main__':
	setup_seed(2021)
	path = "/sujin/dataset/trRosetta/training_set/a3m/train"
	n_per_msa = 10
	mp_setup(path, 100, n_per_msa, out='../Data/trRosetta_train.tsv')

	# data = pd.read_csv(path, sep='\t')
	# tiny_data = data.sample(4000)
	# print(tiny_data)
	# tiny_data.to_csv("../Data/PkU_tiny_training_set.tsv", index=False, sep='\t')