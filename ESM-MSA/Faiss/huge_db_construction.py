# 构建超大型数据库
import os
import numpy as np
import multiprocessing as mp
from Bio import SeqIO
from tqdm import tqdm


def split_fasta(id, path, folder_path):
	if not os.path.exists(folder_path):
		os.mkdir(folder_path)

	for record in tqdm(SeqIO.parse(path, 'fasta'), desc=f"{id}th process is splitting fasta file"):
		path = f"{folder_path}/{record.id}"
		if not os.path.exists(path):
			with open(path, 'w') as f:
			# SeqIO.write([record], path, "fasta")
				f.write(record.description + '\n')
				f.write(str(record.seq) + '\n')


# 多进程切分数据
def mp_split_fasta(n, path, folder_path):
	p_list = []
	for i in range(n):
		p = mp.Process(target=split_fasta, args=(i, path, folder_path))
		p.start()
		p_list.append(p)

	# 等待所有进程执行完毕
	for p in p_list:
		p.join()


if __name__ == '__main__':
	path = "/sujin/dataset/uniclust30/uniclust30_2018_08_15.faa"
	folder_path = "/sujin/dataset/uniclust30/fasta_folders"
	mp_split_fasta(100, path, folder_path)