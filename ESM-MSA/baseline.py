# 评估已有的软件
import os
import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm


def hhblits(query_path, db_path, save_path):
	hhblits = "/home/public/sj/software/hh-suite/build/bin/hhblits"
	file_columns = ['query', 'target', '#match/tLen', '#mismatch', '#gapOpen', 'qstart1', 'qstart', 'qend', 'tstart', 'tend', 'eval', 'score']

	with open(save_path, 'w') as f:
		f.write('query\ttarget\tE-value\n')

		for record in SeqIO.parse(query_path, 'fasta'):
			SeqIO.write([record], 'query.fasta', 'fasta')
			n_line = os.popen(f"{hhblits} -i query.fasta -n 1 -d {db_path} -blasttab out | wc -l").read()
			n_hits = int(n_line) - 10

			data = pd.read_csv("out", sep='\t', error_bad_lines=False, header=None, names=file_columns)
			hits = data.iloc[:n_hits][['query', 'target', 'eval']]

			for _, row in hits.iterrows():
				outputs = "{}\t{}\t{}\n".format(*row.values)
				f.write(outputs)


if __name__ == '__main__':
	# fasta_path = 'Data/test_scope2.08.faa'
	# db_path = 'Data/hhblits_scope/scope2.08'
	# save_path = 'Data/query_hhblits.tsv'
	# hhblits(fasta_path, db_path, save_path)
	path = "/sujin/TwinTowers/task_eval/af2_twintowers/pdb"
	pdb2fasta = f"{path}/pdb2fasta.py"
	for file in tqdm(os.listdir(path)):
		if "pdb" in file:
			file_path = f"{path}/{file}"
			out_path = f"{path}/{file.replace('pdb', 'faa')}"
			os.system(f"python {pdb2fasta} {file_path} > {out_path}")