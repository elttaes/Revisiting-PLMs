import re
import pandas as pd
import os
import numpy as np
import platform
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
from utils import TimeCounter

abs_path = os.path.dirname(os.path.dirname(__file__))
# use different commands for windows and linux respectively
system = platform.system().lower()
if system == 'linux':
	rm = 'rm {}'
	mv = 'mv {} {}'
	cp = 'cp {} {}'
	align_tool = f"{abs_path}/software/linux_famsa"
	filter_tool = f"{abs_path}/software/cd-hit"
	
	command = align_tool + " {} {} >/dev/null 2>&1"
	filter_cmd = filter_tool + " -i {} -o {} -c {} -n 5 -d 0 -T 8 >/dev/null 2>&1"
	
elif system == 'windows':
	rm = 'del {}'
	mv = 'move {} >nul 2>nul'
	align_tool = f"{abs_path}\\software\\windows_famsa.exe"
	command = align_tool + " {} {} >nul 2>nul"


def align(input, output, fmt):
	query = next(SeqIO.parse(input, 'fasta'))
	query_id = query.id
	build_msa(input, output, query_id, fmt=fmt)


def fasta2a3m(id, path):
	records = list(SeqIO.parse(path, "fasta"))
	gap_id = ord('-')
	# letter_gap = ord('A') - ord('a')
	
	for record in records:
		if record.id == str(id):
			query_record = record
			seq = bytes(query_record.seq)
			seq_ids = np.array([id for id in seq])
			info = query_record.description
			# 查询序列中有'-'的位置
			ori_gaps = seq_ids == gap_id
			seq = ''.join(chr(id) for id in seq_ids[~ori_gaps])
	
			
	with open(f'{path}.a3m', 'w') as f:
		f.write(f'>{info}\n{seq}\n')
		for record in records:
			if record.id == query_record.id:
				continue
				
			seq = bytes(record.seq)
			info = record.description
			seq_ids = np.array([id for id in seq])
			
			# 同源序列中有'-'的位置
			# seq_gaps = seq_ids == gap_id
			#
			# seq_ids[ori_gaps] = seq_ids[ori_gaps] - letter_gap
			#
			# mask = seq_gaps & ori_gaps
			seq = ''.join(chr(id) for id in seq_ids[~ori_gaps])
			
			f.write(f'>{info}\n{seq}\n')
	
	os.system(mv.format(f"{path}.a3m", path))


def build_msa(fasta_path, out_path, id, fmt='a3m'):
	os.system(command.format(fasta_path, out_path))
	if fmt == 'a3m':
		fasta2a3m(id, out_path)


def tsv_to_msa(tsv_path, output, filter_threshold, align=True, fmt='a3m', id=None):
	data = pd.read_csv(tsv_path, sep='\t')
	
	with TimeCounter('Generating files...'):
		for i, item in tqdm(enumerate(data.groupby("query_id").indices.items())):
			query_id, indices = item
			if id and query_id != id:
					continue
			info = data.iloc[indices].values
			
			if id:
				out_name = f"{output}/{id}.faa" if output else f"{id}.faa"
			else:
				out_name = f"{output}/query_{i}.faa" if output else f"query_{i}.faa"
			
			# with open(out_name, 'w') as f:
			# 	query_description, query_seq = info[0][1], info[0][4]
			# 	f.write(f">{query_description} \n{query_seq}\n")
			#
			# 	for target in info:
			# 		f.write(f">{target[3]} \n{target[5]}\n")
			
			with open(out_name+'temp', 'w') as f:
				query_description, query_seq = info[0][1], info[0][4]
				f.write(f">{query_description} \n{query_seq}\n")

				for target in info:
					f.write(f">{target[3]} \n{target[5]}\n")
		
			# filter msa
			os.system(filter_cmd.format(out_name+'temp', out_name, filter_threshold))
			os.system(rm.format(out_name + 'temp'))
			os.system(rm.format(out_name + '.clstr'))
			
			# add query sequence into fasta file in case it is filtered out
			with open(out_name, 'r+') as f:
				content = f.readlines()
				is_single = True if len(content) == 2 else False
				if query_description not in content[0]:
					is_single = False
					f.seek(0, 0)
					f.write(f">{query_description}\n{query_seq}\n")
					for line in content:
						f.write(line)
				
			if align:
				if is_single:
					os.system(cp.format(out_name, f"{out_name}_aligned"))
				else:
					build_msa(out_name, f"{out_name}_aligned", query_id, fmt=fmt)
					
				if fmt == 'a3m':
					os.system(mv.format(f"{out_name}_aligned", f"{out_name[:-4]}.a3m"))
					os.system(rm.format(out_name))
				elif fmt == 'fasta':
					os.system(mv.format(f"{out_name}_aligned", out_name))


def parse_tsv(tsv_path, id=None):
	data = pd.read_csv(tsv_path, sep='\t')
	for i, item in enumerate(data.groupby("query_id").indices.items()):
		if id:
			if item[0] != id:
				continue
		
		query_id, indices = item
		info = data.iloc[indices].values
		query_description, query_seq = info[0][1], info[0][4]
		# if len(query_seq) > 200:
		# 	continue
		
		print(f"no.{i+1}\nquery id: {query_id}")
		print(f"description: {query_description}")
		print(f"sequence length: {len(query_seq)}\n")
		
		print("detected sequences:")
		print(f"number:{len(info)}")
		print(f"{'target id':<40}{'length':<10}{'distance':<10}{'description':^50}")
		for target in info:
			hit_len = len(target[-2])
			hit_id = target[2]
			hit_description = target[3]
			distance = target[-1]
			print(f"{hit_id:<40}{hit_len:<10}{distance:.4f}\t{hit_description:<50}")
			
		print("*" * 100 + '\n\n')
		

def write_to_tsv(q_info_path, db_pointer_path, db_path, out_path, homo_list, dist_list, max_num):
	# 输出查询结果
	with open(out_path, 'w') as f:
		f.write(f"query_id\tquery_description\ttarget_id\ttarget_description\tquery_seq\ttarget_seq\tdistance\n")
		q_info = pd.read_csv(q_info_path, sep='\t')
		pointer = pd.read_csv(db_pointer_path, sep='\t')
		homo_dict = {}
		with open(db_path, 'r') as r:
			for i, record in enumerate(q_info.values):
				query_id, query_description, query_seq = record
				
				if len(homo_list[i]) == 0:
					continue
					
				for j, index in enumerate(homo_list[i][:max_num] if max_num != 0 else homo_list[i]):
					if index in homo_dict.keys():
						target_id, target_description, target_seq = homo_dict[index]
					else:
						loc = pointer.iloc[index][0]
						r.seek(loc)
						target_description = r.readline()[1:-1]
						target_id = target_description.split(' ', maxsplit=1)[0]
						target_seq = r.readline()[:-1]
						homo_dict[index] = (target_id, target_description, target_seq)
					
					if j != 0:
						query_seq = ""
					f.write(f"{query_id}\t{query_description}\t{target_id}\t"
					        f"{target_description}\t{query_seq}\t{target_seq}\t{dist_list[i][j]}\n")


if __name__ == '__main__':
	path = 'UniRef30_2020_06_900_1000_index.tsv'
	parse_tsv(path)