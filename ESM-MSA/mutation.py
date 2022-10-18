from model import MSARetrieveModel
import torch
import numpy as np
import random
from tqdm import tqdm
from utils import setup_seed
from LoadingData.tokenizer import Tokenizer


def load_model():
	model = MSARetrieveModel().to(device)
	path = 'PretrainedModels/esm1b_t33_650M_UR50S.pt'
	model.load_state_dict(torch.load(path, map_location='cpu'))
	
	classifer_path = "PretrainedModels/RetrieveModel_t17_I430_B256_sim_p1_n10.pt"
	classifer = MSARetrieveModel().to(device)
	classifer.load_state_dict(torch.load(classifer_path, map_location='cpu'))
	
	return model, classifer


def get_loc(model, seqs):
	# loc = torch.tensor([[random.randint(0, len(seq)-1)] for seq in seqs])
	# return loc
	info_dict = tokenizer.batch_encode(seqs, padding=True)
	input_ids = info_dict['input_ids'].to(device)
	lengths = info_dict['lengths']
	
	with torch.no_grad():
		contact_map = model.esm1b.predict_contacts(input_ids, lengths=lengths)[0]
		contact_map = torch.triu(contact_map, 1)
		n = contact_map.size(0)
		num = int((n - 1) * n / 2 * 0.05)
		rank = torch.argsort(contact_map.flatten(), descending=True)[:num]
		indices = [random.choice(rank) for _ in range(len(seqs))]
		
		loc = torch.empty(len(indices), 2, dtype=torch.long)
		
		for i, index in enumerate(indices):
			row = int(index / n)
			column = index % n
			loc[i] = torch.tensor([row, column])
			
		return loc


def mutate(model, seqs, loc):
	info_dict = tokenizer.batch_encode(seqs, padding=True)
	ori_ids = info_dict['input_ids']
	input_ids = ori_ids
	
	for i in range(len(seqs)):
		input_ids[i] = tokenizer.add_mask_token(input_ids[i], loc[i] + 1)
	input_ids = input_ids.to(device)
	
	with torch.no_grad():
		vec = model.get_lm_head(input_ids)[:, 1:-1]
		for i, l in enumerate(loc):
			# a = vec[i][l].argsort(dim=-1, descending=True
			# for a, b in zip(ori_tokens, mut_tokens):
			# 	print(f"{a} --> {b}")
			for j in l:
				ori_id = ori_ids[i][j]
				logits = vec[i][j]
				ori_prob = logits[ori_id]
				
				mask = logits >= ori_prob
				candi = logits[mask].softmax(dim=-1)
				indices = mask.nonzero()
				index = torch.multinomial(candi, 1)
				
				token = tokenizer.convert_id_to_token(indices[index])
				seqs[i] = seqs[i][:j] + token + seqs[i][j + 1:]

		return seqs


def visualize(a, b, loc):
	a_token = [t for t in a]
	b_token = [t for t in b]
	
	for l in loc:
		if a_token[l] != b_token[l]:
			a_token[l] = f"\033[1;31m{a_token[l]}\033[0m"
			b_token[l] = f"\033[1;31m{b_token[l]}\033[0m"
	
	a = ''.join(a_token)
	b = ''.join(b_token)
	
	print(f"{a}\n{b}\n")
	

def classify(model, ori, seqs, batch=64):
	seqs = [ori] + seqs
	vectors = torch.empty(len(seqs), model.dim).to(device)
	with torch.no_grad():
		for st in range(0, len(seqs), batch):
			slice = seqs[st: st + batch]
			info_dict = tokenizer.batch_encode(slice, padding=True)
			input_ids = info_dict['input_ids'].to(device)
			
			vec = model(input_ids)['vec']
			vectors[st: st + batch] = vec
		
		ori, mutations = vectors[0:1], vectors[1:]
		dist = torch.norm(ori-mutations, 2, dim=-1)
		print(dist)


# get seqs with all amino acids at one location
def get_aa_seqs(ori, seq, loc):
	seqs = [ori]
	# id of each amino acid except for sequence's original amino acid
	for i in range(4, 24):
		mutation = seq[:loc] + tokenizer.convert_id_to_token(i) + seq[loc+1:]
		seqs.append(mutation)
	
	input_ids = tokenizer.batch_encode(seqs, padding=True)['input_ids']
	return input_ids


def get_aa_dist(model, ori, seq, loc):
	input_ids = get_aa_seqs(ori, seq, loc).to(device)
	with torch.no_grad():
		vec = model(input_ids)['vec']
		
		ori, mutations = vec[0:1], vec[1:]
		dist = torch.norm(ori - mutations, 2, dim=-1).to('cpu')
	
	# print(dist)
	# print(dist.size())
	# print(input_ids.size())
	return dist, input_ids[1:]


def get_dist(model, ori, candi_ids, batch=64):
	vectors = torch.empty(candi_ids.size(0), model.dim).to(device)
	with torch.no_grad():
		ori_ids = tokenizer.batch_encode([ori], padding=True)['input_ids'].to(device)
		ori_vec = model(ori_ids)['vec']
		
		candi_ids = candi_ids.to(device)
		for st in range(0, candi_ids.size(0), batch):
			slice = candi_ids[st: st + batch]
			vec = model(slice)['vec']
			vectors[st: st + batch] = vec
		
		dist = torch.norm(ori_vec - vectors, 2, dim=-1).to('cpu')
	
	return dist
	

def get_mask_seqs(seq):
	mask_id = tokenizer.vocab[tokenizer.mask_token]
	input_id = torch.tensor(tokenizer.single_encode(seq))
	
	input_ids = []
	for i in range(len(seq)):
		mask_seq = input_id.clone()
		mask_seq[i+1] = mask_id
		input_ids.append(mask_seq)
	
	return torch.stack(input_ids)


def get_ins_seqs(seq):
	mask_id = tokenizer.vocab[tokenizer.mask_token]
	input_id = torch.tensor(tokenizer.single_encode(seq))
	input_ids = torch.empty(len(seq)+1, input_id.size(0)+1, dtype=torch.long)
	for i in range(0, len(seq)+1):
		input_ids[i, :i+1] = input_id[:i+1]
		input_ids[i, i+1] = mask_id
		input_ids[i, i+2:] = input_id[i+1:]
	
	return input_ids


def get_del_seqs(seq):
	seqs = [seq[0:i] + seq[i+1:] for i in range(len(seq))]
	input_ids = tokenizer.batch_encode(seqs, padding=True)['input_ids']
	return input_ids


def get_mask_dist(model, seq, batch=64):
	input_ids = get_mask_seqs(seq).to(device)
	vectors = torch.empty(input_ids.size(0), model.dim).to(device)
	with torch.no_grad():
		for st in range(0, input_ids.size(0), batch):
			slice = input_ids[st: st + batch]
			vec = model(slice)['vec']
			vectors[st: st + batch] = vec
		
		ori, mutations = vectors[0:1], vectors[1:]
		dist = torch.norm(ori - mutations, 2, dim=-1).to('cpu')
	
	return dist


def mutate_mask(ori, seq):
	mask_i = random.randint(0, len(seq) - 1)
	dist, input_ids = get_aa_dist(classifer, ori, seq, mask_i)
	index = torch.argmin(dist)
	mask_id = index + 4
	mask_token = tokenizer.convert_id_to_token(mask_id)
	next_seq = seq[:mask_i] + mask_token + seq[mask_i+1:]
	return next_seq
	
	mask_ids = get_mask_seqs(seq)
	# mask_dist = get_dist(model, ori, mask_ids)
	
	# mask_i = random.choice(torch.argsort(mask_dist)[:10])
	# mask_v, mask_i = torch.min(mask_dist, dim=0)
	loc = mask_i
	input_ids = mask_ids[mask_i: mask_i + 1].to(device)
		
	with torch.no_grad():
		vec = model.get_lm_head(input_ids)[0][1:-1]
		ori_id = tokenizer.convert_token_to_id(seq[loc])
		logits = vec[loc]
		ori_prob = logits[ori_id]
		
		mask = logits >= ori_prob
		candi = logits[mask].softmax(dim=-1)
		indices = mask.nonzero()
		index = torch.multinomial(candi, 1)
		
		# indices = torch.argsort(logits, descending=True)
		# candi = logits[indices].softmax(dim=-1)
		# index = torch.multinomial(candi, 1)
		
		token = tokenizer.convert_id_to_token(indices[index])
		return seq[:loc] + token + seq[loc + 1:]


def mutate_ins(ori, seq):
	ins_i = random.randint(0, len(seq))
	ins_seq = seq[:ins_i] + 'A' + seq[ins_i:]
	dist, input_ids = get_aa_dist(classifer, ori, seq, ins_i)
	index = torch.argmin(dist)
	ins_id = index + 4
	ins_token = tokenizer.convert_id_to_token(ins_id)
	next_seq = seq[:ins_i] + ins_token + seq[ins_i:]
	return next_seq
	
	ins_ids = get_ins_seqs(seq)
	# ins_dist = get_dist(model, ori, ins_ids)
	
	ins_i = random.randint(0, ins_ids.size(0)-1)
	# ins_i = random.choice(torch.argsort(ins_dist)[:10])
	# ins_v, ins_i = torch.min(ins_dist, dim=0)
	loc = ins_i
	input_ids = ins_ids[ins_i: ins_i + 1].to(device)

	with torch.no_grad():
		vec = model.get_lm_head(input_ids)[0][1:-1]
		logits = vec[loc]
		index = torch.argmax(logits)
		# indices = torch.argsort(logits, descending=True)[:3]
		# candi = logits[indices].softmax(dim=-1)
		# index = torch.multinomial(candi, 1)
		
		token = tokenizer.convert_id_to_token(index)
		return seq[:loc] + token + seq[loc:]


def mutate_del(ori, seq):
	del_ids = get_del_seqs(seq)
	del_dist = get_dist(classifer, ori, del_ids)
	del_v, del_i = torch.min(del_dist, dim=0)
	# del_i = random.randint(0, len(seq) - 1)
	# print('del', del_v)
	
	return seq[:del_i] + seq[del_i + 1:]
	

if __name__ == '__main__':
	device = 'cuda'
	setup_seed(2022)
	model, classifer = load_model()
	tokenizer = Tokenizer()
	
	ori_seq = "DFNIVAVASNFKLPLEVLKEMEANARKAGCTRGCLICLSHIKCTPKMKKFIPGRCHTYEIVDIPAIPRFKDLEPMEQFIAQVDLCVDCTTGCLKGLANV"
	ori_ids = torch.tensor([ord(c) for c in ori_seq]).unsqueeze(0)
	mask_ids = get_mask_seqs(ori_seq).to(device)
	with torch.no_grad():
		vecs = model.get_lm_head(mask_ids)[0]
	print(vecs.size())
	print(mask_ids.size())
	print(len(ori_seq))
	print(mask_ids)
	for token in ori_seq:
		id = tokenizer.convert_token_to_id(token)
		p = vecs
	# ids = get_ins_seqs(ori_seq)
	# print(ids)
	# dist = get_dist(classifer, ori_seq, ids)
	# print(dist)
	# print(ids)
	# print(ids.size())
	
	batch = 256
	ori = [ori_seq] * batch
	seqs = [ori_seq] * batch
	
	f = open("test.a3m", 'w')
	f.write(f">0\n{ori_seq}\n")
	
	t_list = np.arange(0.4, 0.5, 1)
	cnt = 1
	loc_list = []
	print(ori_seq)
	
	# initialize mutated sequence set
	candi_dict = [ori_seq]
	aa_beam_width = 2
	mutation_list = []
	res_num = 16
	
	mask_dist = get_mask_dist(classifer, seqs[0])
	rank = iter(torch.argsort(mask_dist))
	for t in t_list:
		# while len(mutation_list) < res_num:
		# 	loc = next(rank)
		# 	next_list = []
		# 	for seq in candi_dict:
		# 		dist, input_ids = get_aa_dist(classifer, ori_seq, seq, loc)
		# 		for i in torch.argsort(dist)[:aa_beam_width]:
		# 			next_seq_list = tokenizer.convert_ids_to_tokens(input_ids[i])
		# 			next_seq = ''.join(next_seq_list[1:-1])
		#
		# 			next_list.append([next_seq, dist[i]])
		# 			print(dist[i])
		#
		# 	next_list = sorted(next_list, key=lambda x: x[1])[:res_num]
		# 	seq_ids = torch.tensor([[ord(c) for c in seq[0]] for seq in next_list])
		# 	sim = (seq_ids - ori_ids == 0).sum(dim=-1) / len(ori_seq)
		# 	res = sim < t
		# 	suc = torch.nonzero(res)
		#
		# 	fail = torch.nonzero(~res)
		# 	# t = t[~res]
		#
		# 	for i in suc:
		# 		mutation_list.append(next_list[i][0])
		#
		# 	# for prev, l, seq in zip(prevs, loc, seqs):
		# 	# 	visualize(prev, seq, l)
		#
		# 	candi_dict = [next_list[i][0] for i in fail]
		#
		# for seq in mutation_list:
		# 	print(seq)
		for time in tqdm(range(100)):
			# prevs = [seq for seq in seqs]
			# loc = get_loc(model, seqs)
			# print(loc)
			# dist = get_mask_dist(classifer, seqs[0])
			# dist_rank = torch.argsort(dist)
			# for i in dist_rank:
			# 	if i not in loc_list:
			# 		loc_list.append(i)
			# 		loc = i.view(1, 1)
			# 		break
			# loc = random.choice(dist_rank[:10]).view(1, 1)
			# print(loc)
			# loc = next(rank).view(1, 1)
			# loc_list += loc.squeeze(0).tolist()
			for i in range(len(seqs)):
					mutate_func = random.choice([mutate_del, mutate_ins, mutate_mask])
					seqs[i] = mutate_func(ori_seq, seqs[i])
				# loc = torch.tensor(random.randint(0, len(ori_seq)-1)).view(1, 1)
				# dist, input_ids = get_aa_dist(classifer, ori_seq, seqs[i], loc)
			# prob = (1 - dist).softmax(dim=-1)
			# print(dist)
			# print(prob)
			# dist_rank = torch.multinomial(prob, 1)
			# 	dist_rank = random.choice(torch.argsort(dist)[1])
			# 	dist_rank = torch.argmin(dist)
			# 	seq_list = tokenizer.convert_ids_to_tokens(input_ids[dist_rank])
			# 	seqs[i] = ''.join(seq_list[1:-1])
	
				# seqs = mutate(model, seqs, loc)
				# classify(classifer, ori[0], seqs[i: i+1])

			# ori_ids = torch.tensor([ord(c) for c in ori[0]]).unsqueeze(0)
			# seq_ids = torch.tensor([[ord(c) for c in seq] for seq in seqs])
			# sim = (seq_ids - ori_ids == 0).sum(dim=-1) / len(ori[0])
			# print(sim)
			# res = sim < t
			# suc = torch.nonzero(res)
			# fail = torch.nonzero(~res)
			# # t = t[~res]
			#
			# for i in suc:
			# 	mutation_list.append(seqs[i])
			#
			# # for prev, l, seq in zip(prevs, loc, seqs):
			# # 	visualize(prev, seq, l)
			#
			# seqs = [seqs[i] for i in fail]

		# loc_list = list(set(loc_list))
		# for seq in mutation_list:
		# 	visualize(ori[0], seq, loc_list)
		
		print(ori_seq)
		for seq in seqs:
			print(seq)
			
		for seq in seqs:
			f.write(f">{cnt}\n{seq}\n")
			cnt += 1
		
		classify(classifer, ori_seq, seqs)
	
