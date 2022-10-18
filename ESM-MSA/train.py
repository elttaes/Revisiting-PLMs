import time

import pandas as pd
import tokenizers
import torch.cuda
import random
import argparse
import numpy as np
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from contextlib import nullcontext
from math import ceil
from torch.nn.parallel import DistributedDataParallel as DDP
from LoadingData.data_construction import load_scope_data, sample_seqs, data_split
from torch.utils.checkpoint import checkpoint
from model import MSARetrieveModel
from LoadingData.dataset import RetrieveDataset
from torch.utils.data import DataLoader
from utils import setup_seed, progress_bar, TimeCounter
from LoadingData.tokenizer import Tokenizer
from vector_construction import mp_get_vec
from torch.cuda import amp
from dynamic_sampling import dynamic_sampling

parser = argparse.ArgumentParser()
parser.add_argument("--world_size", default=1, type=int)
parser.add_argument("--local_size", default=1, type=int)
parser.add_argument("--node_rank", default=0, type=int)
parser.add_argument("--start_loc", default=0, type=int)
parser.add_argument("--total_rank", default=1, type=int)
parser.add_argument("--master_addr", default="127.0.0.1", type=str)
parser.add_argument("--master_port", default="10086", type=str)
args = parser.parse_args()


# 额外添加的loss，用于确定分类阈值并据此来训练模型
class RetrieveLoss:
	def __init__(self, pos_t, neg_t, margin):
		self.pos_t = pos_t
		self.neg_t = neg_t
		self.margin = margin
		
		e = torch.exp(torch.tensor(1))
		self.pos_bias = torch.log(e + 1) - 1
		
		self.ce_loss = torch.nn.CrossEntropyLoss()
		self.mse_loss = torch.nn.MSELoss()
		
	# 考虑欧式距离的损失函数
	def __call__(self, ori, candi, label, identity, pos_weight, neg_weight):
		zero = torch.tensor(0, dtype=torch.double).to(ori)
		dist = (ori - candi).norm(2, dim=1, keepdim=True)
		# pos_dist = (ori - pos).norm(2, dim=1)
		# neg_dist = (ori - neg).norm(2, dim=1)
		
		# neg_dist = self.neg_t - neg_dist
		# neg_loss = torch.where(neg_dist < 0, zero, neg_dist).mean()
		#
		# pos_loss = self.mse_loss(pos_dist, 1-identity)
		# return neg_loss + pos_loss
		
		# Cross entropy loss
		# logits = torch.cat((1/(pos_dist+1e-6), 1/(neg_dist+1e-6)), dim=1)
		# label = torch.zeros(pos_dist.size(0), dtype=torch.long).to(logits.device)
		# ce_loss = self.ce_loss(logits, label)
		
		# 计算triplet loss
		# triplet_dist = pos_dist - neg_dist + self.margin
		
		# triplet_loss = torch.where(triplet_dist < 0, zero, triplet_dist).mean()
		
		# 计算distance loss
		# 正负样本分别计算
		neg = label == 0
		pos = ~neg
		neg_samples = dist[neg]
		pos_samples = dist[pos]
		
		# 正样本采用MSE Loss，负样本采用Distance Loss
		t = torch.full(neg_samples.size(), 2).to(neg_samples)
		logits = torch.cat((neg_samples, t), dim=-1)
		neg_loss = self.ce_loss(logits.squeeze(-1), label[neg]) if neg_samples.size(0) > 0 else 0
		# neg_dist = t - neg_samples
		# neg_loss = torch.where(neg_dist < 0, zero, neg_dist).mean() if neg_samples.size(0) > 0 else 0
		
		identity = identity[pos]
		pos_loss = self.mse_loss(pos_samples.squeeze(-1), 1-identity) if pos_samples.size(0) > 0 else 0
		
		return pos_loss * pos_weight + neg_loss * neg_weight
	
	# dist_loss = dist - t
	# neg = label == 0
	# pos = ~neg
	# dist_loss[neg] = dist_loss[neg] * -1
	# dist_loss[pos] = dist_loss[pos] * pos_weight
	# dist_loss = torch.where(dist_loss < 0, zero, dist_loss)
	# dist_loss = dist_loss.mean()
	# return dist_loss


# pos_dist = pos_dist - self.pos_t
# neg_dist = self.neg_t - neg_dist
# neg_loss = torch.where(neg_dist < 0, zero, neg_dist).mean()

# pos_loss = (torch.log(1 + torch.exp(pos_dist)) - self.pos_bias).mean()
# neg_loss = (torch.log(1 + torch.exp(neg_dist))).mean()
# pos_loss = torch.where(pos_dist < 0, zero, pos_dist).mean()
# neg_loss = torch.where(neg_dist < 0, zero, neg_dist).mean()

# pos_loss = pos_loss / (pos_dist > 0).sum().item() if pos_loss > 0 else pos_loss
# neg_loss = neg_loss / (neg_dist > 0).sum().item() if neg_loss > 0 else neg_loss
# pos_loss /= pos_dist.size(0)
# neg_loss /= neg_dist.size(0)

# dist_loss = neg.size(0) * pos_loss + pos.size(0) * neg_loss
#
# loss = dist_loss
# return loss


# 考虑余弦相似度的损失函数
# def __call__(self, ori, pos, neg, t=0.5):
#     zero = torch.tensor(0, dtype=torch.double).to(ori)
#     value_t = t
#     pos_dist = (ori * pos).sum(dim=1)
#     neg_dist = (ori * neg).sum(dim=1)
#
#     # 计算triplet loss
#     triplet_dist = neg_dist - pos_dist + value_t
#
#     triplet_loss = torch.where(triplet_dist < 0, zero, triplet_dist).sum()
#     triplet_loss = triplet_loss / (triplet_dist > 0).sum().item() if triplet_loss > 0 else triplet_loss
#
#     # 计算distance loss
#     pos_dist = value_t - pos_dist
#     neg_dist = neg_dist - value_t
#
#     pos_loss = torch.where(pos_dist < 0, zero, pos_dist).sum()
#     neg_loss = torch.where(neg_dist < 0, zero, neg_dist).sum()
#
#     pos_loss = pos_loss / (pos_dist > 0).sum().item() if pos_loss > 0 else pos_loss
#     neg_loss = neg_loss / (neg_dist > 0).sum().item() if neg_loss > 0 else neg_loss
#     dist_loss = pos_loss + neg_loss
#
#     loss = triplet_loss + dist_loss
#     return loss


def train(local_rank, world_size, local_size, node_rank, model, dataset, times):
	rank = local_rank + local_size * node_rank
	# backend初始化
	dist.init_process_group("nccl",
	                        init_method="tcp://{}:{}".format(args.master_addr, args.master_port),
	                        rank=rank,
	                        world_size=world_size)
	
	# 训练参数设置
	EPOCH = 1
	BATCH_SIZE = 2
	GRADIENT_ACCUMULATION = 16
	loss_func = RetrieveLoss(1, 1, 1)
	
	model.train()
	model.to(local_rank)
	model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
	
	# 使用torch混合精度训练
	scaler = amp.GradScaler()
	
	# 多机多卡调用
	model = DDP(model, device_ids=[local_rank], output_device=local_rank)
	optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
	
	# 加载DataLoader
	sampler = torch.utils.data.distributed.DistributedSampler(dataset)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, sampler=sampler)
	# 开始训练
	for epoch in range(EPOCH):
		cnt = 0
		ave_loss = 0
		n_iter = ceil(len(dataloader) / GRADIENT_ACCUMULATION)
		sampler.set_epoch(epoch)
		for i, data in enumerate(dataloader):
			# 在梯度累加时取消all reduce操作，加快训练速度
			context = model.no_sync if (i + 1) % GRADIENT_ACCUMULATION != 0 else nullcontext
			with context():
				with amp.autocast():
					input_ids = data['input_ids'].to(local_rank)
					label = data['label'].to(local_rank)
					identity = data['identity'].to(local_rank)
					# ori_ids = data['ori_ids'].to(local_rank)
					# pos_ids = data['pos_ids'].to(local_rank)
					# neg_ids = data['neg_ids'].to(local_rank)
					# identity = data['identity'].to(local_rank)
					
					outputs = model(input_ids)['vec']
					size = int(outputs.size(0) / 2)
					# ori = model(ori_ids)['vec']
					# pos = model(pos_ids)['vec']
					# neg = model(neg_ids)['vec']
					
					loss = loss_func(outputs[:size], outputs[size:], label, identity, 1, 1) / GRADIENT_ACCUMULATION
				
				scaler.scale(loss).backward()
				ave_loss += loss.item()
				
				if (i + 1) % GRADIENT_ACCUMULATION == 0 or i + 1 == len(dataloader):
					scaler.step(optimizer)
					scaler.update()
					optimizer.zero_grad()
					cnt += 1
					if dist.get_rank() == 0:
						desc = f"epoch:{epoch}  loss:{ave_loss}"
						progress_bar(cnt, n_iter, desc)
						ave_loss = 0
						
						if cnt % 100 == 0 or cnt == n_iter:
							batch = world_size * BATCH_SIZE * GRADIENT_ACCUMULATION
							torch.save(model.module.state_dict(),
							           f"PretrainedModels/RetrieveModel_t{times}_I{cnt}_B{batch}_Pmse_Nce_lastlayer.pt")


if __name__ == '__main__':
	seed = 2022
	setup_seed(seed)
	# 读取数据集
	# with TimeCounter("Loading the dataset..."):
		# trRosetta_path = 'Data/trRosetta_sim.tsv'
		# pku_path = 'Data/pku_sim.tsv'
		#
		# trRosetta = pd.read_csv(trRosetta_path, sep='\t')[['ori', 'pos', 'identity']]
		# pku = pd.read_csv(pku_path, sep='\t')
		#
		# data = pd.concat((trRosetta, pku))
		# dataset = RetrieveDataset(path)
	
	# 初始化模型
	with TimeCounter("Initializing the model..."):
		model = MSARetrieveModel()
		path = 'PretrainedModels/esm1b_t33_650M_UR50S.pt'
		model.load_state_dict(torch.load(path, map_location='cpu'))
	
	# with TimeCounter("Loading sample pool..."):
	# 	# 读取文件
	# 	path = 'Data/scope2.08.faa'
	# 	sf_dict = load_scope_data(path)
	# 	# 加载训练样本池
	# 	train_sf, _, _ = data_split(sf_dict)
	
	# mp.spawn(train,
	#          args=(args.world_size, args.local_size, args.node_rank, model, dataset, 1),
	#          nprocs=args.local_size,
	#          join=True)
	
	times = 1
	while True:
		if args.node_rank == 0:
			if os.path.exists("validation.temp"):
				os.remove("validation.temp")
	
		temp_path = "temp_dataset"
		n_per_sf = 10000
		pos_per_msa = 1
		total = 10000
	
		# neg_per_msa = 100
		neg_samples = 600000
		num_candi = 500
		num_final = 10
	
		# dynamically sampling training set
		with TimeCounter("Sampling training set..."):
			dynamic_sampling(args, None, model, temp_path, total, pos_per_msa, neg_samples, num_candi, num_final)
	
		with TimeCounter("Loading the dataset..."):
			dataset = RetrieveDataset(f"{temp_path}.tsv")
	
		# start training
		mp.spawn(train,
		         args=(args.world_size, args.local_size, args.node_rank, model, dataset, times),
		         nprocs=args.local_size,
		         join=True)
	
		if args.node_rank == 0:
			os.remove(f"{temp_path}.tsv")
			with open("validation.temp", 'w') as f:
				pass
	
		else:
			while True:
				if os.path.exists("validation.temp"):
					break
	
		times += 1
