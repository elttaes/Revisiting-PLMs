import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model import MSARetrieveModel
from LoadingData.dataset import ThresholdDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from math import ceil
from contextlib import nullcontext
from utils import setup_seed, progress_bar


def train(device, world_size, model, dataset):
	# backend初始化
	dist.init_process_group("nccl", rank=device, world_size=world_size)
	# 训练参数设置
	EPOCH = 100
	BATCH_SIZE = 2
	GRADIENT_ACCUMULATION = 8
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.threshold_model.parameters(), lr=1e-3)
	model.to(device)
	# 多机多卡调用
	model = DDP(model, device_ids=[device], output_device=device)
	# 加载DataLoader
	sampler = torch.utils.data.distributed.DistributedSampler(dataset)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, sampler=sampler)
	
	# 开始训练
	for epoch in range(EPOCH):
		sampler.set_epoch(epoch)
		ave_loss = 0
		cnt = 0
		n_iter = ceil(len(dataloader) / GRADIENT_ACCUMULATION)
		model.train()
		for i, data in enumerate(dataloader):
			context = model.no_sync if (i + 1) % GRADIENT_ACCUMULATION != 0 else nullcontext
			with context():
				inputs = data['seq_ids'].to(device)
				lengths = data['lengths']
				thresholds = data['thresholds']
	
				t = model.module.get_t(inputs, lengths)
				label = thresholds.to(device)
	
				loss = criterion(t, label) / GRADIENT_ACCUMULATION
				loss.backward()
				ave_loss += loss.item()
				
				if (i + 1) % GRADIENT_ACCUMULATION == 0 or i + 1 == len(dataloader):
					optimizer.step()
					optimizer.zero_grad()
					cnt += 1
					if dist.get_rank() == 0:
						desc = f"epoch:{epoch}  loss:{ave_loss}"
						progress_bar(cnt, n_iter, desc)
					ave_loss = 0
		
		if dist.get_rank() == 0:
			eval(device, model)
			torch.save(model.module.state_dict(), f"PretrainedModels/RetrieveModel_t{epoch+1}.pt")


def eval(device, model):
	path = '/sujin/TwinTowers/Data/threshold_test.tsv'
	dataset = ThresholdDataset(path)
	BATCH_SIZE = 2
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn)
	model.eval()
	model.to(device)
	criterion = torch.nn.MSELoss()
	
	loss = 0
	for i, data in enumerate(dataloader):
		inputs = data['seq_ids'].to(device)
		lengths = data['lengths']
		thresholds = data['thresholds']
		
		t = model.module.get_t(inputs, lengths)
		label = thresholds.to(device)
		
		loss += criterion(t, label)

		progress_bar(i + 1, len(dataloader), "evaluating...", end='')
	
	print(f"loss: {loss/len(dataloader)}")


if __name__ == '__main__':
	# 设置多机多卡环境参数
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '10086'
	os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
	setup_seed(2021)
	world_size = 4

	print('initializing the model...  ', end='')
	model_path = "PretrainedModels/RetrieveModel_t1.pt"
	model = MSARetrieveModel()
	model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
	model.train()
	# 冻结model的esm1b和autoencoder参数
	for param in model.esm1b.parameters():
		param.requires_grad = False
	for param in model.encoder.parameters():
		param.requires_grad = False
	print('finished')

	print('loading the dataset...  ', end='')
	path = '/sujin/TwinTowers/Data/threshold_train.tsv'
	dataset = ThresholdDataset(path)
	print('finished')

	eval('cuda', model)
	# mp.spawn(train, args=(world_size, model, dataset), nprocs=world_size, join=True)

	# import pandas as pd
	# import numpy as np
	# path = '/sujin/dataset/trRosetta/training_set/a3m/train_threshold.tsv'
	# data = pd.read_csv(path, sep='\t')
	# data['LEN'] = np.nan
	# for row_loc in range(len(data)):
	# 	data.iloc[row_loc, -1] = len(data.iloc[row_loc, 0])
	#
	# data.sort_values(by='LEN', inplace=True)
	# data.reset_index(drop=True, inplace=True)
	# n = 1000
	# total = data.shape[0]
	#
	# step = total / n
	# loc = np.arange(0, total, step)
	# loc = np.round(loc).astype(int)
	# seqs = data.iloc[loc]
	#
	# data.drop(index=loc, inplace=True)
	# print(data)
	# print(seqs)
	# data.to_csv('/sujin/TwinTowers/Data/threshold_train.tsv', sep='\t', index=False, columns=['seq', 'threshold'])
	# seqs.to_csv('/sujin/TwinTowers/Data/threshold_test.tsv', sep='\t', index=False,
	#             columns=['seq', 'threshold'])