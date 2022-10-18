import numpy as np
import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from LoadingData.dataset import MaskedLanguageModelingDataset
from torch.utils.data.dataloader import DataLoader
from utils import setup_seed, progress_bar
from model import MSARetrieveModel
from math import ceil
from task_eval.mutation.mutation_eval import calc_spearman
from contextlib import nullcontext


def train(device, world_size, model, dataset, name):
	# backend初始化
	dist.init_process_group("nccl", rank=device, world_size=world_size)
	# 训练参数设置
	EPOCH = 5
	BATCH_SIZE = 1
	GRADIENT_ACCUMULATION = 16
	VOCAB_SIZE = dataset.tokenizer.vocab_size
	
	optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
	loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
	
	model.to(device)
	# 多机多卡调用
	model = DDP(model, device_ids=[device], output_device=device)
	# 加载DataLoader
	sampler = torch.utils.data.distributed.DistributedSampler(dataset)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=dataset.collate_fn, sampler=sampler)

	# 开始训练
	cnt_list, spearman_list = [], []
	total_cnt = 0
	for epoch in range(EPOCH):
		sampler.set_epoch(epoch)
		cnt = 0
		ave_loss = 0
		n_iter = ceil(len(dataloader) / GRADIENT_ACCUMULATION)
		for i, data in enumerate(dataloader):
			context = model.no_sync if (i + 1) % GRADIENT_ACCUMULATION != 0 else nullcontext
			with context():
				input_ids = data['input_ids'].to(device)
				labels = data['labels'].to(device)
				
				logits = model.module.get_lm_head(input_ids)
				# pred = torch.argmax(logits, dim=-1)
				loss = loss_func(logits.view(-1, VOCAB_SIZE), labels.view(-1)) / GRADIENT_ACCUMULATION
				loss.backward()
				ave_loss += loss.item()
				
				if (i + 1) % GRADIENT_ACCUMULATION == 0 or i + 1 == len(dataloader):
					optimizer.step()
					optimizer.zero_grad()
					cnt += 1
					total_cnt += 1
					# indices = labels != -1
					# pred = pred[indices]
					# labels = labels[indices]
					# acc = ((pred == labels).sum() / labels.size(0)).item() * 100
					
					if dist.get_rank() == 0:
						desc = f"epoch:{epoch}  loss:{loss}"
						progress_bar(cnt, n_iter , desc)
						ave_loss = 0
						
						if cnt % 50 == 0 or cnt == n_iter:
							torch.save(model.module.state_dict(),
							           f"PretrainedModels/{name}_GB1_hhblits_{len(dataset)}_I{total_cnt}.pt")
	
	if dist.get_rank() == 0:
	# 	eval(device, model)
	# 	torch.save(model.module.state_dict(), f"PretrainedModels/esm1b_{name}_twintowers_{len(dataset)}.pt")
		eval_data = np.array([cnt_list, spearman_list])
		np.save(f"task_eval/mutation/datasets/{name}/{name}_twintowers_{len(dataset)}.npy", eval_data)


# def eval(device, model, da)


if __name__ == '__main__':
	setup_seed(2022)
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '10086'
	os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
	
	name = "BLAT_ECOLX_Ranganathan2015"
	# fasta_path = f"task_eval/mutation/datasets/{name}/{name}_twintowers.faa"
	fasta_path = "/sujin/dataset/FLIP/hhblits/GB1_hhblits.faa"
	dataset = MaskedLanguageModelingDataset(fasta_path)
	
	model_path = "PretrainedModels/esm1b_t33_650M_UR50S.pt"
	# model_path = "PretrainedModels/AMIE_10.pt"
	model = MSARetrieveModel()
	# model.esm1b.emb_layer_norm_before = None
	# model.esm1b.contact_head = None
	model.load_state_dict(torch.load(model_path, map_location='cpu'))
	model.eval()
	
	world_size = 4
	mp.spawn(train, args=(world_size, model, dataset, name), nprocs=world_size, join=True)
	
