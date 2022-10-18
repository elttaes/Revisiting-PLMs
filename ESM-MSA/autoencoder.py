import torch
import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model import AutoEncoder
from torch.utils.data import DataLoader
from LoadingData.dataset import AutoEncoderDataset
from torch.optim import Adam
from utils import setup_seed, progress_bar


def train(device, world_size, model, dataset):
	# backend初始化
	dist.init_process_group("nccl", rank=device, world_size=world_size)
	# 训练参数设置
	EPOCH = 100
	BATCH_SIZE = 128
	mse = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
	model.to(device)
	# 多机多卡调用
	model = DDP(model, device_ids=[device], output_device=device)
	# 加载DataLoader
	sampler = torch.utils.data.distributed.DistributedSampler(dataset)
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
	
	import matplotlib.pyplot as plt
	# 开始训练
	for epoch in range(EPOCH):
		sampler.set_epoch(epoch)
		model.train()
		for i, data in enumerate(dataloader):
			# inputs = data + torch.randn(data.size())
			inputs = data.to(device).view(-1, 1, 64, 64)
			label = torch.zeros(data.size(0), dtype=torch.long).to(device)
			
			pred = model(inputs)
			# repr = repr.view(-1, 12, 1024)
			# ori, pos_neg = repr[:, 0:1], repr[:, 1:]
			# sim = torch.bmm(ori, pos_neg.permute(0, 2, 1)).squeeze(1).softmax(dim=-1)
			mse_loss = mse(pred, inputs)
			# ce_loss = ce(sim, label)
			loss = mse_loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			#
			# import matplotlib.pyplot as plt
			# plt.imshow(inputs[5, 0].to('cpu').numpy(), 'gray')
			# plt.show()
			# img = torch.where(pred[5, 0] > 0.5, 1, 0)
			# plt.imshow(pred[5, 0].detach().to('cpu').numpy(), 'gray')
			# plt.show()
			# return
			if dist.get_rank() == 0:
				desc = f"epoch:{epoch}  loss:{loss}"
				progress_bar(i + 1, len(dataloader), desc, end='')
		print()
			
		if dist.get_rank() == 0:
			eval(device, model)
			torch.save(model.module.state_dict(), f"PretrainedModels/AutoEncoder_t{epoch+1}.pt")


def eval(device, model):
	path = '/sujin/TwinTowers/Data/AE_train.pt'
	dataset = AutoEncoderDataset(path, 'test')
	
	BATCH_SIZE = 64
	mse = torch.nn.MSELoss()
	ce = torch.nn.CrossEntropyLoss()
	model.to(device)
	model.eval()
	
	# 加载DataLoader
	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
	
	import matplotlib.pyplot as plt
	# 开始训练
	with torch.no_grad():
		total_mse_loss = 0
		total_ce_loss = 0
		cnt = 0
		for i, data in enumerate(dataloader):
			inputs = data.to(device).view(-1, 1, 64, 64)
			label = torch.zeros(data.size(0), dtype=torch.long).to(device)
			
			pred = model(inputs)
			mse_loss = mse(pred, inputs)

			plt.imshow(inputs.to('cpu').numpy()[0, 0], 'gray')
			plt.show()
			plt.imshow(pred.to('cpu').numpy()[0, 0], 'gray')
			plt.show()
			
			#
			total_mse_loss += mse_loss
			# if dist.get_rank() == 0:
			progress_bar(i + 1, len(dataloader), "testing...", end='')
			
		print(f"\nmse loss: {total_mse_loss/len(dataloader)} ")


if __name__ == '__main__':
	# 设置多机多卡环境参数
	os.environ['MASTER_ADDR'] = 'localhost'
	os.environ['MASTER_PORT'] = '10086'
	os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
	setup_seed(2021)
	world_size = 1
	
	
	print('initializing the model...  ', end='')
	model_path = "PretrainedModels/AutoEncoder_t8.pt"
	model = AutoEncoder()
	# model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
	model.train()
	print('finished')
	
	# from model import MSARetrieveModel, ContactMapModel, AutoEncoder
	#
	# device = 'cuda'
	# model = MSARetrieveModel()
	# path = 'PretrainedModels/esm1b_t33_650M_UR50S_AE.pt'
	# # model_path = "PretrainedModels/esm1b_t33_650M_UR50S.pt"
	# model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
	# # model.encoder.load_state_dict(torch.load("PretrainedModels/AutoEncoder.pt", map_location='cpu'))
	# model.to(device)
	# model.eval()
	# model = model.encoder
	
	print('loading the dataset...  ', end='')
	path = '/sujin/TwinTowers/Data/AE_train.pt'
	dataset = AutoEncoderDataset(path, 'train')
	print('finished')

	mp.spawn(train, args=(world_size, model, dataset), nprocs=world_size, join=True)
	# eval('cuda', model)
	