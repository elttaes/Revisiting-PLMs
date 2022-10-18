# -*-coding:utf-8-*-
import argparse
import numpy
import torch
import esm
import numpy as np
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import random
# from gpu_mem_track import MemTracker  
import inspect
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import math
from apex import amp
# from prefetch_generator import BackgroundGenerator
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import cuda
from tqdm import tqdm
import glob
from random import *
from torch.cuda.amp import autocast as autocast
# Distributed
from torch.utils.data.distributed import DistributedSampler

class Configue(object):
    label_list = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask>']
    max_mask = 100 # max number of padding

class AllDataset(Dataset):
    def __init__(self):
        self.get_filename = self.get_txt_file()
        self.data = self.process_filename()
        self.len = len(self.data)

    def get_txt_file(self):  # get filename
        dir_file = []
        s = "./"
        os.chdir(s)
        files = glob.glob('*.txt')
        for iter, filename in enumerate(files):
            dir_file.append(filename)
        return dir_file

    def process_filename(self):
        filename = self.get_filename
        for i in tqdm(range(len(filename)), desc='load all data...(maybe a little slow)'):
            data = np.loadtxt("./data" + filename[i], dtype=list).tolist()  # type from array to list
            if i == 0:
                all = data
            else:
                all += data
        return all

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    # def process_data_mask(self):
    # mask
    @staticmethod
    def collate__fn(batch):

        all_mask_tokens = []
        max_len_data = max(len(batch[i]) for i in range(len(batch)))
        batch_size = len(batch)
        tokens = torch.empty(
            (
                batch_size,
                max_len_data + 2  # <cls> <eos>
            ),
            dtype=torch.int64,
        )
        tokens.fill_(Configue.label_list.index('<pad>'))

        for i in range(batch_size):
            tokens[i,0] = Configue.label_list.index("<cls>")

            label = []
            
            masked_tokens = np.zeros([max_len_data + 2], np.int64) - 1

            for item in batch[i]:
                label.append(Configue.label_list.index(item))  
            
            mask_number = min(Configue.max_mask,max(1,int(len(batch[i])*0.15)))
            cand_mask_pos = [i for i, token in enumerate(batch[i])]  # [0,1,2,3,4..]
            shuffle(cand_mask_pos)

            for pos in cand_mask_pos[:mask_number]:
                masked_tokens[pos] = label[pos]     
                if random() < 0.8:                  # 80% mask
                    label[pos] = Configue.label_list.index('<mask>')
                elif random() > 0.9:                # 10% others
                    index = randint(4, 28)
                    label[pos] = index

            seq = torch.tensor(label)
            tokens[i,1:len(batch[i]) + 1] = seq
            tokens[i, len(batch[i]) + 1] = Configue.label_list.index("<eos>")

            all_mask_tokens.append(masked_tokens)

        all_label_ids = torch.LongTensor(all_mask_tokens)

        return tokens, all_label_ids