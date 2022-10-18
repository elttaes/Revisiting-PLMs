import torch
import os
import numpy as np
import cv2
import logging
import multiprocessing as mp
from Bio import SeqIO
from math import ceil
from LoadingData.tokenizer import Tokenizer
from model import MSARetrieveModel
from utils import progress_bar


# 利用多进程的方式获取数据库序列高维向量
def mp_get_vec(device_ids, model, seqs, save_path=None, batch_size=128, verbose=True):
    torch.multiprocessing.set_start_method('spawn', force=True)
    # 将数据分割为多份
    n = ceil(len(seqs) / len(device_ids))
    p_list = []
    for i, id in enumerate(device_ids):
        if save_path:
            path = save_path + str(i)
        else:
            path = 'temp' + str(i)
        seq_list = seqs[n * i: n * i + n]
        p = mp.Process(target=get_vec, args=(id, model, seq_list, path, batch_size, verbose))
        p.start()
        p_list.append(p)

    # 等待所有进程执行完毕
    for p in p_list:
        p.join()

    # 整合生成的向量数据
    vectors = np.empty((len(seqs), model.dim), dtype=np.float32)
        
    pointer = 0
    for i in range(len(device_ids)):
        if save_path:
            path = save_path + str(i) + ".npy"
        else:
            path = 'temp' + str(i) + ".npy"

        v = np.load(path)
        os.remove(path)
        vectors[pointer: pointer + v.shape[0]] = v
        pointer += v.shape[0]

    if save_path:
        np.save(save_path, vectors)

    return vectors


# 获取数据库序列的高维向量
def get_vec(device, model, seqs, save_path, batch_size=128, verbose=True):
    tokenizer = Tokenizer()
    model.to(device)
    model.eval()
    
    vectors = np.empty((len(seqs), model.dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            slice = seqs[i: i + batch_size]
            info = tokenizer.batch_encode(slice, padding=True)
            token_ids = info['input_ids']
            lengths = info['lengths']
            inputs = token_ids.to(device)
            
            # repr = model.get_map_repr(inputs, lengths)
            # vectors[i: i + batch_size] = repr.to('cpu').numpy()
            res = model(inputs)
            repr = res['vec']
            
            # if get_t:
            #     t = res['t']
            #     vec = np.concatenate((repr.to('cpu'), t.to('cpu').unsqueeze(-1)), axis=1)
            #     vectors[i: i + batch_size] = vec
            vectors[i: i + batch_size] = repr.to('cpu')
            desc = f"device:{device} calculating vectors"
            if verbose:
                progress_bar(i + batch_size, len(seqs), desc)

    if save_path:
        np.save(save_path, vectors)
    
    return vectors


# 构建向量库索引
def construct_index(records, path, type):
    seqs = []
    if type == 'query':
        f = open(path, 'w')
        f.write("id\tdecription\tsequence\n")
    
    for record in records:
        if len(record.seq) > 1022:
            print(f"sequence {record.id} has length > 1022, so ignored.")
            continue
        seqs.append(str(record.seq))
        
        if type == 'query':
            f.write(f"{record.id}\t{record.description}\t{str(record.seq)}\n")

    return seqs


# 将fasta序列转化为numpy的向量
def fasta2vec(device_ids, model_path, fasta_path, save_path, batch_size=128, type='query', cover=False, verbose=True):
    if os.path.exists(f"{save_path}.npy") and not cover:
        if verbose:
            print("npy file already exists.")
        return
    
    assert type in ['query', 'database']
    records = SeqIO.parse(fasta_path, 'fasta')
    seqs = construct_index(records, f"{save_path}_info.tsv", type)
    
    model = MSARetrieveModel()
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    

    return mp_get_vec(device_ids, model, seqs, save_path=save_path, batch_size=batch_size, verbose=verbose)
