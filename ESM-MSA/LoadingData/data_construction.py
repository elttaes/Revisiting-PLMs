import os.path
import re
import random
import pickle
import pandas as pd
import torch
import numpy as np
from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from math import ceil
import multiprocessing as mp
from utils import TimeCounter, progress_bar, truncate
from model import MSARetrieveModel, AutoEncoder
from LoadingData.tokenizer import Tokenizer


# 过滤长度过长的序列
def data_filter(path, out_path, threshold=512):
    with open(out_path, 'w') as f:
        for seq_record in tqdm(SeqIO.parse(path, "fasta")):
            if len(seq_record.seq) > threshold:
                continue
            else:
                f.write(f">{seq_record.description}\n")
                f.write(f"{str(seq_record.seq)}\n")


# SCOP2
def load_data(path):
    sf = {}
    pattern = r'SF=([\d]+)'
    
    # 提取文件里的所有序列，按照超家族分类
    for seq_record in SeqIO.parse(path, "fasta"):
        sf_id = re.search(pattern, seq_record.description).group(1)
        
        if sf_id not in sf.keys():
            sf[sf_id] = [seq_record]
        else:
            sf[sf_id].append(seq_record)
    
    # 去掉所有同源序列数量为1的超家族
    del_ids = []
    for sf_id, seqs in sf.items():
        if len(seqs) == 1:
            del_ids.append(sf_id)
    [sf.pop(key) for key in del_ids]
    
    return sf


# SCOPe v2.08
def load_scope_data(path):
    pattern = r' (.*?\d+.\d+)'
    records = SeqIO.parse(path, 'fasta')
    sf_dict = {}
    
    for seq in records:
        fd_id = re.search(pattern, str(seq.description)).group(1)
        seq.seq = Seq(str(seq.seq).upper())
        if fd_id not in sf_dict.keys():
            sf_dict[fd_id] = [seq]
        else:
            sf_dict[fd_id].append(seq)
    
    sf_dict = {k: v for k, v in sf_dict.items() if len(v) > 1}
    
    return sf_dict


# uniclust30_2018_08_seed
def sample_seqs(fasta_path, pointer_path, n, threshold, path=None):
    pointer = pd.read_csv(pointer_path, sep='\t')
    # total number of sequences
    total = pointer.shape[0]
    
    # sample n negative
    indices = random.sample(list(range(total)), n)
    indices.sort()
    
    if path:
        w = open(path, 'w')
        
    with open(fasta_path, 'r') as f:
        for index in indices:
            loc = pointer.iloc[index][0]
            # loc = pointer.iloc[index][0]
            
            # ensure length of sequence less than threshold
            while True:
                f.seek(loc)
                f.readline()
                seq = f.readline()[:-1]
                # if 'J' in seq:
                #     seq = seq.replace('J', '')
                
                if len(seq) <= threshold:
                    if path:
                        w.write(f"{seq}\n")
                    break
                
                else:
                    new_index = random.randint(0, total - 1)
                    loc = pointer.iloc[new_index][0]


# 对fasta文件构建指针索引
def construct_pointer(fasta_path, save_path):
    with open(save_path, 'w') as w:
        w.write("location\n")
        with open(fasta_path, 'r') as r:
            loc = r.tell()
            line = r.readline()
            while line:
                if line[0] == '>':
                    w.write(f'{loc}\n')
                
                loc = r.tell()
                line = r.readline()


# 将数据划分为训练集和测试集
def data_split(data):
    sorted_sf_dict = sorted(data.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)
    
    test_seqs = []
    test_db = []
    # 系统抽样
    for i in range(10, len(data), 10):
        key = sorted_sf_dict[i][0]
        seqs = data.pop(key)
        test_seqs.append(seqs[0])
        test_db += seqs
    
    return data, test_seqs, test_db


def training_construction(data):
    n_per_sf = 2000
    
    fasta_path = "/sujin/dataset/uniclust30/uniclust30_2018_08_seed.fasta"
    pointer_path = "/sujin/dataset/uniclust30/uniclust30_2018_08_seed.tsv"
    pointer = pd.read_csv(pointer_path, sep='\t').values
    
    with open("../Data/train_scope2.08.tsv", 'w') as f:
        f.write("ori\tpos\tneg\n")
        
        for records in tqdm(data.values()):
            neg_seqs = sample_seqs(fasta_path, pointer, n_per_sf, 512)
            for i in range(n_per_sf):
                ori, pos = random.sample(records, 2)
                ori_seq = str(ori.seq)
                pos_seq = str(pos.seq)
                neg_seq = neg_seqs[i]
                
                f.write(f'{ori_seq}\t{pos_seq}\t{neg_seq}\n')


def threshold_data_construction(sf):
    with open("../Data/thresold_data_scope2.07.tsv", 'w') as f:
        f.write("seq\thomo_num\n")
        for k, v in sf.items():
            for seq in v:
                f.write(f"{str(seq.seq)}\t{len(v) - 1}\n")


def create_tsv(data, path):
    pattern = r' (.*?\d+.\d+)'
    seqs = []
    # 将向量的顺序与id对应起来输出为索引文件
    with open(path, 'w') as f:
        f.write("SF_ID\tSCOP_ID\n")
        for i, record in enumerate(data):
            scop_id = record.id
            sf_id = re.search(pattern, record.description).group(1)
            seqs.append(str(record.seq))
            
            f.write(f"{sf_id}\t{scop_id}\n")


def trRosetta(files, st, id):
    msa_dict = {}
    for i, file in enumerate(files):
        msa_dict[st + i] = list(SeqIO.parse(file, 'fasta'))
        
        progress_bar(i + 1, len(files), id)
    
    with open(f"dict_{id}.pkl", 'wb') as f:
        pickle.dump(msa_dict, f)


def mp_trRosetta(path, n, n_per_msa=500):
    msa_dict = {}
    q = mp.Queue()
    
    file_list = [f"{path}/{file}" for file in os.listdir(path)]
    
    # 将数据分割为多份
    num = ceil(len(file_list) / n)
    p_list = []
    for i in range(n):
        st = num * i
        ed = st + num
        
        p = mp.Process(target=trRosetta, args=(file_list[st: ed], st, i))
        p.start()
        p_list.append(p)
    
    # 等待所有进程执行完毕
    for p in p_list:
        p.join()
    
    for i in tqdm(range(n)):
        temp_path = f"dict_{i}.pkl"
        with open(temp_path, "rb") as tf:
            temp_dict = pickle.load(tf)
            msa_dict.update(temp_dict)
            os.remove(temp_dict)
    
    with open("../Data/trRosetta_dict.pkl", 'wb') as f:
        pickle.dump(msa_dict, f)


def sample_trRosetta(out, n_per_msa, pos_per_n, out_ori, out_pos):
    fasta_path = "/sujin/dataset/trRosetta/training_set/a3m/trRosetta_train.faa"
    pointer_path = "/sujin/dataset/trRosetta/training_set/a3m/trRosetta_train_pointer.tsv"
    index_path = "/sujin/dataset/trRosetta/training_set/a3m/trRosetta_train_index.tsv"
    
    neg_fasta_path = "/sujin/dataset/uniclust30/UniRef30_2020_06.faa"
    neg_pointer_path = "/sujin/dataset/uniclust30/UniRef30_2020_06_pointer.tsv"
    
    data = open(fasta_path, 'r')
    pointer = pd.read_csv(pointer_path, sep='\t')
    index = pd.read_csv(index_path, sep='\t')
    
    neg_data = open(neg_fasta_path, 'r')
    neg_pointer = pd.read_csv(neg_pointer_path, sep='\t')
    
    def get_seq(index, data, pointer, need_id=False):
        loc = pointer.iloc[index][0]
        data.seek(loc)
        if need_id:
            id = data.readline()[1:-1].split('_')[0]
        else:
            data.readline()
        
        seq = data.readline()[:-1]
        
        if need_id:
            return id, seq
        else:
            return seq
    
    total = neg_pointer.shape[0]
    # with open(out, 'w') as f:
    #     f.write("ori\tcandi\tlabel\n")
    
    ori = open(out_ori, 'w')
    pos = open(out_pos, 'w')
    for i, row in index.iterrows():
        st, num = row
        ed = st + num
        if num <= 1:
            continue
        
        num_list = list(range(st, ed))
        
        for _ in range(n_per_msa):
            ori_loc = st
            ori_seq = get_seq(ori_loc, data, pointer)
            ori.write(f"{ori_seq}\n")
            # ori_loc, pos_loc = random.sample(num_list, 2)
            # ori_seq = get_seq(ori_loc, data, pointer)
            # pos_seq = get_seq(pos_loc, data, pointer)
            
            # ori_seqs.append(ori_seq)
            # pos_seqs.append(pos_seq)
            
            # while True:
            # 	neg_loc = random.randint(0, total - 1)
            # 	neg_seq = get_seq(neg_loc, neg_data, neg_pointer)
            # if abs(len(neg_seq) - len(ori_seq)) < 50:
            # 	neg_seqs.append(neg_seq)
            # 	break
            
            for _ in range(pos_per_n):
                # pos_loc = num_list[-1]
                pos_loc = random.choice(num_list[1:])
                pos_seq = get_seq(pos_loc, data, pointer)
                pos.write(f"{pos_seq}\n")
                # f.write(f"{ori_seq}\t{pos_seq}\t1\n")
            
            # for _ in range(neg_per_n):
            #     while True:
            #         neg_loc = random.randint(0, total - 1)
            #         if neg_loc < st or neg_loc >= ed:
            #             neg_seq = get_seq(neg_loc, neg_data, neg_pointer)
            #             if abs(len(neg_seq) - len(ori_seq)) <= 50:
            #                 break
                # neg_seqs.append(neg_seq)
                # f.write(f"{ori_seq}\t{neg_seq}\t0\n")
        # f.write(f"{ori_seq}\t{pos_seq}\t{neg_seq}\n")
        
        progress_bar(i, index.shape[0], "Generating samples...", end='')
        

def mp_trRosetta_threshold():
    path = '/sujin/dataset/trRosetta/training_set/a3m/train'
    out = '/sujin/dataset/trRosetta/training_set/a3m/train_threshold.tsv'
    n = 8

    files = [f"{path}/{file}" for file in os.listdir(path)]
    
    faa = []
    for file in files:
        if 'faa' in file:
            faa.append(file)
    
    # 将数据分割为多份
    num = ceil(len(faa) / n)
    p_list = []
    for i in range(n):
        st = num * i
        ed = st + num
        path = f"{out}_{st}_{ed}.tsv"
    
        p = mp.Process(target=trRosetta_threshold, args=(i, faa[st: ed], path))
        p.start()
        p_list.append(p)

    # 等待所有进程执行完毕
    for p in p_list:
        p.join()

    with open(out, 'w') as f:
        f.write("seq\tthreshold\n")
    
        for i in range(n):
            st = num * i
            ed = st + num
            path = f"{out}_{st}_{ed}.tsv"
            with open(path, 'r') as rf:
                for line in rf.readlines()[1:]:
                    f.write(line)
        
            os.remove(path)


def trRosetta_threshold(device, files, path):
    model = MSARetrieveModel()
    model_path = '/sujin/TwinTowers/PretrainedModels/esm1b_t33_650M_UR50S_AE.pt'
    model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    model.to(device)
    model.eval()

    # encoder = AutoEncoder()
    # encoder.load_state_dict(torch.load('/sujin/TwinTowers/PretrainedModels/AutoEncoder_t30.pt', map_location='cpu'), strict=False)
    # encoder.to(device)
    # encoder.eval()
    
    BATCH = 2
    tokenizer = Tokenizer()
    with open(path, 'w') as f:
        f.write("seq\tthreshold\n")
        for now, file in enumerate(files):
            records = SeqIO.parse(file, 'fasta')
            query = str(next(records).seq).replace('J', '')
            if len(query) > 1022:
                continue
            
            seqs = []
            for record in records:
                if len(record.seq) <= 1022:
                    seqs.append(str(record.seq).replace('J', ''))
    
            with torch.no_grad():
                info_dict = tokenizer.batch_encode([query], padding=True)
                query_ids = info_dict['input_ids'].to(device)
                lengths = info_dict['lengths']

                ori_map = model.get_contact_map(query_ids, lengths).unsqueeze(1)
                ori = model.encoder.encode(ori_map)
                dist = 0
                seqs = seqs[1:]
                if len(seqs) > 1024:
                    step = len(seqs) / 1024
                    loc = np.arange(0, len(seqs), step)
                    loc = np.round(loc).astype(int)
                    seqs = [seqs[i] for i in loc]
        
                for i in range(0, len(seqs), BATCH):
                    samples = seqs[i: i + BATCH]
                    info_dict = tokenizer.batch_encode(samples, padding=True)
                    lengths = info_dict['lengths']
                    inputs = info_dict['input_ids'].to(device)
                    
                    maps = model.get_contact_map(inputs, lengths).unsqueeze(1)
                    vec = model.encoder.encode(maps)
                    # vec[i: i+BATCH] = outputs.to('cpu')
                    dist += torch.mm(ori, vec.T).sum()
                    # dist += (ori - vec).norm(2, dim=1).sum()
                    # dist = torch.mm(ori, outputs.T) / (ori.norm(2, dim=1) * outputs.norm(2, dim=1))
                
                threshold = (dist / len(seqs)).to('cpu').item()
                print(threshold)
                
            f.write(f"{query}\t{threshold}\n")
            f.flush()
            progress_bar(now+1, len(files), f"{device} Generating samples...")


def mp_auto_encoder(device_ids, model, seqs, save_path=None, batch_size=128):
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
        p = mp.Process(target=auto_encoder, args=(id, model, seq_list, path, batch_size))
        p.start()
        p_list.append(p)
    # 等待所有进程执行完毕
    for p in p_list:
        p.join()
    
    vectors = torch.empty(len(seqs), 64, 64)
    pointer = 0
    for i in range(len(device_ids)):
        if save_path:
            path = save_path + str(i)
        else:
            path = 'temp' + str(i)
        
        v = torch.load(path)
        os.remove(path)
        vectors[pointer: pointer + v.shape[0]] = v
        pointer += v.shape[0]
    
    if save_path:
        torch.save(vectors, save_path)
    
    return vectors


def auto_encoder(device, model, seqs, save_path, batch_size=128):
    tokenizer = Tokenizer()
    model.to(device)
    model.eval()

    vectors = torch.empty(len(seqs), 64, 64)
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            slice = seqs[i: i + batch_size]
            info = tokenizer.batch_encode(slice, padding=True)
            token_ids = info['input_ids']
            lengths = info['lengths']
            inputs = token_ids.to(device)
            
            repr = model.get_contact_map(inputs, lengths)
            vectors[i: i + batch_size] = repr
            
            desc = f"device:{device} calculating vectors"
            progress_bar(i + batch_size, len(seqs), desc)
    
    if save_path:
        torch.save(vectors, save_path)
    return vectors


def parse_a3m(id, files, out):
    with open(out, 'w') as f:
        for now, file in enumerate(files):
            data = pd.DataFrame(columns=['seq', 'identity'])
            
            pattern = r'[a-z]+'
            records = list(SeqIO.parse(file, 'fasta'))
            query = str(records[0].seq)
            if len(query) > 512:
                continue
            
            for i, record in enumerate(records):
                ori = str(record.seq).replace('-', '').upper()
                if len(ori) > 512:
                    continue
                    
                seq = re.sub(pattern, '', str(record.seq))
                cnt = 0
                total = 0
                for q, c in tuple(zip(query, seq)):
                    if c != '-':
                        total += 1
                        cnt += int(q == c)
                identitiy = cnt / total
                data = data.append({'seq': ori, 'identity': identitiy}, ignore_index=True)
        
            age_bins = np.arange(0, 1.1, 0.1).tolist()
            data['interval'] = pd.cut(x=data["identity"], bins=age_bins, right=False)
            
            for name, info in data.groupby('interval'):
                v = info.values
                if v.shape[0] <= 10:
                    for seq, identitiy, _ in v:
                        f.write(f"{query}\t{seq}\t{identitiy}\n")
                elif v.shape[0] > 10:
                    np.random.shuffle(v)
                    for seq, identitiy, _ in v[:10]:
                        f.write(f"{query}\t{seq}\t{identitiy}\n")

            progress_bar(now+1, len(files), id)


def mp_parse_a3m(n, files, save_path):
    # 将数据分割为多份
    num = ceil(len(files) / n)
    
    p_list = []
    for i in range(n):
        st = num * i
        ed = st + num
        temp_path = save_path + str(i)
        
        p = mp.Process(target=parse_a3m, args=(i, files[st: ed], temp_path))
        p.start()
        p_list.append(p)
        
    # 等待所有进程执行完毕
    for p in p_list:
        p.join()

    with open(save_path, 'w') as f:
        f.write("ori\tpos\tidentity\n")
    
        for i in range(n):
            temp_path = save_path + str(i)
            with open(temp_path, 'r') as rf:
                for line in rf.readlines():
                    f.write(line)
        
            os.remove(temp_path)


def sample_sim_trRosetta(n, out_ori, out_pos, out_sim):
    data = pd.read_csv('/sujin/TwinTowers/Data/trRosetta_sim.tsv', sep='\t')
    samples = data.sample(n)
    

    # ori_file = open(out_ori, 'w')
    # pos_file = open(out_pos, 'w')
    # sim_file = open(out_sim, 'w')
    #
    ori, pos, _, identity = tuple(zip(*samples.values.tolist()))
    print(identity)
    
    # for ori, pos, _, identity in samples.values:
    #     print(identity)


if __name__ == '__main__':
    # 文件的位置
    # path = "../Data/scope2.08_<=512.faa"
    # # out_path = "../Data/scope2.08_<=512.faa"
    # # data_filter(path, out_path, threshold=512)
    # # 读取文件
    # sf_dict = load_scope_data(path)
    # for k, v in sf_dict.items():
    # 	seqs = [len(str(r.seq)) for r in v]
    # 	print(k, len(v), min(seqs), max(seqs), sum(seqs)/len(v))
    # # 划分训练集测试集
    # train_sf, test_seqs, test_db = data_split(sf_dict)
    # # 将测试集输出为fasta文件
    # SeqIO.write(test_seqs, "../Data/test_scope2.08.faa", "fasta")
    # SeqIO.write(test_db, "../Data/test_db_scope2.08.faa", "fasta")
    # # 构建测试tsv文件
    # create_tsv(test_seqs, "../Data/test_scope2.08.tsv")
    # create_tsv(test_db, "../Data/test_db_scope2.08.tsv")
    # # 构建训练集
    # training_construction(train_sf)
    
    # fasta_path = f"../Data/scope2.08_<=512.faa"
    # pointer_path = f"../Data/scope2.08_<=512.faa_pointer.tsv"
    # if not os.path.exists(pointer_path):
    # 	construct_pointer(fasta_path, pointer_path)
    #
    # negs = sample_seqs(fasta_path, pointer_path, 100, 1022)
    # print(negs)
    
    # path = "/sujin/dataset/trRosetta/training_set/a3m/train"
    # mp_trRosetta(path, 100)
    # random.seed(2021)
    out = "/sujin/TwinTowers/Data/trRosetta_train_o1_p1_n100.tsv"
    # sample_trRosetta(out, 1, 1, 100)
    # print(random.randint(0, 1))
    # mp_trRosetta_threshold()
    
    # device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
    # model_path = f"/sujin/TwinTowers/PretrainedModels/esm1b_t33_650M_UR50S.pt"
    # fasta_path = "/sujin/TwinTowers/Data/trRosetta_ae_1_1_10.tsv"
    # save_path = '/sujin/TwinTowers/Data/AE_train.pt'
    #
    # # records = SeqIO.parse(fasta_path, 'fasta')
    # # seqs = [str(record.seq) for record in records]
    # seqs = pd.read_csv(fasta_path, sep='\t')['seq'].values
    #
    # model = MSARetrieveModel()
    # model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=False)
    # mp_auto_encoder(device_ids, model, seqs, save_path=save_path, batch_size=2)

    # path = '/sujin/dataset/MSA_Uniref50/Example_100G'
    # files = []
    # for dir in os.listdir(path):
    #     dir_path = path + '/' + dir
    #     for file in os.listdir(dir_path):
    #         if '.a3m' in file:
    #             a3m = dir_path + "/" + file
    #             files.append(a3m)
    
    out = '/sujin/TwinTowers/Data/trRosetta_sim.tsv'
    # mp_parse_a3m(100, files, out)
    
    data = pd.read_csv(out, sep='\t')
    print(data[['ori', 'pos', 'identity']])

    # fasta_path = "/sujin/dataset/uniclust30/UniRef30_2020_06.faa"
    # pointer_path = "/sujin/dataset/uniclust30/UniRef30_2020_06_pointer.tsv"
    # out_neg = '/sujin/TwinTowers/Data/trRosetta_sim_neg.tsv'
    # sample_seqs(fasta_path, pointer_path, data.shape[0], 512, out_neg)
    #
    # with open(out_neg, 'r') as r:
    #     with open(out+'1', 'w') as w:
    #         w.write("ori\tpos\tneg\tidentity\n")
    #
    #         for ori, pos, identity in data.values:
    #             neg = r.readline()[:-1]
    #             w.write(f"{ori}\t{pos}\t{neg}\t{identity}\n")

    # input = '/sujin/TwinTowers/Data/trRosetta_sim.tsv1'
    # out = '/sujin/TwinTowers/Data/trRosetta_sim.tsv'
    # with open(input, 'r') as r:
    #     with open(out, 'w') as w:
    #         for line in tqdm(r.readlines()):
    #             text = line.replace('J', '')
    #             w.write(text)
    
    # sample_sim_trRosetta(10, 1, 1, 1)