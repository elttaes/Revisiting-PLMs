import pandas as pd
import random
import numpy as np
import os
import time
from LoadingData.PkU_data import mp_setup
from math import ceil
from utils import TimeCounter, truncate
from LoadingData.data_construction import load_scope_data, sample_seqs, data_split
from vector_construction import mp_get_vec


def remove_temp_files(args, temp_path):
    remove_list = [f"{temp_path}_ori.seq",
                   f"{temp_path}_pos.seq",
                   f"{temp_path}_neg.seq",
                   f"{temp_path}_identity.seq",
                   f"{temp_path}_ori.seq.npy",
                   f"{temp_path}_pos.seq.npy",
                   f"{temp_path}_neg.seq.npy",
                   f"generation_done"]
    
    for i in range(args.total_rank):
        remove_list.append(f"{temp_path}_ori.seq_{i}.npy")
        remove_list.append(f"{temp_path}_pos.seq_{i}.npy")
        remove_list.append(f"{temp_path}_neg.seq_{i}.npy")
    
    for file in remove_list:
        if os.path.exists(file):
            os.remove(file)


def write_sequences(seqs, save_path):
    with open(save_path, 'w') as f:
        for seq in seqs:
            f.write(f"{seq}\n")


def load_sequences(path):
    with open(path, 'r') as f:
        return [seq[:-1] for seq in f.readlines()]


def generate_training_set(path, n_per_sf, neg_samples):
    out_ori = f"{path}_ori.seq"
    out_pos = f"{path}_pos.seq"
    out_neg = f"{path}_neg.seq"
    out_identity = f"{path}_identity.seq"
    
    # sampling negative samples
    fasta_path = "/sujin/dataset/uniclust30/UniRef30_2020_06.faa"
    pointer_path = "/sujin/dataset/uniclust30/UniRef30_2020_06_pointer.tsv"
    sample_seqs(fasta_path, pointer_path, neg_samples, 512, out_neg)

    # sampling original and positive samples
    trRosetta_path = 'Data/trRosetta_sim.tsv'
    pku_path = 'Data/pku_sim.tsv'

    trRosetta = pd.read_csv(trRosetta_path, sep='\t')[['ori', 'pos', 'identity']]
    # pku = pd.read_csv(pku_path, sep='\t')

    # data = pd.concat((trRosetta, pku))
    samples = trRosetta.sample(n_per_sf)
    ori, pos, identity = tuple(zip(*samples.values.tolist()))
    
    write_sequences(ori, out_ori)
    write_sequences(pos, out_pos)
    write_sequences(identity, out_identity)
    
    # sampling original and positive samples
    # msa_path = "/sujin/dataset/trRosetta/training_set/a3m"
    # training_set = '/sujin/TwinTowers/Data/trRosetta_train.tsv'
    # msa_path = "/sujin/dataset/MSA_Uniref50/Example_100G"
    # training_set = '/sujin/TwinTowers/Data/temp_train.tsv'
    #
    # n_per_msa = 1
    # if not os.path.exists(training_set):
    #     mp_setup(msa_path, 100, n_per_msa, out=training_set)
    
    # pku_data = pd.read_csv(training_set, sep='\t')
    # ori_seqs = pku_data['ori'].values
    # pos_seqs = pku_data['pos'].values

    # ori_seqs, pos_seqs = [], []
    # for records in data.values():
    #     for i in range(n_per_sf):
    #         ori, pos = random.sample(records, 2)
    #         ori_seqs.append(str(ori.seq))
    #         pos_seqs.append(str(pos.seq))
    
    # ori_seqs = [truncate(seq, 512) for seq in ori_seqs]
    # pos_seqs = [truncate(seq, 512) for seq in pos_seqs]
    # neg_samples = [truncate(seq, 512) for seq in neg_seqs]
    
    # write_sequences(neg_samples, f"{path}_neg.seq")
    # write_sequences(ori_seqs, f"{path}_ori.seq")
    # write_sequences(pos_seqs, f"{path}_pos.seq")

    with open(f"generation_done", 'w') as f:
        pass
    

def calculate_vectors(args, model, temp_path):
    device_ids = list(range(args.local_size))
    seqs = load_sequences(temp_path)
 
    len_per_part = ceil(len(seqs)/args.world_size)
    data_start = len_per_part * args.start_loc
    data_end = data_start + len_per_part * args.local_size
    sub_seqs = seqs[data_start:data_end]
    
    # calculating vectors of negative samples
    save_path = f"{temp_path}_{args.node_rank}.npy"
    if not os.path.exists(save_path):
        mp_get_vec(device_ids, model, sub_seqs, save_path=save_path, batch_size=64)
    
    # wait until all vectors calculated
    while True:
        flag = True
        for i in range(args.total_rank):
            path = f"{temp_path}_{i}.npy"
            if not os.path.exists(path):
                flag = False
                break
        
        if flag:
            break
    
    # master host merges several vectors and others wait
    if args.node_rank == 0:
        vectors = np.empty((len(seqs), model.dim), dtype=np.float32)
        pointer = 0
        for i in range(args.total_rank):
            path = f"{temp_path}_{i}.npy"
            v = np.load(path)
            vectors[pointer: pointer + v.shape[0]] = v
            pointer += v.shape[0]
    
        np.save(f"{temp_path}.npy", vectors)
    
    else:
        while True:
            if os.path.exists(f"{temp_path}.npy"):
                break


def build_training_file(args, temp_path, n_per_sf, pos_per_sample, num_candi, num_final):
    if args.node_rank == 0:
        ori_seqs = load_sequences(f"{temp_path}_ori.seq")
        pos_seqs = load_sequences(f"{temp_path}_pos.seq")
        neg_seqs = load_sequences(f"{temp_path}_neg.seq")
        identity = load_sequences(f"{temp_path}_identity.seq")
        
        ori_vecs = np.load(f"{temp_path}_ori.seq.npy")
        # pos_vecs = np.load(f"{temp_path}_pos.seq.npy")
        neg_vecs = np.load(f"{temp_path}_neg.seq.npy")
        
        # choose the negative sample with nearest distance to the original sample
        # and write results into a tsv format file
        with open(f"{temp_path}.tsv", 'w') as f:
            f.write("ori\tcandi\tlabel\tidentity\n")
        
            for i, ori_vec in enumerate(ori_vecs):
                # pos_st = int(i / n_per_sf) * pos_per_sample
                # # pos_st = i * pos_per_sample
                # sub_pos_vecs = pos_vecs[pos_st: pos_st+pos_per_sample]
                # pos_dist = np.linalg.norm(sub_pos_vecs - ori_vec, ord=2, axis=1)
                # max_loc = np.argmax(pos_dist)
                # pos_seq = pos_seqs[pos_st + max_loc]
                # if pos_dist[max_loc] != np.inf:
                #     f.write(f"{ori_seqs[i]}\t{pos_seq}\t1\n")
                f.write(f"{ori_seqs[i]}\t{pos_seqs[i]}\t1\t{identity[i]}\n")
                
                # neg_st = i * neg_per_sample
                # sub_neg_vecs = neg_vecs[neg_st: neg_st+neg_per_sample]
                sub_neg_list = [random.randint(0, len(neg_seqs)-1) for _ in range(num_candi)]
                sub_neg_vecs = neg_vecs[sub_neg_list]
                dist = np.linalg.norm(sub_neg_vecs - ori_vec, ord=2, axis=1)
                ranks = np.argsort(dist)[:num_final]
                
                for rank in ranks:
                    if dist[rank] != np.inf:
                        neg_seq = neg_seqs[sub_neg_list[rank]]
                        f.write(f"{ori_seqs[i]}\t{neg_seq}\t0\t0\n")
                # if dist[min_loc] != np.inf:
                #     f.write(f"{ori_seqs[i]}\t{pos_seqs[i]}\t{neg_seq}\n")
        
        # master host sends a signal that indicates the work is done
        with open(f"{temp_path}_0_validation", 'w') as f:
            pass
    
    else:
        while True:
            if os.path.exists(f"{temp_path}_0_validation"):
                break


def dynamic_sampling(args, data, model, temp_path, n_per_sf, pos_per_msa, neg_samples, num_candi, num_final):
    if not os.path.exists(f"{temp_path}.tsv"):
        # master computer generates training set
        if args.node_rank == 0:
            generate_training_set(temp_path, n_per_sf, neg_samples)
        
        # others wait until the training set built
        else:
            while True:
                if os.path.exists(f"generation_done"):
                    break
    
        # divide training set into pieces for each host to calculate vectors
        # ori seqs
        calculate_vectors(args, model, f"{temp_path}_ori.seq")
        # pos seqs
        # calculate_vectors(args, model, f"{temp_path}_pos.seq")
        # neg_seqs
        calculate_vectors(args, model, f"{temp_path}_neg.seq")
        
        # master host builds training file
        build_training_file(args, temp_path, n_per_sf, pos_per_msa, num_candi, num_final)
        
        if args.node_rank == 0:
            # check if every host breaks the loop
            while True:
                flag = True
                for i in range(1, args.total_rank):
                    path = f"{temp_path}_{i}_validation"
                    if not os.path.exists(path):
                        flag = False
                        break
    
                if flag:
                    break
                
            remove_temp_files(args, temp_path)
            for i in range(args.total_rank):
                os.remove(f"{temp_path}_{i}_validation")
                
        else:
            with open(f"{temp_path}_{args.node_rank}_validation", 'w') as f:
                pass
    