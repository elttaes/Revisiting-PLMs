import os
import json
import numpy as np
from Bio import SeqIO


def calc_neff(path):
    with open(path, 'r') as f:
        data = f.readlines()
        ids = np.array([[ord(sign) for sign in data[i][:-1]] for i in range(1, len(data), 2)])
        masks = ids == ord('-')
        length = ids.shape[1]

    # ignore gap of any column between any two sequences
    temp = masks
    masks = np.expand_dims(masks, 1).repeat(masks.shape[0], axis=1)
    masks = masks | temp

    # calculate sequence identity between any two sequences
    temp = ids
    ids = np.expand_dims(ids, 1).repeat(ids.shape[0], axis=1)
    ids = ids - temp

    # invert the effective length of the sequence to avoid divided by 0
    valid_len = (~masks).sum(axis=-1)
    invert_len = np.where(valid_len != 0, 1 / valid_len, 0)

    identity_matrix = ((ids == 0) * ~masks).sum(axis=-1) * invert_len
    denominator = (identity_matrix >= 0.8).sum(axis=1)
    denominator[denominator == 0] = 1
    neff = (1 / denominator).sum() / np.sqrt(length)
    return neff


if __name__ == '__main__':
    path = 'out'
    cnt = 1
    hh_ave = 0
    tt_ave = 0
    hh_tm_ave = 0
    tt_tm_ave = 0
    for file in os.listdir(path):
        if 'twintowers' in file:
            if os.path.exists(f"{path}/{file}/ranking_debug.json"):
                record = next(SeqIO.parse(f"a3m_list/{file}.a3m", 'fasta'))
                id = record.id.replace(":", '')
                
                tmscore = "tmscore/TMscore"
                template = f"pdb_list/{id}.pdb"
                hh_pred = f"{path}/{id}_hhblits_formatted/ranked_0.pdb"
                tt_pred = f"{path}/{id}_twintowers/ranked_0.pdb"
                
                try:
                    hh_res = os.popen(f"{tmscore} {hh_pred} {template}").read()
                    hh_score = float(hh_res.split('\n')[16].split(' ')[5])
    
                    tt_res = os.popen(f"{tmscore} {tt_pred} {template}").read()
                    tt_score = float(tt_res.split('\n')[16].split(' ')[5])
                
                except Exception:
                    continue
                
                hh_result = json.load(open(f"{path}/{id}_hhblits_formatted/ranking_debug.json"))['plddts']['model_1']
                tt_result = json.load(open(f"{path}/{id}_twintowers/ranking_debug.json"))['plddts']['model_1']
                
                hh_num = len(list(SeqIO.parse(f"{path}/{id}_hhblits_formatted/msas/bfd_uniclust_hits.a3m", 'fasta')))
                tt_num = len(list(SeqIO.parse(f"a3m_list/{file}.a3m", 'fasta')))
                
               # hh_neff = calc_neff(f"{path}/{id}_hhblits_formatted/msas/bfd_uniclust_hits.a3m")
               # tt_neff = calc_neff(f"{path}/{id}_twintowers/msas/bfd_uniclust_hits.a3m")

               # print(f"no.{cnt} id: {id: <10}hhblits: {hh_result:.4f}   {hh_neff:.4f}   twintowers: {tt_result:.4f}   {tt_neff:.4f}")
                print(f"no.{cnt} seq_len: {len(record.seq)} id: {id: <10}hhblits: {hh_score:.4f} msa_num:{hh_num}"
                      f"    twintowers: {tt_score:.4f} msa_num:{tt_num}")
                cnt += 1
                hh_ave += hh_result
                tt_ave += tt_result
                hh_tm_ave += hh_score
                tt_tm_ave += tt_score

    print(f"ave hhblits: {hh_tm_ave/(cnt-1):.4f} twintowers: {tt_tm_ave/(cnt-1):.4f}")