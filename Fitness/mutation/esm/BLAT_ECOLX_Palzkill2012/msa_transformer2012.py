from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
import itertools
from typing import List, Tuple
import torch
import scipy.stats
import numpy as np
from BLAT_ECOLX_Palzkill2012 import seq,res

def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.mstats.spearmanr(target_array, prediction_array).correlation

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""

    msa = [
        (record.description, str(record.seq))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    msa = [(desc, seq.upper()[:263]) for desc, seq in msa]
    return msa

model, alphabet = pretrained.esm_msa1b_t12_100M_UR50S()
model.eval()
msa_num=[1,2,4,8,16,24,32,64,128,256]
for q in range(len(msa_num)):
    data = [read_msa('/home/my/nips/mutation/mutation_data/msa/BLAT_ECOLX_Palzkill2012/bfd_uniclust_hits.a3m', msa_num[q])]
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    result=model(batch_tokens)['logits'][0][0][1:]
    result=torch.log_softmax(result,dim=1)
    seq_res=[]
    exp_res=[]
    for i in range(len(seq)):
        #print(result[i][constants.HHBLITS_AA_TO_ID[seq[i][-1]]])
        count=int(seq[i][1:-1])-24
        mutation=alphabet.get_idx(seq[i][-1])
        origin=alphabet.get_idx(seq[i][0])
        seq_res.append(result[count][mutation].detach().numpy()-result[count][origin].detach().numpy())
    for i in range(len(res)):
        exp_res.append(float(res[i]))
    print(spearmanr(seq_res,exp_res))