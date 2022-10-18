import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import sys
sys.path.append("..")
from torch.utils.data.dataloader import DataLoader
import model_down
import time
import numpy as np
import esm
import itertools
model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
batch_converter = alphabet.get_batch_converter()
from Bio import SeqIO
import string
from typing import List, Tuple,Any
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)
def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)
def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)
def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

if __name__ == '__main__':
    batch_size = 1
    best_acc = 0

    model = model_down.ProteinBertForContactPrediction().cuda()
    model.load_state_dict(torch.load('msa_transformer_MLP_best_cp_ori.pt')['model_state_dict'])

    model.eval()
    val_tic = time.time()
    val_loss = 0
    val_p = 0
    val_step = 0
    name="7WED"
    a=read_msa('casp14/'+name+'.a3m',64)
    contact_inputs = batch_converter([a])[2]
    contact_targets = torch.from_numpy( np.load('casp14/'+name+'.npy')[np.newaxis,:] )
    protein_lengths = torch.as_tensor([len(contact_inputs[0][0])])
    contact_inputs, contact_targets, protein_lengths = contact_inputs.cuda(), contact_targets.cuda(), protein_lengths.cuda()
    with torch.no_grad():
        outputs = model(contact_inputs, protein_lengths, targets=contact_targets,finetune=False)
        loss_precision, value_prediction = outputs
        loss = loss_precision[0].mean()
        precision = loss_precision[1]['precision_at_l5'].mean()

    val_loss += loss.item()
    val_p += precision.item()
    val_step += 1

    # print("\n L/5 Step: {} / {} finish. Validating Loss: {:.5f}. Validating Precision: {:.5f}.\n".
    #         format(val_step, len(valid_loader), (val_loss / val_step), (val_p / val_step)))