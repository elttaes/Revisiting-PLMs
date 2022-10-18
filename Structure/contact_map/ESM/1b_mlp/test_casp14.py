import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import sys
sys.path.append("..")
import model_down
import time
import numpy as np
import esm
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
from Bio import SeqIO
import string
from typing import List, Tuple
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)
def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

if __name__ == '__main__':
    batch_size = 1
    best_acc = 0

    model = model_down.ProteinBertForContactPrediction().cuda()
    model.load_state_dict(torch.load('ESM1b_MLP_cp_ori.pt')['model_state_dict'])

    model.eval()
    val_tic = time.time()
    val_loss = 0
    val_p = 0
    val_step = 0
    name='6POO'
    a,b=read_sequence('casp14/rcsb_pdb_'+name+'.fasta')
    contact_inputs = batch_converter([[a,b]])[2]
    contact_targets = torch.from_numpy( np.load('casp14/'+name+'.npy')[np.newaxis,:] )
    protein_lengths = torch.as_tensor([len(contact_inputs[0])])
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