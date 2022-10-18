from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from Bio import SeqIO
import itertools
from typing import List, Tuple
import torch
import scipy.stats
import numpy as np
#from BLAT_ECOLX_Ostermeier2014 import seq,res
from BG_STRSQ_Abate2015 import seq,res

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
    msa = [(desc, seq.upper()) for desc, seq in msa]
    return msa

model, alphabet = pretrained.load_model_and_alphabet('/home/my/nips/pretrained_models/esm1b_t33_650M_UR50S.pt')
#print(alphabet.get_tok(0))

model.eval()
#data = [read_msa('/home/my/nips/mutation/mutation_data/msa/BLAT_ECOLX_1_b0.5.a3m', 1)][0]
batch_converter = alphabet.get_batch_converter()
#print(alphabet.get_tok(32))
#print(alphabet.get_idx('A'))
data="VPAAQQTAMAPDAALTFPEGFLWGSATASYQIEGAAAEDGRTPSIWDTYARTPGRVRNGDTGDVATDHYHRWREDVALMAELGLGAYRFSLAWPRIQPTGRGPALQKGLDFYRRLADELLAKGIQPVATLYHWDLPQELENAGGWPERATAERFAEYAAIAADALGDRVKTWTTLNEPWCSAFLGYGSGVHAPGRTDPVAALRAAHHLNLGHGLAVQALRDRLPADAQCSVTLNIHHVRPLTDSDADADAVRRIDALANRVFTGPMLQGAYPEDLVKDTAGLTDWSFVRDGDLRLAHQKLDFLGVNYYSPTLVSEADGSGTHNSDGHGRSAHSPWPGADRVAFHQPPGETTAMGWAVDPSGLYELLRRLSSDFPALPLVITENGAAFHDYADPEGNVNDPERIAYVRDHLAAVHRAIKDGSDVRGYFLWSLLDNFEWAHGYSKRFGAVYVDYPTGTRIPKASARWYAEVARTGVLPTAGDPNSSSVDKLAAALEHHHHHH"
batch_labels, batch_strs, batch_tokens = batch_converter([['sss',data]])
result=model(batch_tokens)['logits'][0][1:]
result=torch.log_softmax(result,dim=1)
seq_res=[]
exp_res=[]
for i in range(len(seq)):
    count=int(seq[i][1:-1])-2
    mutation=alphabet.get_idx(seq[i][-1])
    origin=alphabet.get_idx(seq[i][0])
    seq_res.append(result[count][mutation].detach().numpy()-result[count][origin].detach().numpy())
for i in range(len(res)):
    exp_res.append(float(res[i]))
print(spearmanr(seq_res,exp_res))