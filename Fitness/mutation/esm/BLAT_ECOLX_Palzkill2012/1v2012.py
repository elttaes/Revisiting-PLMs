from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
from Bio import SeqIO
import itertools
from typing import List, Tuple
import torch
import scipy.stats
import numpy as np
#from BLAT_ECOLX_Ostermeier2014 import seq,res
from BLAT_ECOLX_Palzkill2012 import seq,res

def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.mstats.spearmanr(target_array, prediction_array).correlation

model1, alphabet = pretrained.load_model_and_alphabet('/home/my/nips/pretrained_models/esm1v_t33_650M_UR90S_1.pt')
model2, alphabet = pretrained.load_model_and_alphabet('/home/my/nips/pretrained_models/esm1v_t33_650M_UR90S_2.pt')
model3, alphabet = pretrained.load_model_and_alphabet('/home/my/nips/pretrained_models/esm1v_t33_650M_UR90S_3.pt')
model4, alphabet = pretrained.load_model_and_alphabet('/home/my/nips/pretrained_models/esm1v_t33_650M_UR90S_4.pt')
model5, alphabet = pretrained.load_model_and_alphabet('/home/my/nips/pretrained_models/esm1v_t33_650M_UR90S_5.pt')
#print(alphabet.get_tok(0))

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()

#data = [read_msa('/home/my/nips/mutation/mutation_data/msa/BLAT_ECOLX_1_b0.5.a3m', 1)][0]
batch_converter = alphabet.get_batch_converter()
#print(alphabet.get_tok(32))
#print(alphabet.get_idx('A'))
data="HPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTFKVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIGGPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEADKVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
batch_labels, batch_strs, batch_tokens = batch_converter([['sss',data]])
result=[]
result1=model1(batch_tokens)['logits'][0][1:]
result.append(torch.log_softmax(result1,dim=1))
result2=model2(batch_tokens)['logits'][0][1:]
result.append(torch.log_softmax(result2,dim=1))
result3=model3(batch_tokens)['logits'][0][1:]
result.append(torch.log_softmax(result3,dim=1))
result4=model4(batch_tokens)['logits'][0][1:]
result.append(torch.log_softmax(result4,dim=1))
result5=model5(batch_tokens)['logits'][0][1:]
result.append(torch.log_softmax(result5,dim=1))

spear=[]
for z in range(len(result)):
    seq_res=[]
    exp_res=[]
    for i in range(len(seq)):
        count=int(seq[i][1:-1])-24
        mutation=alphabet.get_idx(seq[i][-1])
        origin=alphabet.get_idx(seq[i][0])
        seq_res.append(result[z][count][mutation].detach().numpy()-result[z][count][origin].detach().numpy())
    for i in range(len(res)):
        exp_res.append(float(res[i]))
    spear.append(spearmanr(seq_res,exp_res))
print(spear)
print(np.mean(np.array(spear)))
# import numpy as np
# aa=[0.5073380425733428, 0.5353412103394046, 0.5172433844231531, 0.5349215274078652, 0.5517251904780761]
# print(np.mean(np.array(aa)))
