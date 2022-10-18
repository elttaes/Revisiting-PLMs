import esm
import torch
import os
from Bio import SeqIO
import itertools
from typing import List, Tuple
import string

torch.set_grad_enabled(False)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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


#esm1b, esm1b_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
#esm1b = esm1b.eval().cuda()
esm1b_alphabet = esm.data.Alphabet.from_architecture('ESM-1b')

esm1b_batch_converter = esm1b_alphabet.get_batch_converter()

esm1b_data = [
    read_sequence("examples/1a3a_1_A.a3m"),
    read_sequence("examples/5ahw_1_A.a3m"),
    read_sequence("examples/1xcr_1_A.a3m"),
]

esm1b_batch_labels, esm1b_batch_strs, esm1b_batch_tokens = esm1b_batch_converter(esm1b_data)
esm1b_batch_tokens = esm1b_batch_tokens.cuda()
print(esm1b_batch_tokens.size(), esm1b_batch_tokens.dtype)  # Should 

#msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
#msa_transformer = msa_transformer.eval().cuda()
msa_alphabet = esm.data.Alphabet.from_architecture('msa_transformer')
#msa_alphabet.get_batch_converter()
msa_batch_converter = msa_alphabet.get_batch_converter()

# msa_data = [
#     read_msa("examples/1a3a_1_A.a3m", 64),
#     read_msa("examples/5ahw_1_A.a3m", 64),
#     read_msa("examples/1xcr_1_A.a3m", 64),
# ]
s3m1_dir='/root/metal/msa-transformer/a3m1'
filenames = [
    name for name in os.listdir(s3m1_dir)
    if os.path.splitext(name)[-1] == '.a3m'
]  #选择指定目录下的.png图片
msa_data=[]
for i in range(len(filenames)):
    msa_data.append(read_msa(os.path.join(s3m1_dir, filenames[i]), 64))

msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
msa_batch_tokens = msa_batch_tokens.cuda()


