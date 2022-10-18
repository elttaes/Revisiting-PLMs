import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import esm
import os
from Bio import SeqIO
import itertools
from typing import List, Tuple,Any
import string
import torch
import numpy as np
from collections import OrderedDict

DRUG_VOCAB = OrderedDict([
('Fusidic acid',0),
('Fluoroquinolones',1),
('Betalactams',2),
('Nucleosides',3),
('Glycopeptides',4),
('Rifampin',5),
('Triclosan',6),
('Trimethoprim',7),
('Bacitracin',8),
('Phenicol',9),
('Tetracyclines',10),
('MLS',11),
('Aminoglycosides',12),
('Aminocoumarins',13),
('Peptide',14),
('Multi-drug resistance',15),
('Sulfonamide',16),
('Fosfomycin',17),
('Mupirocin',18)])


class Drug_Tokenizer():

    def __init__(self, vocab: str = 'drug'):
        if vocab == 'drug':
            self.vocab = DRUG_VOCAB
        self.tokens = list(self.vocab.keys())
        self._vocab_type = vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def convert_token_to_id(self, token: str) -> int:
        """ Converts a token (str/unicode) in an id using the vocab. """
        try:
            return self.vocab[token]
        except KeyError:
            raise KeyError(f"Unrecognized token: '{token}'")

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.convert_token_to_id(token) for token in tokens]

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

    def convert_ids_to_tokens(self, indices: List[int]) -> List[str]:
        return [self.convert_id_to_token(id_) for id_ in indices]

a3m_dir='/home/public/bigdata/my/datasets/drug/a3m/label'
filenames = [
    os.path.join(a3m_dir,name) for name in os.listdir(a3m_dir)
    #if os.path.splitext(name)[-1] == '.a3m'
]
magic_name=Drug_Tokenizer()
label_dict=dict()
for file in filenames:
    f = open(file)
    #读取文件名
    name=os.path.split(file)[-1]
    #读取label
    lines = f.read()
    label_dict[name]=magic_name.convert_token_to_id(lines)
f.close()


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
    
msa_alphabet = esm.data.Alphabet.from_architecture('msa_transformer')
msa_batch_converter = msa_alphabet.get_batch_converter()

class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, list):
        'Initialization'
        self.list = list

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list)

    def __getitem__(self, index):
        ID = self.list[index]
        return ID

    def collate_fn(self, batch: List[Tuple[Any, ...]]):
        msa_data = []
        label = []
        for i in range(len(batch)):
            msa_data.append(read_msa(os.path.join(batch[i]), 1))
        msa_batch_label, msa_batch_str, msa_batch_token = msa_batch_converter(msa_data)
        for i in range(len(batch)):
            filedir,filename=os.path.split(batch[i])
            label.append(label_dict[filename[:-4]])
        return [torch.as_tensor(msa_batch_token),torch.as_tensor(label)]
