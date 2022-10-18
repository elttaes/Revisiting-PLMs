from typing import Union, List, Tuple, Sequence, Dict, Any, Optional, Collection
from copy import copy
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizers import Tokenizer

f = open('../data/esm_ssd/trrosetta_msa_id.txt', 'r')

id_list = []
for i, id in enumerate(f.readlines()):
    id_list.append(tuple((id.replace('\n', ''), i)))

ID_VOCAB = OrderedDict(id_list)

class ID_Tokenizer():

    def __init__(self, vocab: str = 'id'):
        if vocab == 'id':
            self.vocab = ID_VOCAB
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

    def convert_id_to_token(self, index: int) -> str:
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        try:
            return self.tokens[index]
        except IndexError:
            raise IndexError(f"Unrecognized index: '{index}'")

SSP_VOCAB = OrderedDict([
    ('-',  -1),
    ('H', 0),
    ('G',  1),
    ('I',  2),
    ('E',  3),
    ('B',  4),
    ('T',  5),
    ('S',  6),
    ('X',  7)])

class SSP_Tokenizer():

    def __init__(self, vocab: str = 'ssp'):
        if vocab == 'ssp':
            self.vocab = SSP_VOCAB
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

def pad_sequences(sequences: Sequence, constant_value=0, dtype=None) -> np.ndarray:
    batch_size = len(sequences)
    shape = [batch_size] + np.max([seq.shape for seq in sequences], 0).tolist()

    if dtype is None:
        dtype = sequences[0].dtype

    if isinstance(sequences[0], np.ndarray):
        array = np.full(shape, constant_value, dtype=dtype)
    elif isinstance(sequences[0], torch.Tensor):
        array = torch.full(shape, constant_value, dtype=dtype)

    for arr, seq in zip(array, sequences):
        arrslice = tuple(slice(dim) for dim in seq.shape)
        arr[arrslice] = seq

    return array

# task('ssp')
class SSPDataset(Dataset):

    def __init__(self,
                 split: str,
                 tokenizer: Union[str, Tokenizer] = 'iupac'):
        super().__init__()
        if split not in ('train', 'val', 'test'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'val', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_file = f'../data/esm_ssd/msa_{split}_fold.npz'
        data = np.load(data_file, allow_pickle=True)
        # self.data = data
        self.msa_name = data['msa_name']
        self.src_seq = data['src_seq']
        self.ssp = data['ssp']
        self.dist = data['dist']

    def __len__(self) -> int:
        # return len(self.data)
        return len(self.msa_name)

    def __getitem__(self, index):
        msa_name = self.msa_name[index]
        id_tokenizer = ID_Tokenizer(vocab='id')
        msa_id = id_tokenizer.convert_token_to_id(msa_name)
        src_seq = self.src_seq[index]
        ssp = self.ssp[index]
        dist = self.dist[index]
        seq_length = len(src_seq)
        if seq_length <= 256:
            tokens = self.tokenizer.tokenize(src_seq)
        else:
            tokens = self.tokenizer.tokenize(src_seq[:256])
            ssp = ssp[:256]
            dist = dist[:256, :256]
        tokens = self.tokenizer.add_special_tokens(tokens)
        ori_tokens = copy(tokens)
        ori_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(ori_tokens), np.int64)
        ssp_tokenizer = SSP_Tokenizer(vocab='ssp')
        ssp = ssp_tokenizer.convert_tokens_to_ids(ssp)
        ssp_labels = np.asarray(ssp, np.int64)
        ssp_labels = np.pad(ssp_labels, (1, 1), 'constant', constant_values=-1)

        return msa_id, ori_token_ids, ssp_labels, dist

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        msa_id, ori_token_ids, ssp_labels, dist = tuple(zip(*batch))
        msa_id = np.array(msa_id, np.int64)
        msa_ids = torch.from_numpy(msa_id)
        ori_ids = torch.from_numpy(pad_sequences(ori_token_ids, 1))
        ssp_label_ids = torch.from_numpy(pad_sequences(ssp_labels, -1))

        return {'msa_ids': msa_ids,
                'ori_ids': ori_ids,
                'ssp_label_ids': ssp_label_ids,
                }

# task('cp')
class CPDataset(Dataset):

    def __init__(self,
                 split: str,
                 tokenizer: Union[str, Tokenizer] = 'iupac'):
        super().__init__()
        if split not in ('train', 'val', 'test'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'val', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_file = f'../data/esm_ssd/msa_{split}_fold.npz'
        data = np.load(data_file, allow_pickle=True)
        # self.data = data
        self.msa_name = data['msa_name']
        self.src_seq = data['src_seq']
        self.ssp = data['ssp']
        self.dist = data['dist']

    def __len__(self) -> int:
        # return len(self.data)
        return len(self.msa_name)

    def __getitem__(self, index):
        msa_name = self.msa_name[index]
        id_tokenizer = ID_Tokenizer(vocab='id')
        msa_id = id_tokenizer.convert_token_to_id(msa_name)
        src_seq = self.src_seq[index]
        dist = self.dist[index]
        seq_length = len(src_seq)
        if seq_length <= 256:
            tokens = self.tokenizer.tokenize(src_seq)
        else:
            seq_length = 256
            tokens = self.tokenizer.tokenize(src_seq[:256])
            dist = dist[:256, :256]
        tokens = self.tokenizer.add_special_tokens(tokens)
        ori_tokens = copy(tokens)
        ori_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(ori_tokens), np.int64)
        contact_labels = (dist < 8).astype(int)
        valid_mask = np.full((seq_length, ), True)
        yind, xind = np.indices(contact_labels.shape)
        invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
        invalid_mask |= np.abs(yind - xind) < 6
        contact_labels[invalid_mask] = -1

        return msa_id, ori_token_ids, contact_labels, seq_length

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        msa_id, ori_token_ids, contact_labels, seq_length = tuple(zip(*batch))
        msa_id = np.array(msa_id, np.int64)
        msa_ids = torch.from_numpy(msa_id)
        ori_ids = torch.from_numpy(pad_sequences(ori_token_ids, 1))
        contact_label_ids = torch.from_numpy(pad_sequences(contact_labels, -1))
        protein_length = torch.LongTensor(seq_length)

        return {'msa_ids': msa_ids,
                'ori_ids': ori_ids,
                'contact_label_ids': contact_label_ids,
                'protein_length': protein_length
                }