from typing import Union, List, Tuple, Sequence, Dict, Any
from copy import copy
from pathlib import Path
from collections import OrderedDict
import random
import numpy as np
from einops import repeat
import torch
from torch.utils.data import Dataset
from .tokenizers import Tokenizer

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

class MLMDataset_Uniref50(Dataset):

    def __init__(self,
                 split: str,
                 tokenizer: Union[str, Tokenizer] = 'iupac'):
        super().__init__()
        if split not in ('train', 'val'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'val']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_file = f'../data/pretrain/uniref50_shuffle_{split}.txt'
        data = open(data_file, 'r', encoding='utf-8')
        self.data = data.readlines()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        item = item.replace('\n', '')
        seq_length = len(item)
        # tokens = self.tokenizer.tokenize(item)
        if seq_length <= 256:
            tokens = self.tokenizer.tokenize(item)
        else:
            tokens = self.tokenizer.tokenize(item[:256])
        tokens = self.tokenizer.add_special_tokens(tokens)
        ori_tokens = copy(tokens)
        ori_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(ori_tokens), np.int64)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return ori_token_ids, masked_token_ids, labels

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        ori_ids, masked_ids, lm_label_ids = tuple(zip(*batch))

        ori_ids = torch.from_numpy(pad_sequences(ori_ids, 1))
        masked_ids = torch.from_numpy(pad_sequences(masked_ids, 1))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

        return {'ori_ids': ori_ids,
                'masked_ids': masked_ids,
                'targets': lm_label_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                pass

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels

class MLMDataset_MSA(Dataset):

    def __init__(self,
                 split: str,
                 tokenizer: Union[str, Tokenizer] = 'iupac',
                 random_sample_msa: bool = False,
                 max_seq_len: int = 256,
                 max_msa_num: int = 64):
        super().__init__()
        if split not in ('train', 'val', 'test'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'val', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer
        self.random_sample_msa = random_sample_msa
        self.max_seq_len = max_seq_len
        self.max_msa_num = max_msa_num

        data_file = f'../data/pretrain/msa_{split}.npy'
        self.data = np.load(data_file, allow_pickle=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        msa = np.array([np.array(self.tokenizer.add_special_tokens(list(seq))) for seq in self.data[index]])
        seq = msa[0]
        item = {
            "seq": seq,
            "msa": msa
        }
        item["msa"] = self.sample(item["msa"], self.max_msa_num, self.random_sample_msa)
        item = self.crop(item, self.max_seq_len)

        return item

    def crop(self, item, max_seq_len: int):
        seq_len = len(item["seq"])

        if seq_len <= max_seq_len or max_seq_len <= 0:
            return item

        start = 0
        end = start + max_seq_len
        item["seq"] = item["seq"][start:end]
        item["msa"] = item["msa"][:, start:end]

        return item

    def sample(self, msa, max_msa_num: int, random: bool):
        num_msa, seq_len = len(msa), len(msa[0])

        if num_msa <= max_msa_num or max_msa_num <= 0:
            return msa

        if random:
            num_sample = max_msa_num - 1
            indices = np.random.choice(num_msa - 1, size=num_sample, replace=False) + 1
            indices = np.pad(indices, [1, 0], "constant")
            return msa[indices]
        else:
            return msa[:max_msa_num]

    def collate_fn(self, batch):
        b = len(batch)
        batch = {k: [item[k] for item in batch] for k in batch[0]}
        seq = batch["seq"]
        msa = batch["msa"]

        masked_seq, masked_seq_labels = self.apply_bert_mask_seq(seq)
        masked_msa, masked_msa_labels = self.apply_bert_mask_msa(msa)

        lengths = torch.LongTensor([len(x[0]) for x in msa])
        depths = torch.LongTensor([len(x) for x in msa])
        max_len = lengths.max()
        max_depth = depths.max()

        seq_id = pad_sequences(
            [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(seq_)) for seq_ in seq], 1,
        )

        msa_id = pad_sequences(
            [torch.LongTensor([self.tokenizer.convert_tokens_to_ids(seq_) for seq_ in msa_]) for msa_ in msa],
            1,
        )

        masked_seq_id = pad_sequences(
            [torch.LongTensor(self.tokenizer.convert_tokens_to_ids(seq_)) for seq_ in masked_seq], 1,
        )

        masked_seq_labels = pad_sequences(
            [torch.LongTensor(seq_) for seq_ in masked_seq_labels], -1,
        )

        masked_msa_id = pad_sequences(
            [torch.LongTensor([self.tokenizer.convert_tokens_to_ids(seq_) for seq_ in msa_]) for msa_ in masked_msa],
            1,
        )

        masked_msa_labels = pad_sequences(
            [torch.LongTensor([seq_ for seq_ in msa_]) for msa_ in masked_msa_labels],
            -1,
        )

        mask = repeat(torch.arange(max_len), "l -> b l", b=b) < repeat(
            lengths, "b -> b l", l=max_len
        )
        msa_seq_mask = repeat(
            torch.arange(max_len), "l -> b s l", b=b, s=max_depth
        ) < repeat(lengths, "b -> b s l", s=max_depth, l=max_len)
        msa_depth_mask = repeat(
            torch.arange(max_depth), "s -> b s l", b=b, l=max_len
        ) < repeat(depths, "b -> b s l", s=max_depth, l=max_len)
        msa_mask = msa_seq_mask & msa_depth_mask

        return {'seq_id': seq_id,
                'masked_seq_id': masked_seq_id,
                'masked_seq_labels': masked_seq_labels,
                'msa_id': msa_id,
                'masked_msa_id': masked_msa_id,
                'masked_msa_labels': masked_msa_labels,
                'mask': mask,
                'msa_mask': msa_mask
                }

    def apply_bert_mask_seq(self, batch_tokens):
        masked_tokens_list = []
        labels_list = []
        for tokens in batch_tokens:
            masked_tokens = copy(tokens)
            labels = np.zeros([len(tokens)], np.int64) - 1
            for i, token in enumerate(tokens):
                # Tokens begin and end with start_token and stop_token, ignore these
                if token in (self.tokenizer.start_token, self.tokenizer.stop_token, '-'):
                    pass

                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    labels[i] = self.tokenizer.convert_token_to_id(token)

                    if prob < 0.8:
                        # 80% random change to mask token
                        token = self.tokenizer.mask_token
                    elif prob < 0.9:
                        # 10% chance to change to random token
                        token = self.tokenizer.convert_id_to_token(
                            random.randint(0, self.tokenizer.vocab_size - 1))
                    else:
                        # 10% chance to keep current token
                        pass

                    masked_tokens[i] = token

            masked_tokens_list.append(masked_tokens)
            labels_list.append(labels)

        return masked_tokens_list, labels_list

    def apply_bert_mask_msa(self, batch_msas):
        msa_masked_tokens_list = []
        msa_labels_list = []
        for batch_tokens in batch_msas:
            masked_tokens_list = []
            labels_list = []
            for tokens in batch_tokens:
                masked_tokens = copy(tokens)
                labels = np.zeros([len(tokens)], np.int64) - 1
                for i, token in enumerate(tokens):
                    # Tokens begin and end with start_token and stop_token, ignore these
                    if token in (self.tokenizer.start_token, self.tokenizer.stop_token, '-'):
                        pass

                    prob = random.random()
                    if prob < 0.15:
                        prob /= 0.15
                        labels[i] = self.tokenizer.convert_token_to_id(token)

                        if prob < 0.8:
                            # 80% random change to mask token
                            token = self.tokenizer.mask_token
                        elif prob < 0.9:
                            # 10% chance to change to random token
                            token = self.tokenizer.convert_id_to_token(
                                random.randint(0, self.tokenizer.vocab_size - 1))
                        else:
                            # 10% chance to keep current token
                            pass

                        masked_tokens[i] = token

                masked_tokens_list.append(masked_tokens)
                labels_list.append(labels)

            msa_masked_tokens_list.append(masked_tokens_list)
            msa_labels_list.append(labels_list)

        return msa_masked_tokens_list, msa_labels_list

class MLMDataset_ESM_SSD(Dataset):

    def __init__(self,
                 split: str,
                 tokenizer: Union[str, Tokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()
        if split not in ('train', 'val', 'test'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'val', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_file = f'../data/esm_ssd/trrosetta_msa_{split}.txt'
        data = open(data_file, 'r', encoding='utf-8')
        self.data = data.readlines()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        # item = item.replace('\n', '')
        item = item.split('\t')
        seq = item[0]
        msa_id = item[1].replace('\n', '')
        seq_length = len(seq)
        if seq_length <= 256:
            tokens = self.tokenizer.tokenize(seq)
        else:
            tokens = self.tokenizer.tokenize(seq[:256])
        tokens = self.tokenizer.add_special_tokens(tokens)
        ori_tokens = copy(tokens)
        ori_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(ori_tokens), np.int64)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return ori_token_ids, masked_token_ids, labels

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        ori_ids, masked_ids, lm_label_ids = tuple(zip(*batch))

        ori_ids = torch.from_numpy(pad_sequences(ori_ids, 1))
        masked_ids = torch.from_numpy(pad_sequences(masked_ids, 1))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

        return {'ori_ids': ori_ids,
                'masked_ids': masked_ids,
                'targets': lm_label_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                pass

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels

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

class MLMDataset_ESM_SSD_ID(Dataset):

    def __init__(self,
                 split: str,
                 tokenizer: Union[str, Tokenizer] = 'iupac',
                 in_memory: bool = False):
        super().__init__()
        if split not in ('train', 'val', 'test'):
            raise ValueError(
                f"Unrecognized split: {split}. "
                f"Must be one of ['train', 'val', 'test']")
        if isinstance(tokenizer, str):
            tokenizer = Tokenizer(vocab=tokenizer)
        self.tokenizer = tokenizer

        data_file = f'../data/esm_ssd/trrosetta_msa_{split}.txt'
        data = open(data_file, 'r', encoding='utf-8')
        self.data = data.readlines()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        item = item.split('\t')
        seq = item[0]
        msa_name = item[1].replace('\n', '')
        id_tokenizer = ID_Tokenizer(vocab='id')
        msa_id = id_tokenizer.convert_token_to_id(msa_name)
        seq_length = len(seq)
        if seq_length <= 256:
            tokens = self.tokenizer.tokenize(seq)
        else:
            tokens = self.tokenizer.tokenize(seq[:256])
        tokens = self.tokenizer.add_special_tokens(tokens)
        ori_tokens = copy(tokens)
        ori_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(ori_tokens), np.int64)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = np.array(
            self.tokenizer.convert_tokens_to_ids(masked_tokens), np.int64)

        return msa_id, ori_token_ids, masked_token_ids, labels

    def collate_fn(self, batch: List[Any]) -> Dict[str, torch.Tensor]:
        msa_id, ori_ids, masked_ids, lm_label_ids = tuple(zip(*batch))
        msa_id = np.array(msa_id, np.int64)
        msa_ids = torch.from_numpy(msa_id)
        ori_ids = torch.from_numpy(pad_sequences(ori_ids, 1))
        masked_ids = torch.from_numpy(pad_sequences(masked_ids, 1))
        # ignore_index is -1
        lm_label_ids = torch.from_numpy(pad_sequences(lm_label_ids, -1))

        return {'msa_ids': msa_ids,
                'ori_ids': ori_ids,
                'masked_ids': masked_ids,
                'targets': lm_label_ids}

    def _apply_bert_mask(self, tokens: List[str]) -> Tuple[List[str], List[int]]:
        masked_tokens = copy(tokens)
        labels = np.zeros([len(tokens)], np.int64) - 1

        for i, token in enumerate(tokens):
            # Tokens begin and end with start_token and stop_token, ignore these
            if token in (self.tokenizer.start_token, self.tokenizer.stop_token):
                pass

            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                labels[i] = self.tokenizer.convert_token_to_id(token)

                if prob < 0.8:
                    # 80% random change to mask token
                    token = self.tokenizer.mask_token
                elif prob < 0.9:
                    # 10% chance to change to random token
                    token = self.tokenizer.convert_id_to_token(
                        random.randint(0, self.tokenizer.vocab_size - 1))
                else:
                    # 10% chance to keep current token
                    pass

                masked_tokens[i] = token

        return masked_tokens, labels