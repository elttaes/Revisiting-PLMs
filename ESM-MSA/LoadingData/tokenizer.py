from typing import List
import logging
import torch
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)
ESM_1B = OrderedDict([
    ("<cls>", 0),
    ("<pad>", 1),
    ("<sep>", 2),
    ("<unk>", 3),
    ("L", 4),
    ("A", 5),
    ("G", 6),
    ("V", 7),
    ("S", 8),
    ("E", 9),
    ("R", 10),
    ("T", 11),
    ("I", 12),
    ("D", 13),
    ("P", 14),
    ("K", 15),
    ("Q", 16),
    ("N", 17),
    ("F", 18),
    ("Y", 19),
    ("M", 20),
    ("H", 21),
    ("W", 22),
    ("C", 23),
    ("X", 24),
    ("B", 25),
    ("U", 26),
    ("Z", 27),
    ("O", 28),
    (".", 29),
    ("-", 30),
    ("<null_1>", 31),
    ("<mask>", 32)])


class Tokenizer():
    def __init__(self):
        self.vocab = ESM_1B
        self.tokens = list(self.vocab.keys())
        assert self.start_token in self.vocab and self.stop_token in self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def start_token(self) -> str:
        return "<cls>"

    @property
    def stop_token(self) -> str:
        return "<sep>"

    @property
    def mask_token(self) -> str:
        if "<mask>" in self.vocab:
            return "<mask>"
        else:
            raise RuntimeError(f"{self._vocab_type} vocab does not support masking")

    def tokenize(self, text: str) -> List[str]:
        return [x for x in text]

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

    def convert_tokens_to_string(self, tokens: str) -> str:
        """ Converts a sequence of tokens (string) in a single string. """
        return ''.join(tokens)

    def add_special_tokens(self, token_ids: List[str]) -> List[str]:
        """
        Adds special tokens to the a sequence for sequence classification tasks.
        A BERT sequence has the following format: [CLS] X [SEP]
        """
        cls_token = [self.start_token]
        sep_token = [self.stop_token]
        return cls_token + token_ids + sep_token

    def single_encode(self, text: str) -> np.ndarray:
        tokens = self.tokenize(text)
        tokens = self.add_special_tokens(tokens)
        token_ids = self.convert_tokens_to_ids(tokens)
        return np.array(token_ids, np.int64)

    def batch_encode(self, seqs, padding=True):
        batch = [self.single_encode(seq) for seq in seqs]
        pad_id = self.convert_token_to_id('<pad>')
        lengths = [len(seq) for seq in seqs]
        input_ids = torch.from_numpy(self.pad_sequences(batch, pad_id)) if padding else batch

        out = {'input_ids': input_ids, 'lengths': lengths}
        return out
    
    def add_mask_token(self, ids, locations):
        mask_id = self.vocab[self.mask_token]
        ids[locations] = mask_id
        return ids

    # 进行序列的padding操作
    def pad_sequences(self, sequences, constant_value=0, dtype=None) -> np.ndarray:
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
