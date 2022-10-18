import torch
import pandas as pd
import numpy as np
import random
from .tokenizer import Tokenizer
from torch.utils.data import Dataset
from Bio import SeqIO
from copy import copy


class RetrieveDataset(Dataset):
    def __init__(self, path):
        data = pd.read_csv(path, sep='\t')
        self.ori = data['ori'].values
        self.candi = data['candi'].values
        self.label = data['label'].values
        self.identity = data['identity'].values
        
        # self.ori = data['ori'].values
        # self.pos = data['pos'].values
        # self.neg = data['neg'].values
        # self.identity = data['identity'].values
        
        self.tokenizer = Tokenizer()

    def __len__(self):
        return len(self.ori)

    def __getitem__(self, index):
        return self.ori[index], self.candi[index], self.label[index], self.identity[index]

    def collate_fn(self, batch):
        ori, candi, label, identity = tuple(zip(*batch))
        # pad_id = self.tokenizer.convert_token_to_id('<pad>')
        seqs = ori + candi
        input_ids = self.tokenizer.batch_encode(seqs, padding=True)['input_ids']
        
        return {"input_ids": input_ids,
                "label": torch.tensor(label, dtype=torch.long),
                "identity": torch.tensor(identity, dtype=torch.float)}
        
        # ori_ids, ori_lengths = self.tokenizer.batch_encode(ori, padding=True).values()
        # pos_ids, pos_lengths = self.tokenizer.batch_encode(pos, padding=True).values()
        # neg_ids, neg_lengths = self.tokenizer.batch_encode(neg, padding=True).values()

        # return {'ori_ids': ori_ids,
        #         'pos_ids': pos_ids,
        #         'neg_ids': neg_ids,
        #         'identity': torch.tensor(identity, dtype=torch.float)}


class ThresholdDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path, sep='\t')
        self.seqs, self.thresholds = tuple(zip(*self.data.values))
        self.tokenizer = Tokenizer()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        seq, t = self.seqs[index], self.thresholds[index]
        return seq, t

    def collate_fn(self, batch):
        seqs, thresholds = tuple(zip(*batch))
        seq_ids, lengths = self.tokenizer.batch_encode(seqs, padding=True).values()
        thresholds = torch.tensor(thresholds)

        return {'seq_ids': seq_ids,
                'lengths': lengths,
                'thresholds': thresholds}


class AutoEncoderDataset(Dataset):
    def __init__(self, path, type='train'):
        self.data = torch.load(path).view(-1, 1, 64, 64)
        if type == 'train':
            self.data = self.data[:13000*12]
        else:
            self.data = self.data[13000*12:]

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, index):
        img = self.data[index]
        v = img.flatten().topk(200)[0][-1]
        return torch.where(img >= v, 1., 0.)


class MutationPredictionDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_csv(path, sep='\t').values
        self.tokenizer = Tokenizer()
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        seq, fitness = self.data[index]
        return seq, fitness

    def collate_fn(self, batch):
        seqs, fitness = tuple(zip(*batch))
        input_ids = self.tokenizer.batch_encode(seqs, padding=True)['input_ids']
        fitness = torch.tensor(fitness)
    
        return {"input_ids": input_ids,
                "labels": fitness}


class MaskedLanguageModelingDataset(Dataset):
    def __init__(self, fasta_path):
        super().__init__()
        self.tokenizer = Tokenizer()
        records = SeqIO.parse(fasta_path, 'fasta')
        self.seqs = [str(record.seq) for record in records]
        
    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, index):
        seq = self.seqs[index][:1022]
        tokens = self.tokenizer.tokenize(seq)
        tokens = self.tokenizer.add_special_tokens(tokens)
        masked_tokens, labels = self._apply_bert_mask(tokens)
        masked_token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(masked_tokens), dtype=torch.long)
    
        return masked_token_ids, labels

    def collate_fn(self, batch):
        input_ids, lm_label_ids = tuple(zip(*batch))
        pad_id = self.tokenizer.convert_token_to_id('<pad>')
    
        input_ids = self.tokenizer.pad_sequences(input_ids, pad_id)
        # ignore_index is -1
        lm_label_ids = self.tokenizer.pad_sequences(lm_label_ids, -1)
    
        return {'input_ids': input_ids,
                'labels': lm_label_ids}

    def _apply_bert_mask(self, tokens):
        masked_tokens = copy(tokens)
        labels = torch.full((len(tokens),), -1)
        for i in range(1, len(tokens)-1):
            token = tokens[i]
        
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
