import sys
import torch
import esm
import torch.nn as nn
import argparse
import msa_transformer_parser

args = msa_transformer_parser.params_parser()
# Load ESM-1b model
alphabet = esm.data.Alphabet.from_architecture('msa_transformer')
model = esm.model.MSATransformer(args, alphabet)
batch_converter = alphabet.get_batch_converter()


def _init_weights(module):
    """ Initialize the weights """
    if isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "avg"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, tokens, outputs):
        last_hidden = outputs
        attention_mask = 1 - tokens.eq(alphabet.padding_idx).type_as(outputs)

        if self.pooler_type in ['cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            last_hidden=torch.mean(last_hidden,dim=1)
            last_hidden=torch.mean(last_hidden,dim=1)
            #x=torch.mean(x)
            #return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
            return last_hidden
        else:
            raise NotImplementedError


class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.utils.weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            nn.utils.weight_norm(nn.Linear(hid_dim, out_dim), dim=None))
        self.apply(_init_weights)

    def forward(self, x):
        return self.main(x)


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleMLP(hidden_size, 1280, num_labels)

    def forward(self, pooled_output, targets=None):
        logits = self.classify(pooled_output)
        outputs = (logits,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            #print('logits',logits)
            classification_loss = loss_fct(logits, targets)
            metrics = {'accuracy': accuracy(logits, targets)}
            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs

        return outputs  # (loss), logits


class ProteinBertForSequenceClassification(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_labels = 1195
        self.bert = model
        self.classify = SequenceClassificationHead(
            model.args.embed_dim, self.num_labels)

    @torch.cuda.amp.autocast()
    def forward(self, input_ids, targets=None, finetune=True, finetune_emb=True):
        pooler_type = "avg"
        pooler = Pooler(pooler_type)
        what=self.bert.named_parameters()
        ##finetune will not use this for loop
        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
                print("1")
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
                print("2")
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False
                print("3")

        outputs = self.bert(input_ids, repr_layers=[12])

        sequence_output = outputs['representations'][12]

        pooled_output = pooler(input_ids, sequence_output)

        outputs = self.classify(pooled_output, targets)
        # (loss), prediction_scores

        return outputs
