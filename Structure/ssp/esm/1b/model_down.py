import torch
import esm
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
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
            last_hidden = last_hidden[:,0,0]
            return last_hidden
        elif self.pooler_type == "avg":
            last_hidden=torch.mean(last_hidden,dim=1)
            last_hidden=torch.mean(last_hidden,dim=1)
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

class SimpleConv(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_dim), 
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x

def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels)# * valid_mask
        return correct.sum().float() / logits.shape[0]#valid_mask.sum().float()

class Accuracy(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target, self.ignore_index)
    
class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 ignore_index: int = -100):
        super().__init__()
        self.classify = SimpleConv(hidden_size, 1280, num_labels)
        self.num_labels = num_labels
        self._ignore_index = ignore_index

    def forward(self, sequence_output, targets=None):
        sequence_logits = self.classify(sequence_output)
        outputs = (sequence_logits,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            classification_loss = loss_fct(
                sequence_logits.view(-1, self.num_labels), targets.view(-1))
            acc_fct = Accuracy(ignore_index=self._ignore_index)
            metrics = {'accuracy':
                       acc_fct(sequence_logits.view(-1, self.num_labels), targets.view(-1))}
            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs

class ProteinBertForSequence2Sequence(nn.Module):

    def __init__(self):
        super().__init__()
        self.num_labels = 10
        self.bert = model
        self.classify = SequenceToSequenceClassificationHead(
            model.args.embed_dim, self.num_labels)

    @torch.cuda.amp.autocast()
    def forward(self, input_ids, targets=None, finetune=True, finetune_emb=True):
        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False


        outputs = self.bert(input_ids, repr_layers=[33])
        sequence_output = outputs['representations'][33]
        outputs = self.classify(sequence_output, targets)

        return outputs
