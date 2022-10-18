import esm
from .model_utils import ValuePredictionHead
from .model_utils import SequenceClassificationHead
from .model_utils import SequenceToSequenceClassificationHead
from .model_utils import PairwiseContactPredictionHead
import torch.nn as nn

esm1_pretrained, esm1_alphabet = esm.pretrained.load_model_and_alphabet("../pretrained_models/esm1b_t33_650M_UR50S.pt")

esm1 = esm.model_paf.ProteinBertModel_paf(esm1_pretrained.args, esm1_alphabet)

for k_pretrained, v_pretrained in esm1_pretrained.named_parameters():
    for k, v in esm1.named_parameters():
        if k == k_pretrained:
            v.data = v_pretrained.clone()
        if 'layers' in k_pretrained and 'parallel_' + k_pretrained in k:
            v.data = v_pretrained.clone()

for k, v in esm1.named_parameters():
    v.requires_grad = True
    if k.startswith('layers'):
        v.requires_grad = False


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
        attention_mask = 1 - tokens.eq(esm1_alphabet.padding_idx).type_as(outputs)

        if self.pooler_type in ['cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError

# task_model('fluorescence', 'transformer')
# task_model('stability', 'transformer')
class ProteinBertForValuePrediction(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = esm1
        self.predict = ValuePredictionHead(esm1.args.embed_dim)


    def forward(self, input_ids, targets=None, policy=None, finetune=True, finetune_emb=True):
        pooler_type = "cls"
        pooler = Pooler(pooler_type)

        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False

        outputs = self.bert(input_ids, policy, repr_layers=[33])

        sequence_output = outputs['representations'][33]

        pooled_output = pooler(input_ids, sequence_output)

        outputs = self.predict(pooled_output, targets)
        # (loss), prediction_scores

        return outputs


# task_model('remote_homology', 'transformer')
class ProteinBertForSequenceClassification(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_labels = 1195
        self.bert = esm1
        self.classify = SequenceClassificationHead(
            esm1.args.embed_dim, self.num_labels)

    def forward(self, input_ids, targets=None, policy=None, finetune=True, finetune_emb=True):
        pooler_type = "cls"
        pooler = Pooler(pooler_type)

        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False

        outputs = self.bert(input_ids, policy, repr_layers=[33])

        sequence_output = outputs['representations'][33]

        pooled_output = pooler(input_ids, sequence_output)

        outputs = self.classify(pooled_output, targets)
        # (loss), prediction_scores

        return outputs


# task_model('secondary_structure', 'transformer')
class ProteinBertForSequenceToSequenceClassification(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_labels = 8
        self.bert = esm1
        self.classify = SequenceToSequenceClassificationHead(
            esm1.args.embed_dim, self.num_labels, ignore_index=-1)

    def forward(self, input_ids, targets=None, policy=None, finetune=True, finetune_emb=True):
        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False

        outputs = self.bert(input_ids, policy, repr_layers=[33])

        sequence_output = outputs['representations'][33]

        outputs = self.classify(sequence_output, targets)
        # (loss), prediction_scores

        return outputs


# task_model('contact_prediction', 'transformer')
class ProteinBertForContactPrediction(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = esm1
        self.predict = PairwiseContactPredictionHead(esm1.args.embed_dim, ignore_index=-1)

    def forward(self, input_ids, protein_length, targets=None, policy=None, finetune=True, finetune_emb=True):
        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False

        outputs = self.bert(input_ids, policy, repr_layers=[33])

        sequence_output = outputs['representations'][33]

        outputs = self.predict(sequence_output, protein_length, targets)
        # (loss), prediction_scores

        return outputs
