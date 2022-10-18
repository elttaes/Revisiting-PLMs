import esm
from .model_utils import ValuePredictionHead
from .model_utils import SequenceClassificationHead
from .model_utils import SequenceToSequenceClassificationHead
from .model_utils import PairwiseContactPredictionHead
import torch.nn as nn
import param_parser_esm1b
import torch

args = param_parser_esm1b.params_parser()
esm1b_alphabet = esm.data.Alphabet.from_architecture(args.arch)
esm1 = esm.model.ProteinBertModel(args, esm1b_alphabet)
model_path = '../save/msa_id/best_esm_ssd_val.pt'
state_dict = torch.load(model_path)
state_esm1b = state_dict['model_state_dict']
esm1.load_state_dict({k.replace("module.", ""): v for k, v in state_esm1b.items()})

esm1_id = esm.model_id.ProteinBertModel_ID(args, esm1b_alphabet)
model_path = '../save/msa_id/best_esm_ssd_id_val.pt'
state_dict = torch.load(model_path)
state_esm1b = state_dict['model_state_dict']
esm1_id.load_state_dict({k.replace("module.", ""): v for k, v in state_esm1b.items()})

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
        attention_mask = 1 - tokens.eq(esm1b_alphabet.padding_idx).type_as(outputs)

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
        self.predict = ValuePredictionHead(args.embed_dim)


    def forward(self, input_ids, targets=None, finetune=True, finetune_emb=True):
        pooler_type = "cls"
        pooler = Pooler(pooler_type)

        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False

        outputs = self.bert(input_ids, repr_layers=[6])

        sequence_output = outputs['representations'][6]

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
            args.embed_dim, self.num_labels)

    def forward(self, input_ids, targets=None, finetune=True, finetune_emb=True):
        pooler_type = "cls"
        pooler = Pooler(pooler_type)

        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False

        outputs = self.bert(input_ids, repr_layers=[6])

        sequence_output = outputs['representations'][6]

        pooled_output = pooler(input_ids, sequence_output)

        outputs = self.classify(pooled_output, targets)
        # (loss), prediction_scores

        return outputs


# task_model('secondary_structure', 'transformer')
class ProteinBertForSequenceToSequenceClassification(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_labels = 8
        # self.num_labels = 3
        self.bert = esm1
        self.classify = SequenceToSequenceClassificationHead(
            args.embed_dim, self.num_labels, ignore_index=-1)

    def forward(self, input_ids, targets=None, finetune=True):
        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False

        outputs = self.bert(input_ids, repr_layers=[6])

        sequence_output = outputs['representations'][6]

        outputs = self.classify(sequence_output, targets)
        # (loss), prediction_scores

        return outputs

# task_model('secondary_structure', 'transformer')
class ProteinBertForSequenceToSequenceClassification_ID(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_labels = 8
        # self.num_labels = 3
        self.bert = esm1_id
        self.classify = SequenceToSequenceClassificationHead(
            args.embed_dim, self.num_labels, ignore_index=-1)

    def forward(self, input_ids, msa_ids, targets=None, finetune=False):
        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False

        outputs = self.bert(input_ids, msa_ids, repr_layers=[6])

        sequence_output = outputs['representations'][6]

        outputs = self.classify(sequence_output, targets)
        # (loss), prediction_scores

        return outputs


# task_model('contact_prediction', 'transformer')
class ProteinBertForContactPrediction(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = esm1
        self.predict = PairwiseContactPredictionHead(args.embed_dim, ignore_index=-1)

    def forward(self, input_ids, protein_length, targets=None, finetune=True):
        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False

        outputs = self.bert(input_ids, repr_layers=[6])

        sequence_output = outputs['representations'][6]

        outputs = self.predict(sequence_output, protein_length, targets)
        # (loss), prediction_scores

        return outputs


# task_model('contact_prediction', 'transformer')
class ProteinBertForContactPrediction_ID(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = esm1_id
        self.predict = PairwiseContactPredictionHead(args.embed_dim, ignore_index=-1)

    def forward(self, input_ids, msa_ids, protein_length, targets=None, finetune=True):
        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False

        outputs = self.bert(input_ids, msa_ids, repr_layers=[6])

        sequence_output = outputs['representations'][6]

        outputs = self.predict(sequence_output, protein_length, targets)
        # (loss), prediction_scores

        return outputs


# task_model('contact_prediction', 'transformer')
# class ProteinBertForContactPrediction(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#
#         self.bert = esm1
#         self.args = args
#         self.prepend_bos = esm1.prepend_bos
#         self.append_eos = esm1.append_eos
#         self.eos_idx = esm1.eos_idx
#         # self.predict = PairwiseContactPredictionHead(args.embed_dim, ignore_index=-1)
#         self.predict = PairwiseContactPredictionHead(
#                             self.args.layers * self.args.attention_heads,
#                             self.prepend_bos,
#                             self.append_eos,
#                             eos_idx=self.eos_idx,
#                             ignore_index=-1
#                         )
#
#     def forward(self, input_ids, protein_length, targets=None, finetune=False):
#         for k, v in self.bert.named_parameters():
#             if not finetune:
#                 v.requires_grad = False
#
#         outputs = self.bert(input_ids, need_head_weights=True)
#
#         attentions = outputs["attentions"]
#
#         outputs = self.predict(input_ids, attentions, protein_length, targets)
#         # (loss), prediction_scores
#
#         return outputs
