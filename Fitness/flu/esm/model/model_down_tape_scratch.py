import esm
from .model_utils import ValuePredictionHead,SequenceClassificationHead
import torch.nn as nn
import torch
import msa_transformer_parser
#esm1, esm1_alphabet = esm.pretrained.load_model_and_alphabet("./pretrained_models/esm1b_t33_650M_UR50S.pt")
args = msa_transformer_parser.params_parser()
msa_transformer_alphabet = esm.data.Alphabet.from_architecture(args.arch)
msa_transformer = esm.model.MSATransformer(args, msa_transformer_alphabet)
#batch_converter = alphabet.get_batch_converter()

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
        attention_mask = 1 - tokens.eq(msa_transformer_alphabet.padding_idx).type_as(outputs)

        if self.pooler_type in ['cls']:
            return last_hidden[:,0,0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError

# task_model('fluorescence', 'transformer')
# task_model('stability', 'transformer')
class ProteinBertForValuePrediction(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = msa_transformer
        self.predict = ValuePredictionHead(msa_transformer.args.embed_dim)

    @torch.cuda.amp.autocast()
    def forward(self, input_ids, targets=None, finetune=True, finetune_emb=True):
        pooler_type = "cls"
        pooler = Pooler(pooler_type)

        # for k, v in self.bert.named_parameters():
        #     if not finetune:
        #         v.requires_grad = False
        #     elif not finetune_emb and 'embed_tokens.weight' in k:
        #         v.requires_grad = False
        #     elif not finetune_emb and 'embed_positions.weight' in k:
        #         v.requires_grad = False

        outputs = self.bert(input_ids, repr_layers=[12])

        sequence_output = outputs['representations'][12]

        pooled_output = pooler(input_ids, sequence_output)

        outputs = self.predict(pooled_output, targets)
        # (loss), prediction_scores

        return outputs

class ProteinBertForSequenceClassification(nn.Module):

    def __init__(self):
        super().__init__()

        self.num_labels = 19
        self.bert = msa_transformer
        self.classify = SequenceClassificationHead(
            msa_transformer.args.embed_dim, self.num_labels)

    @torch.cuda.amp.autocast()
    def forward(self, input_ids, targets=None, finetune=True, finetune_emb=True):
        pooler_type = "cls"
        pooler = Pooler(pooler_type)

        # for k, v in self.bert.named_parameters():
        #     #print(k)
        #     if not finetune:
        #         v.requires_grad = False
        #     elif not finetune_emb and 'embed_tokens.weight' in k:
        #         v.requires_grad = False
        #     elif not finetune_emb and 'embed_positions.weight' in k:
        #         v.requires_grad = False

        outputs = self.bert(input_ids, repr_layers=[12])

        sequence_output = outputs['representations'][12]

        pooled_output = pooler(input_ids, sequence_output)

        outputs = self.classify(pooled_output, targets)
        # (loss), prediction_scores

        return outputs