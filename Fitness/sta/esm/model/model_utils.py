import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
from typing import Optional

def symmetrize(x):
    "Make layer symmetric in final two dimensions, used for contact prediction."
    return x + x.transpose(-1, -2)

def apc(x):
    "Perform average product correct, used for contact prediction."
    a1 = x.sum(-1, keepdims=True)
    a2 = x.sum(-2, keepdims=True)
    a12 = x.sum((-1, -2), keepdims=True)

    avg = a1 * a2
    avg.div_(a12)  # in-place to reduce memory
    normalized = x - avg
    return normalized

def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()

def _init_weights(module):
    """ Initialize the weights """
    if isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()

class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
            )
        self.apply(_init_weights)

    def forward(self, x):
        return self.main(x)

# class SimpleMLP(nn.Module):

#     def __init__(self,
#                  in_dim: int,
#                  hid_dim: int,
#                  out_dim: int,
#                  dropout: float = 0.):
#         super().__init__()
#         self.main = nn.Sequential(
#             weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
#             nn.ReLU(),
#             nn.Dropout(dropout, inplace=True),
#             weight_norm(nn.Linear(hid_dim, out_dim), dim=None))
#         self.apply(_init_weights)

#     def forward(self, x):
#         return self.main(x)


class SimpleConv(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_dim),  # Added this
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x


class Accuracy(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target, self.ignore_index)


class ValuePredictionHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.):
        super().__init__()
        self.value_prediction = SimpleMLP(hidden_size, 1280, 1, dropout)

    def forward(self, pooled_output, targets=None):
        value_pred = self.value_prediction(pooled_output)
        outputs = (value_pred,)

        if targets is not None:
            loss_fct = nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            outputs = (value_pred_loss,) + outputs
        return outputs  # (loss), value_prediction


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleMLP(hidden_size, 1280, num_labels)

    def forward(self, pooled_output, targets=None):
        logits = self.classify(pooled_output)
        outputs = (logits,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, targets)
            metrics = {'accuracy': accuracy(logits, targets)}
            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs

        return outputs  # (loss), logits


class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 ignore_index: int = -100):
        super().__init__()
        self.classify = SimpleConv(
            hidden_size, 1280, num_labels)
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
        return outputs  # (loss), sequence_logits


class PairwiseContactPredictionHead(nn.Module):

    def __init__(self, hidden_size: int, ignore_index=-100):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Dropout(), nn.Linear(2 * hidden_size, 2))
        self._ignore_index = ignore_index

    def forward(self, inputs, sequence_lengths, targets=None):
        prod = inputs[:, :, None, :] * inputs[:, None, :, :]
        diff = inputs[:, :, None, :] - inputs[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        # prediction = apc(prediction)
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove start/stop tokens
        outputs = (prediction,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(
                prediction.view(-1, 2), targets.view(-1))
            metrics = {'precision_at_l5':
                       self.compute_precision_at_l5(sequence_lengths, prediction, targets)}
            loss_and_metrics = (contact_loss, metrics)
            outputs = (loss_and_metrics,) + outputs

        return outputs

    def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total


# class PairwiseContactPredictionHead(nn.Module):
#
#     def __init__(
#             self,
#             in_features: int,
#             prepend_bos: bool,
#             append_eos: bool,
#             bias=True,
#             eos_idx: Optional[int] = None,
#             ignore_index=-1
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.prepend_bos = prepend_bos
#         self.append_eos = append_eos
#         self._ignore_index = ignore_index
#         if append_eos and eos_idx is None:
#             raise ValueError(
#                 "Using an alphabet with eos token, but no eos token was passed in."
#             )
#         self.eos_idx = eos_idx
#         self.regression = nn.Linear(in_features, 2, bias)
#         self.activation = nn.Sigmoid()
#
#     def forward(self, tokens, attentions, sequence_lengths, targets=None):
#         if self.append_eos:
#             eos_mask = tokens.ne(self.eos_idx).to(attentions)
#             eos_mask = eos_mask.unsqueeze(1) * eos_mask.unsqueeze(2)
#             attentions = attentions * eos_mask[:, None, None, :, :]
#             attentions = attentions[..., :-1, :-1]
#         # remove cls token attentions
#         if self.prepend_bos:
#             attentions = attentions[..., 1:, 1:]
#         batch_size, layers, heads, seqlen, _ = attentions.size()
#         attentions = attentions.view(batch_size, layers * heads, seqlen, seqlen)
#
#         # features: B x C x T x T
#         attentions = attentions.to(next(self.parameters()))  # attentions always float32, may need to convert to float16
#         attentions = apc(symmetrize(attentions))
#         attentions = attentions.permute(0, 2, 3, 1)
#         # prediction = self.regression(attentions)
#         prediction = self.regression(attentions).squeeze(3)
#         # prediction = self.activation(self.regression(attentions).squeeze(3))
#         # prediction_inverse = 1 - prediction_forward
#         # prediction = torch.cat((prediction_forward.unsqueeze(-1), prediction_inverse.unsqueeze(-1)), dim=-1)
#         outputs = (prediction,)
#
#         if targets is not None:
#             # loss_fct = nn.BCELoss()
#             loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
#             contact_loss = loss_fct(
#                 prediction.view(-1, 2), targets.view(-1))
#             metrics = {'precision_at_l5':
#                        self.compute_precision_at_l5(sequence_lengths, prediction, targets)}
#             loss_and_metrics = (contact_loss, metrics)
#             outputs = (loss_and_metrics,) + outputs
#
#         return outputs
#
#     def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
#         with torch.no_grad():
#             valid_mask = labels != self._ignore_index
#             seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
#             x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
#             valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
#             # probs = prediction[:, :, :, 0]
#             probs = F.softmax(prediction, 3)[:, :, :, 1]
#             valid_mask = valid_mask.type_as(probs)
#             correct = 0
#             total = 0
#             for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
#                 masked_prob = (prob * mask).view(-1)
#                 most_likely = masked_prob.topk(length // 5, sorted=False)
#                 selected = label.view(-1).gather(0, most_likely.indices)
#                 correct += selected.sum().float()
#                 total += selected.numel()
#             return correct / total
