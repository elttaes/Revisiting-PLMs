import torch
import esm
#import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
batch_converter = alphabet.get_batch_converter()


class PairwiseContactPredictionHead(nn.Module):

    def __init__(self, hidden_size: int, ignore_index=-1):
        super().__init__()
        # self.predict = nn.Sequential(
        #     nn.Dropout(), nn.Linear(hidden_size, 2))
        self._ignore_index = ignore_index

    def mse_loss(self, input, target, ignore_index, reduction):
        mask = target == ignore_index
        out = (input[~mask]-target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out

    def forward(self, prediction, sequence_lengths, targets=None):
        #prediction = inputs
        outputs = (prediction,)

        if targets is not None:
            #loss_fct = self.mse_loss()
            contact_loss = self.mse_loss(
                input=prediction, target=targets.type_as(prediction),ignore_index=self._ignore_index,reduction="mean")
            metrics = {'precision_at_l':
                       self.compute_precision_at_l(sequence_lengths, prediction, targets)
                       ,
                       'precision_at_l2':
                       self.compute_precision_at_l2(sequence_lengths, prediction, targets)
                       ,
                       'precision_at_l5':
                       self.compute_precision_at_l5(sequence_lengths, prediction, targets)}
            print(metrics)
            loss_and_metrics = (contact_loss, metrics)
            outputs = (loss_and_metrics,) + outputs

        return outputs

    def compute_precision_at_l(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            #print(valid_mask)
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)
            #probs = torch.softmax(prediction, 3)[:, :, :, 1]
            probs = prediction
            valid_mask = valid_mask.type_as(prediction)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total

    def compute_precision_at_l2(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            #print(valid_mask)
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)
            #probs = torch.softmax(prediction, 3)[:, :, :, 1]
            probs = prediction
            valid_mask = valid_mask.type_as(prediction)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length//2, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total
        
    def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            #print(valid_mask)
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 24).unsqueeze(0)
            #probs = torch.softmax(prediction, 3)[:, :, :, 1]
            probs = prediction
            valid_mask = valid_mask.type_as(prediction)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length//5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total

class ProteinBertForContactPrediction(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = model
        self.predict = PairwiseContactPredictionHead(model.args.embed_dim, ignore_index=-1)

    @torch.cuda.amp.autocast()
    def forward(self, input_ids, protein_length, targets=None, finetune=True, finetune_emb=True):
        for k, v in self.bert.named_parameters():
            if k not in ['contact_head.regression.weight','contact_head.regression.bias']:
                v.requires_grad = False
        # for k, v in self.bert.named_parameters():
        #     if not finetune:
        #         v.requires_grad = False
        #     elif not finetune_emb and 'embed_tokens.weight' in k:
        #         v.requires_grad = False
        #     elif not finetune_emb and 'embed_positions.weight' in k:
        #         v.requires_grad = False
        outputs = self.bert.predict_contacts(input_ids)
        #outputs = self.bert(input_ids, repr_layers=[33])

        #sequence_output = outputs['representations'][33]

        outputs = self.predict(outputs, protein_length, targets)
        # (loss), prediction_scores

        return outputs