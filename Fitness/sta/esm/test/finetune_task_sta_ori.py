import os
from turtle import down
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import esm
import model
import torch
import os
import dataset
import time
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import numpy as np
import scipy.stats
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(30)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def mean_squared_error(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.square(target_array - prediction_array))

def mean_absolute_error(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return np.mean(np.abs(target_array - prediction_array))

def spearmanr(target, prediction):
    target_array = np.asarray(target)
    prediction_array = np.asarray(prediction)
    return scipy.stats.mstats.spearmanr(target_array, prediction_array).correlation

def accuracy(target, prediction):
    if isinstance(target[0], int):
        # non-sequence case
        return np.mean(np.asarray(target) == np.asarray(prediction).argmax(-1))
    else:
        correct = 0
        total = 0
        for label, score in zip(target, prediction):
            label_array = np.asarray(label)
            pred_array = np.asarray(score).argmax(-1)
            mask = label_array != -1
            is_correct = label_array[mask] == pred_array[mask]
            correct += is_correct.sum()
            total += is_correct.size
        return correct / total

if __name__ == '__main__':
    data_path = './data/downstream'
    epochs = 30
    batch_size = 128

    stability_test_data = dataset.datasets_downstream_tape.StabilityDataset(data_path, 'test')


    stability_test_loader = DataLoader(
        stability_test_data, batch_size=batch_size, shuffle=False, collate_fn=stability_test_data.collate_fn
    )

    downstream_model = model.model_down_tape.ProteinBertForValuePrediction().cuda()
    downstream_model.load_state_dict(torch.load('save/downstream/best_flu_pretrain.pt')['model_state_dict'])

    downstream_model.eval().cuda()
    test_loss = 0
    test_p = 0
    test_step = 0
    test_tic = time.time()
    for idx, batch in enumerate(stability_test_loader):
        stability_inputs = batch['input_ids']
        stability_targets = batch['targets']
        stability_inputs, stability_targets = stability_inputs.cuda(), stability_targets.cuda()
        with torch.no_grad():
            outputs = downstream_model(stability_inputs, targets=stability_targets)
            loss, value_prediction = outputs
            p = spearmanr(stability_targets.detach().cpu().numpy(), value_prediction.detach().cpu().numpy())
    
        test_loss += loss.item()
        test_p += p
        test_step += 1
    
    test_toc = time.time()
    
    print("Step: {} / {} finish. Test Loss: {:.2f}. Test Spearman’s: {:.2f}. Test Time: {:.2f}.".
          format(test_step, len(stability_test_loader), (test_loss / test_step), (test_p / test_step),
                 (test_toc - test_tic)))
