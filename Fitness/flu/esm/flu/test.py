import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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
    batch_size = 1024
    
    fluorescence_test_data = dataset.datasets_downstream_tape.FluorescenceDataset(data_path, 'test')

    fluorescence_test_loader = DataLoader(
        fluorescence_test_data, batch_size=batch_size, shuffle=False, collate_fn=fluorescence_test_data.collate_fn
    )

    # downstream_model = model.model_down_tape.ProteinBertForValuePrediction().cuda()
    # #downstream_model.load_state_dict(torch.load('save/downstream/best_flu_pretrain.pt')['model_state_dict'])
    # downstream_model.load_state_dict(torch.load('save/downstream/best_flu_pretrain_MLP.pt')['model_state_dict'])
    downstream_model = model.model_down_tape_scratch.ProteinBertForValuePrediction().cuda()
    #downstream_model.load_state_dict(torch.load('save/downstream/best_flu_pretrain.pt')['model_state_dict'])
    downstream_model.load_state_dict(torch.load('save/downstream/best_flu_scratch.pt')['model_state_dict'])
    
    downstream_model.eval().cuda()
    test_loss = 0
    test_p = 0
    test_step = 0
    test_tic = time.time()
    test_fluorescence_targets=[]
    test_value_prediction=[]
    for idx, batch in enumerate(fluorescence_test_loader):
        fluorescence_inputs = batch['input_ids']
        fluorescence_targets = batch['targets']
        fluorescence_inputs, fluorescence_targets = fluorescence_inputs.cuda(), fluorescence_targets.cuda()
        with torch.no_grad():
            outputs = downstream_model(fluorescence_inputs, targets=fluorescence_targets)
            loss, value_prediction = outputs
            test_fluorescence_targets.extend(fluorescence_targets.detach().cpu().numpy())
            test_value_prediction.extend(value_prediction.detach().cpu().numpy())
        test_loss += loss.mean()
        #test_p += p
        test_step += 1
    p = spearmanr(test_value_prediction,test_fluorescence_targets)
    test_toc = time.time()
    
    print("Step: {} / {} finish. Test Loss: {:.2f}. Test Spearman’s: {:.2f}. Test Time: {:.2f}.".
          format(test_step, len(fluorescence_test_loader), (test_loss / test_step), (p), (test_toc - test_tic)))
