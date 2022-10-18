import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
    epochs = 90
    batch_size = 16
    fluorescence_train_data = dataset.datasets_downstream_tape.FluorescenceDataset(data_path, 'train')
    fluorescence_valid_data = dataset.datasets_downstream_tape.FluorescenceDataset(data_path, 'valid')
    fluorescence_test_data = dataset.datasets_downstream_tape.FluorescenceDataset(data_path, 'test')

    fluorescence_train_loader = DataLoader(
        fluorescence_train_data, batch_size=batch_size, shuffle=True, collate_fn=fluorescence_train_data.collate_fn
    )
    fluorescence_valid_loader = DataLoader(
        fluorescence_valid_data, batch_size=batch_size, shuffle=False, collate_fn=fluorescence_valid_data.collate_fn
    )
    fluorescence_test_loader = DataLoader(
        fluorescence_test_data, batch_size=batch_size, shuffle=False, collate_fn=fluorescence_test_data.collate_fn
    )

    downstream_model = model.model_down_tape.ProteinBertForValuePrediction().cuda()
    optimizer = torch.optim.AdamW(downstream_model.parameters(), lr=1e-5)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = 100000
    best_p = 0
    downstream_model.train()
    for epoch in range(epochs):
        train_tic = time.time()
        train_loss = 0
        train_p = 0
        train_step = 0
        for idx, batch in enumerate(fluorescence_train_loader):
            fluorescence_inputs = batch['input_ids']
            #fluorescence_inputs=fluorescence_inputs[:,np.newaxis,:]
            fluorescence_targets = batch['targets']
            fluorescence_inputs, fluorescence_targets = fluorescence_inputs.cuda(), fluorescence_targets.cuda()
            outputs = downstream_model(fluorescence_inputs, targets=fluorescence_targets,finetune=False)
            loss, value_prediction = outputs
            p = spearmanr(fluorescence_targets.detach().cpu().numpy(), value_prediction.detach().cpu().numpy())
            loss = loss.mean()
            train_loss += loss
            train_p += p
            train_step += 1
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if train_step > 0 and train_step % 100 == 0:
                print("Step: {} / {} finish. Training Loss: {:.2f}. Training Spearman’s: {:.2f}."
                      .format(train_step, len(fluorescence_train_loader), (train_loss / train_step), (train_p / train_step)))

        train_toc = time.time()

        downstream_model.eval()
        val_tic = time.time()
        val_loss = 0
        val_p = 0
        val_step = 0
        val_fluorescence_targets=[]
        val_value_predictions=[]
        for idx, batch in enumerate(fluorescence_valid_loader):
            fluorescence_inputs = batch['input_ids']
            fluorescence_targets = batch['targets']
            fluorescence_inputs, fluorescence_targets = fluorescence_inputs.cuda(), fluorescence_targets.cuda()
            with torch.no_grad():
                outputs = downstream_model(fluorescence_inputs, targets=fluorescence_targets)
                loss, value_prediction = outputs
                val_fluorescence_targets.extend(fluorescence_targets.detach().cpu().numpy())
                val_value_predictions.extend(value_prediction.detach().cpu().numpy())
            val_loss += loss.mean()
            #val_p += p
            val_step += 1
        p=spearmanr(val_value_predictions,val_fluorescence_targets)
        print("\nStep: {} / {} finish. Validating Loss: {:.2f}. Validating Spearman’s: {:.2f}.\n".
              format(val_step, len(fluorescence_valid_loader), (val_loss / val_step), p))
        val_toc = time.time()
        val_loss = val_loss / val_step
        val_p = p
        # if val_loss < best_loss:
        if val_p > best_p:
            save_data = {"model_state_dict": downstream_model.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val Spearman’s is: {:.2f}.".format(val_p))
            torch.save(save_data, "./save/downstream/best_flu_pretrain_MLP.pt")
            best_p = val_p
            # best_loss = val_loss
        print("\nEpoch: {} / {} finish. Training Loss: {:.2f}. Training Time: {:.2f} s. Validating Loss: {:.2f}. Validating Time: {:.2f} s.\n"
              .format(epoch + 1, epochs, train_loss/train_step, (train_toc - train_tic), val_loss, (val_toc - val_tic)))

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
