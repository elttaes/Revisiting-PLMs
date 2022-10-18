import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import model
import torch
import os
import dataset_drug
import time
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import numpy as np
#import scipy.stats
#from apex import amp
#from apex.parallel import convert_syncbn_model

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
    epochs = 60
    batch_size = 16
    
    a3m_dir='./train'
    filenames = [
        os.path.join(a3m_dir,name) for name in os.listdir(a3m_dir)
        if os.path.splitext(name)[-1] == '.a3m'
    ]  #选择指定目录下的.png图片
    drug_train_data = dataset_drug.Dataset(filenames)
    drug_train_loader = DataLoader(
        drug_train_data, batch_size=batch_size, shuffle=True, collate_fn=drug_train_data.collate_fn
    )
    
    a3m_dir_test='./test'
    filenames = [
        os.path.join(a3m_dir_test,name) for name in os.listdir(a3m_dir_test)
        if os.path.splitext(name)[-1] == '.a3m'
    ]  #选择指定目录下的.png图片
    drug_test_data = dataset_drug.Dataset(filenames)
    drug_test_loader = DataLoader(
        drug_test_data, batch_size=batch_size, shuffle=True, collate_fn=drug_test_data.collate_fn
    )
    
    downstream_model = model.model_down_tape.ProteinBertForSequenceClassification().cuda()
    downstream_model=torch.nn.DataParallel(downstream_model)
    optimizer = torch.optim.AdamW(downstream_model.parameters(), lr=1e-5)

    #downstream_model = convert_syncbn_model(downstream_model)
    #downstream_model, optimizer = amp.initialize(downstream_model, optimizer, opt_level='O0')
    scaler = torch.cuda.amp.GradScaler()
    best_loss = 100000
    best_acc = 0
    downstream_model.train()
    for epoch in range(epochs):
        train_tic = time.time()
        train_loss = 0
        train_acc = 0
        train_step = 0
        for idx, batch in enumerate(drug_train_loader):
            drug_inputs = batch[0]
            drug_targets = batch[1]
            drug_inputs, drug_targets = drug_inputs.cuda(), drug_targets.cuda()
            outputs = downstream_model(drug_inputs, targets=drug_targets)
            loss_acc, value_prediction = outputs
            loss = loss_acc[0].mean()
            acc = loss_acc[1]['accuracy'].mean()

            train_loss += loss.item()
            train_acc += acc.item()
            train_step += 1

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if train_step > 0 and train_step % 100 == 0:
                print("Step: {} / {} finish. Training Loss: {:.2f}. Training Accuracy: {:.2f}."
                      .format(train_step, len(drug_train_loader), (train_loss / train_step), (train_acc / train_step)))

        train_toc = time.time()

        downstream_model.eval()
        val_tic = time.time()
        val_loss = 0
        val_acc = 0
        val_step = 0
        for idx, batch in enumerate(drug_test_loader):
            drug_inputs = batch[0]
            drug_targets = batch[1]
            drug_inputs, drug_targets = drug_inputs.cuda(), drug_targets.cuda()
            with torch.no_grad():
                outputs = downstream_model(drug_inputs, targets=drug_targets)
                loss_acc, value_prediction = outputs
                loss = loss_acc[0].mean()
                acc = loss_acc[1]['accuracy'].mean()

            val_loss += loss.item()
            val_acc += acc.item()
            val_step += 1

        print("\nStep: {} / {} finish. Validating Loss: {:.4f}. Validating Accuracy: {:.4f}.\n".
              format(val_step, len(drug_test_loader), (val_loss / val_step), (val_acc / val_step)))
        val_toc = time.time()
        val_loss = val_loss / val_step
        val_acc = val_acc / val_step

        if val_acc > best_acc:
            save_data = {"model_state_dict": downstream_model.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val Accuracy is: {:.4f}.".format(val_acc))
            torch.save(save_data, "./save/downstream/best_drug_ori.pt")
            best_acc = val_acc
            # best_loss = val_loss
        print("\nEpoch: {} / {} finish. Training Loss: {:.2f}. Training Time: {:.2f} s. Validating Loss: {:.2f}. Validating Time: {:.2f} s.\n"
              .format(epoch + 1, epochs, train_loss/train_step, (train_toc - train_tic), val_loss, (val_toc - val_tic)))