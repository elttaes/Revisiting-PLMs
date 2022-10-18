import time
import torch
import sys

sys.path.append("..")
import esm
import os
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from model import model_down_random
import dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

if __name__ == '__main__':
    epochs = 30
    batch_size = 16
    best_acc = 0

    train_data = dataset.FinetuneFunction('train')
    valid_data = dataset.FinetuneFunction('val')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=valid_data.collate_fn)

    model = model_down_random.ProteinBertForSequenceClassification().cuda()
    batch_converter = model_down_random.alphabet.get_batch_converter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        print(f'epoch:{epoch + 1} start!')
        train_loss = 0
        train_acc = 0
        train_step = 0
        # with tqdm(total=len(train_dataloader), desc=f'epoch {epoch}') as pbar:
        for idx, batch in enumerate(train_dataloader):
            ids = batch['ids']
            seqs = batch['seqs']
            targets = batch['labels']
            data = []
            for id, seq in zip(ids, seqs):
                data.append((id, seq))
            # print(data)
            batch_labels, batch_strs, inputs = batch_converter(data)
            inputs, targets = torch.tensor(inputs).cuda(), torch.tensor(targets).cuda()
            outputs = model(inputs, targets=targets)
            loss_acc, value_prediction = outputs
            loss = loss_acc[0]
            acc = loss_acc[1]['accuracy']

            train_loss += loss.item()
            train_acc += acc.item()
            train_step += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # pbar.update(1)
            # pbar.set_postfix(loss=loss, acc=acc)

            if train_step > 0 and train_step % 100 == 0:
                print("Step: {} / {} finish. Training Loss: {:.8f}. Training Accuracy: {:.8f}."
                      .format(train_step, len(train_dataloader), (train_loss / train_step),
                              (train_acc / train_step)))

        val_loss = 0
        val_acc = 0
        val_step = 0
        for idx, batch in enumerate(valid_dataloader):
            ids = batch['ids']
            seqs = batch['seqs']
            targets = batch['labels']
            data = []
            for id, seq in zip(ids, seqs):
                data.append((id, seq))
            # print(data)
            batch_labels, batch_strs, inputs = batch_converter(data)
            inputs, targets = torch.tensor(inputs).cuda(), torch.tensor(targets).cuda()

            with torch.no_grad():
                outputs = model(inputs, targets=targets)
                loss_acc, value_prediction = outputs
                loss = loss_acc[0]
                acc = loss_acc[1]['accuracy']

            val_loss += loss.item()
            val_acc += acc.item()
            val_step += 1

        print("\nStep: {} / {} finish. Validating Loss: {:.8f}. Validating Accuracy: {:.8f}.\n".
              format(val_step, len(valid_dataloader), (val_loss / val_step), (val_acc / val_step)))

        val_loss = val_loss / val_step
        val_acc = val_acc / val_step
        # if val_loss < best_loss:
        if val_acc > best_acc:
            save_data = {"model_state_dict": model.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val Accuracy is: {:.8f}.".format(val_acc))
            torch.save(save_data, "best_model_val.pt")
            best_acc = val_acc
            # best_loss = val_loss
        print(
            "\nEpoch: {} / {} finish. Training Loss: {:.8f}.  Validating Loss: {:.8f}.\n"
                .format(epoch + 1, epochs, train_loss / train_step, val_loss))
