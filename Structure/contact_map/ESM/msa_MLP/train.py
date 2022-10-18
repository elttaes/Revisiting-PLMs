import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import sys
sys.path.append("..")
from torch.utils.data.dataloader import DataLoader
import model_down
from esm1.data import ESMStructuralSplitDataset
import random
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('tmp_msa/MLP')

if __name__ == '__main__':
    epochs = 30
    batch_size = 8
    best_acc = 0
    model = model_down.ProteinBertForContactPrediction().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    esm_structural_train = ESMStructuralSplitDataset(
        split_level='fold', 
        cv_partition='0', 
        split='train', 
        root_path = os.path.expanduser('/root/ssp/msa/'),
        download=False
    )
    esm_structural_valid = ESMStructuralSplitDataset(
        split_level='fold', 
        cv_partition='0', 
        split='valid', 
        root_path = os.path.expanduser('/root/ssp/msa/'),
        download=False
    )
    
    train_loader = DataLoader(dataset=esm_structural_train,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=esm_structural_train.collate_fn,
                          num_workers=8,drop_last=True
                          )
    
    valid_loader = DataLoader(dataset=esm_structural_valid,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=esm_structural_train.collate_fn,
                          num_workers=8,drop_last=True
                          )

    scaler = torch.cuda.amp.GradScaler()
    best_p=0
    train_count=0
    for epoch in range(epochs):
        train_tic = time.time()
        print(f'epoch:{epoch + 1} start!')
        train_loss = 0
        train_p = 0
        train_step = 0
        for idx, batch in enumerate(train_loader):
            train_count+=1
            contact_inputs = batch['input_ids'].squeeze()
            contact_targets = torch.from_numpy(batch['targets'])
            protein_lengths = batch['protein_length']
            contact_inputs, contact_targets, protein_lengths = contact_inputs.cuda(), contact_targets.cuda(), protein_lengths.cuda()
            
            outputs = model(contact_inputs, protein_lengths, targets=contact_targets,finetune=False)
            
            loss_precision, value_prediction = outputs
            loss = loss_precision[0].mean()
            precision = loss_precision[1]['precision_at_l5'].mean()
            
            writer.add_scalar('train_loss', loss, global_step=train_count)
            writer.add_scalar('train_precision', precision, global_step=train_count)
            
            train_loss += loss.item()
            train_p += precision.item()
            train_step += 1

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            #break

        train_toc = time.time()
        model.eval()
        val_tic = time.time()
        val_loss = 0
        val_p = 0
        val_step = 0
        for idx, batch in enumerate(valid_loader):
            contact_inputs = batch['input_ids'].squeeze()
            contact_targets = torch.from_numpy(batch['targets'])
            protein_lengths = batch['protein_length']
            contact_inputs, contact_targets, protein_lengths = contact_inputs.cuda(), contact_targets.cuda(), protein_lengths.cuda()
            with torch.no_grad():
                outputs = model(contact_inputs, protein_lengths, targets=contact_targets,finetune=False)
                loss_precision, value_prediction = outputs
                loss = loss_precision[0].mean()
                precision = loss_precision[1]['precision_at_l5'].mean()

            val_loss += loss.item()
            val_p += precision.item()
            val_step += 1
        writer.add_scalar('test_loss', val_loss / val_step, global_step=epoch)
        writer.add_scalar('test_precision', val_p / val_step, global_step=epoch)
        print("\nStep: {} / {} finish. Validating Loss: {:.2f}. Validating Precision: {:.2f}.\n".
              format(val_step, len(valid_loader), (val_loss / val_step), (val_p / val_step)))
        val_toc = time.time()
        val_loss = val_loss / val_step
        val_p = val_p / val_step
        if val_p > best_p:
            save_data = {"model_state_dict": model.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val Precision is: {:.2f}.".format(val_p))
            torch.save(save_data, "msa_transformer_MLP_best_cp_ori.pt")
            best_p = val_p
        print("\nEpoch: {} / {} finish. Training Loss: {:.2f}. Training Time: {:.2f} s. Validating Loss: {:.2f}. Validating Time: {:.2f} s.\n"
              .format(epoch + 1, epochs, train_loss/train_step, (train_toc - train_tic), val_loss, (val_toc - val_tic)))