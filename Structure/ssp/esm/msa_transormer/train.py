import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
import sys
sys.path.append("..")
from torch.utils.data.dataloader import DataLoader
import model_down
from esm.data import ESMStructuralSplitDataset

for split_level in ['family', 'superfamily', 'fold']:
    for cv_partition in ['0', '1', '2', '3', '4']:
        esm_structural_train = ESMStructuralSplitDataset(
            split_level=split_level, 
            cv_partition=cv_partition, 
            split='train', 
            root_path = os.path.expanduser('.'),
            download=False
        )
        esm_structural_valid = ESMStructuralSplitDataset(
            split_level=split_level, 
            cv_partition=cv_partition, 
            split='valid', 
            root_path = os.path.expanduser('.'),
            download=False
        )

if __name__ == '__main__':
    epochs = 60
    batch_size = 16
    best_acc = 0
    model = model_down.ProteinBertForSequence2Sequence().cuda()
    model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    esm_structural_train = ESMStructuralSplitDataset(
        split_level='fold', 
        cv_partition='0', 
        split='train', 
        root_path = os.path.expanduser('.'),
        download=False
    )
    esm_structural_valid = ESMStructuralSplitDataset(
        split_level='fold', 
        cv_partition='0', 
        split='valid', 
        root_path = os.path.expanduser('.'),
        download=False
    )
    
    train_loader = DataLoader(dataset=esm_structural_train,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=esm_structural_train.__collate_fn__
                          )
    valid_loader = DataLoader(dataset=esm_structural_valid,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=esm_structural_train.__collate_fn__
                          )
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        print(f'epoch:{epoch + 1} start!')
        train_loss = 0
        train_acc = 0
        train_step = 0
        for idx, batch in enumerate(train_loader):
            seqs = torch.squeeze(batch['input_ids'])
            targets = batch['targets']
            inputs, targets = torch.as_tensor(seqs).cuda(), torch.as_tensor(targets).cuda()
            outputs = model(inputs, targets=targets)
            loss_acc, value_prediction = outputs
            loss = loss_acc[0]
            acc = loss_acc[1]['accuracy']
            loss=torch.mean(loss)
            acc=torch.mean(acc)

            train_loss += loss
            train_acc += acc
            train_step += 1

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            if train_step > 0 and train_step % 100 == 0:
                print("Step: {} / {} finish. Training Loss: {:.8f}. Training Accuracy: {:.8f}."
                      .format(train_step, len(train_loader), (train_loss / train_step),
                              (train_acc / train_step)))

        val_loss = 0
        val_acc = 0
        val_step = 0
        for idx, batch in enumerate(valid_loader):
            seqs = torch.squeeze(batch['input_ids'])
            targets = batch['targets']
            inputs, targets = torch.tensor(seqs).cuda(), torch.tensor(targets).cuda()
            with torch.no_grad():
                outputs = model(inputs, targets=targets)
                loss_acc, value_prediction = outputs
                loss = loss_acc[0]
                acc = loss_acc[1]['accuracy']
            loss=torch.mean(loss)
            acc=torch.mean(acc)
            val_loss += loss.item()
            val_acc += acc.item()
            val_step += 1

        print("\nStep: {} / {} finish. Validating Loss: {:.8f}. Validating Accuracy: {:.8f}.\n".
              format(val_step, len(valid_loader), (val_loss / val_step), (val_acc / val_step)))

        val_loss = val_loss / val_step
        val_acc = val_acc / val_step
        if val_acc > best_acc:
            save_data = {"model_state_dict": model.module.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val Accuracy is: {:.8f}.".format(val_acc))
            torch.save(save_data, "best_model_val_pretrain.pt")
            best_acc = val_acc
        print(
            "\nEpoch: {} / {} finish. Training Loss: {:.8f}.  Validating Loss: {:.8f}.\n"
                .format(epoch + 1, epochs, train_loss / train_step, val_loss))