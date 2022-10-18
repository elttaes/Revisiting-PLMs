import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import sys
sys.path.append("..")
from torch.utils.data.dataloader import DataLoader
import model_down
from esm1.data_test import ESMStructuralSplitDataset
import time

if __name__ == '__main__':
    batch_size = 1
    best_acc = 0

    model = model_down.ProteinBertForContactPrediction().cuda()
    model.load_state_dict(torch.load('ESM1b_MLP_cp_ori.pt')['model_state_dict'])
    esm_structural_valid = ESMStructuralSplitDataset(
        split_level='fold', 
        cv_partition='0', 
        split='valid', 
        root_path = os.path.expanduser('/root/ssp/msa_transormer_sujindata/'),
        download=False
    )
    
    valid_loader = DataLoader(dataset=esm_structural_valid,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=esm_structural_valid.collate_fn,
                          num_workers=8,drop_last=True
                          )
    model.eval()
    val_tic = time.time()
    val_loss = 0
    val_p = 0
    val_step = 0
    for idx, batch in enumerate(valid_loader):
        print(idx)
        contact_inputs = batch['input_ids']#.squeeze()
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

    print("\n L/5 Step: {} / {} finish. Validating Loss: {:.5f}. Validating Precision: {:.5f}.\n".
            format(val_step, len(valid_loader), (val_loss / val_step), (val_p / val_step)))