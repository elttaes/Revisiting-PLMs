import time
import torch
import sys
sys.path.append("..")
import esm
import os
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader
import torch.utils.data as Data
import model_down_random as model_down
from torch.utils.data.dataset import TensorDataset
from Bio import SeqIO
import itertools
from typing import List, Tuple,Any
import string
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str) -> Tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    a=[(record.description, remove_insertions(str(record.seq))[:256])
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]
    return a
    


if __name__ == '__main__':
    epochs = 60
    batch_size = 8
    best_acc = 0
    msa_alphabet = esm.data.Alphabet.from_architecture('msa_transformer')
    msa_batch_converter = msa_alphabet.get_batch_converter()
    # Load ESM-1b model
    # model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
    # msa_batch_converter = alphabet.get_batch_converter()
    class Dataset(torch.utils.data.Dataset):
        'Characterizes a dataset for PyTorch'

        def __init__(self, list):
            'Initialization'
            self.list = list

        def __len__(self):
            'Denotes the total number of samples'
            return len(self.list)

        def __getitem__(self, index):
            ID = self.list[index]
            return ID

        def collate_fn(self, batch: List[Tuple[Any, ...]]):
            #print(batch)
            msa_data = []
            label = []
            for i in range(len(batch)):
                msa_data.append(read_msa(os.path.join(batch[i]), 56))
            msa_batch_label, msa_batch_str, msa_batch_token = msa_batch_converter(msa_data)
            for i in range(len(batch)):
                batch_i=os.path.split(batch[i])
                batch_i_name=batch_i[1][:-4]
                batch_i_path=os.path.join('label', batch_i_name)
                with open(batch_i_path,"r")as f:
                    label_i=f.readline()
                label.append(int(label_i))
            return msa_batch_token,label

        def collate_fn_test(self, batch: List[Tuple[Any, ...]]):
            #print(batch)
            msa_data = []
            label = []
            for i in range(len(batch)):
                msa_data.append(read_msa(os.path.join(batch[i]), 56))
            msa_batch_label, msa_batch_str, msa_batch_token = msa_batch_converter(msa_data)
            for i in range(len(batch)):
                batch_i=os.path.split(batch[i])
                batch_i_name=batch_i[1][:-4]
                batch_i_path=os.path.join('label_test', batch_i_name)
                with open(batch_i_path,"r")as f:
                    label_i=f.readline()
                label.append(int(label_i))
            return msa_batch_token,label
        
    a3m_dir='a3m'
    filenames = [
        os.path.join(a3m_dir,name) for name in os.listdir(a3m_dir)
        if os.path.splitext(name)[-1] == '.a3m'
    ]  #选择指定目录下的.png图片
    dataset = Dataset(filenames)
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            pin_memory=True,
                                            num_workers=8,
                                            collate_fn=dataset.collate_fn)

    test_dir='a3m_test'
    testnames = [
        os.path.join(test_dir,name) for name in os.listdir(test_dir)
        if os.path.splitext(name)[-1] == '.a3m'
    ]  #选择指定目录下的.png图片
    test_dataset = Dataset(testnames)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            drop_last=True,
                                            pin_memory=True,
                                            num_workers=8,
                                            collate_fn=test_dataset.collate_fn_test)

    model = model_down.ProteinBertForSequenceClassification().cuda()
    model = torch.nn.DataParallel(model)
    #batch_converter = model_down.alphabet.get_batch_converter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    #optimizer = torch.nn.DataParallel(optimizer)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        print(f'epoch:{epoch + 1} start!')
        train_loss = 0
        train_acc = 0
        train_step = 0
        for idx, batch in enumerate(train_dataloader):
            seqs = batch[0]
            targets = batch[1]
            inputs, targets = torch.as_tensor(seqs).cuda(), torch.as_tensor(targets).cuda()
            outputs = model(inputs, targets=targets)
            loss_acc, value_prediction = outputs
            loss = loss_acc[0]
            acc = loss_acc[1]['accuracy']
            loss=torch.mean(loss)
            acc=torch.mean(acc)

            train_loss += loss.item()
            train_acc += acc.item()
            train_step += 1

            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if train_step > 0 and train_step % 100 == 0:
                print("Step: {} / {} finish. Training Loss: {:.8f}. Training Accuracy: {:.8f}."
                      .format(train_step, len(train_dataloader), (train_loss / train_step),
                              (train_acc / train_step)))

        val_loss = 0
        val_acc = 0
        val_step = 0
        for idx, batch in enumerate(test_dataloader):
            seqs = batch[0]
            targets = batch[1]
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
              format(val_step, len(test_dataloader), (val_loss / val_step), (val_acc / val_step)))

        val_loss = val_loss / val_step
        val_acc = val_acc / val_step
        # if val_loss < best_loss:
        if val_acc > best_acc:
            save_data = {"model_state_dict": model.module.state_dict(),
                         "optim_state_dict": optimizer.state_dict(),
                         "epoch": epoch}
            print("Save model! Best val Accuracy is: {:.8f}.".format(val_acc))
            torch.save(save_data, "best_model_val_random.pt")
            best_acc = val_acc
            # best_loss = val_loss
        print(
            "\nEpoch: {} / {} finish. Training Loss: {:.8f}.  Validating Loss: {:.8f}.\n"
                .format(epoch + 1, epochs, train_loss / train_step, val_loss))