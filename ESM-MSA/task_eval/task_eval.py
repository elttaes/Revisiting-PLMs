import json
import os
import random
import re
import pandas as pd
# import tape
import torch
from tqdm import tqdm
# import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from Bio import SeqIO


def rename_by_id(path, suffix):
    for i in range(10000):
        file_path = f"{path}/query_{i}.{suffix}"
        if not os.path.exists(file_path):
            continue
        with open(file_path, 'r') as f:
            id = f.readline()[1:-1].split(' ')[0].replace(":", '')

        os.system(f"mv -f {file_path} {path}/{id}_twintowers.{suffix}")


def calc_neff(path):
    with open(path, 'r') as f:
        data = f.readlines()
        ids = np.array([[ord(sign) for sign in data[i][:-1]] for i in range(1, len(data), 2)])
        masks = ids == ord('-')
        length = ids.shape[1]

    # ignore gap of any column between any two sequences
    temp = masks
    masks = np.expand_dims(masks, 1).repeat(masks.shape[0], axis=1)
    masks = masks | temp

    # calculate sequence identity between any two sequences
    temp = ids
    ids = np.expand_dims(ids, 1).repeat(ids.shape[0], axis=1)
    ids = ids - temp

    # invert the effective length of the sequence to avoid divided by 0
    valid_len = (~masks).sum(axis=-1)
    invert_len = np.where(valid_len != 0, 1 / valid_len, 0)
    
    identity_matrix = ((ids == 0) * ~masks).sum(axis=-1) * invert_len
    denominator = (identity_matrix >= 0.8).sum(axis=1)
    denominator[denominator == 0] = 1
    neff = (1 / denominator).sum() / np.sqrt(length)
    return neff


def parse_a3m(path, out):
    pattern = r'[a-z]+'
    with open(out, 'w') as f:
        for record in SeqIO.parse(path, 'fasta'):
            f.write(f">{record.description}\n{re.sub(pattern, '', str(record.seq))}\n")
            

def create_label(item):
    valid_mask = np.array(item['valid_mask'])
    contact_map = np.less(squareform(pdist(item['tertiary'])), 8.0).astype(np.int64)
    
    yind, xind = np.indices(contact_map.shape)
    invalid_mask = ~(valid_mask[:, None] & valid_mask[None, :])
    invalid_mask |= np.abs(yind - xind) < 6
    contact_map[invalid_mask] = -1
    
    return contact_map


def eval_data(path):
    data = json.load(open(path, 'r'))
    
    hhblits_total_correct = 0
    hhblits_total = 0
    twintowers_total_correct = 0
    twintowers_total = 0
    mm_total_correct = 0
    mm_total = 0
    for seq in data:
        id = seq['id']
        if len(seq['primary']) > 200:
            continue
        
        # if id != "40#1X51_1_A":
        #     continue

        # id = 'query_25'
        # print(id)
        tt_path = f"twintowers/{id}_twintowers_mtx.npy"
        tt_no_gap = f"twintowers/{id}_twintowers_nogap.npy"
        
        hh_path = f"hhblits/{id}_mtx.npy"
        hh_nogap = f"hhblits/{id}_nogap.npy"
        
        if os.path.exists(hh_path) and os.path.exists(tt_path):
            label = create_label(seq)
            tt_pred = squareform(np.load(tt_path))
            hh_pred = squareform(np.load(hh_path))
            
            # if np.isnan(tt_pred).sum() > 0 or np.isnan(hh_pred).sum() > 0:
            #     continue

            print("*" * 100)
            print(id)
            print(len(seq['primary']))
            tt_no_gap = np.load(tt_no_gap)
            label = label[tt_no_gap][:, tt_no_gap]
            
            print('twintowers:', end=' ')
            correct, num = contact_eval(tt_pred, label, divisor=5, dist_tpye='long')
            twintowers_total_correct += correct
            twintowers_total += num

            label = create_label(seq)
    
            hh_no_gap = np.load(hh_nogap)
            label = label[hh_no_gap][:, hh_no_gap]

            print('hhblits:', end=' ')
            correct, num = contact_eval(hh_pred, label, divisor=5, dist_tpye='long')
            hhblits_total_correct += correct
            hhblits_total += num
            
            # info_dict = tokenizer.batch_encode([seq['primary']], padding=True)
            # query_ids = info_dict['input_ids'].to(device)
            # lengths = info_dict['lengths']
            #
            # with torch.no_grad():
            #     pred = model.esm1b.predict_contacts(query_ids, lengths)[0].to('cpu').numpy()
            #     pred = pred[hh_no_gap][:, hh_no_gap]
            #
            #     label = create_label(seq)
            #     label = label[hh_no_gap][:, hh_no_gap]
            #
            # correct, num = contact_eval(pred, label, divisor=2, dist_tpye='long')
            # mm_total_correct += correct
            # mm_total += num

    return hhblits_total_correct, hhblits_total, twintowers_total_correct, twintowers_total, mm_total_correct, mm_total


def contact_eval(pred, label, divisor=5, dist_tpye='long'):
    length = int(pred.shape[0] / divisor)
    assert dist_tpye in ['short', 'medium', 'long', 'all']
    if dist_tpye == 'long':
        dist_left = 24
        dist_right = 10000
    elif dist_tpye == 'medium':
        dist_left = 12
        dist_right = 23
    elif dist_tpye == 'short':
        dist_left = 6
        dist_right = 11
    elif dist_tpye == 'all':
        dist_left = 0
        dist_right = 10000
    
    yind, xind = np.indices(pred.shape)
    pred[(yind - xind < dist_left) | (yind - xind > dist_right) | (label == -1)] = -1
    pred_1d = pred.flatten()
    # pred_1d[np.isnan(pred_1d)] = 100
    order = np.argsort(-pred_1d)
    
    x, y = np.unravel_index(order, pred.shape)
    x = x[:length]
    y = y[:length]
    label = label[x, y]
    label = label[label != -1]
    if len(label) > 0:
        correct = int(label.sum())
        print(f"{correct}/{len(label)}  acc:{correct/len(label)*100:.2f}%")
        
        return correct, len(label)
    
    else:
        return 0, 0


if __name__ == '__main__':
    # from model import MSARetrieveModel
    # import torch
    #
    # device = 'cuda'
    # model = MSARetrieveModel()
    # path = '../PretrainedModels/esm1b_t33_650M_UR50S.pt'
    # model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    # model.to(device)
    #
    # from LoadingData.tokenizer import Tokenizer
    # tokenizer = Tokenizer()
    #
    # types = ['test', 'valid']
    # # path = "/sujin/dataset/contact/proteinnet/proteinnet_{}.json"
    # path = "proteinnet_{}.json"
    # hh_total_correct = 0
    # hh_total = 0
    # tt_total_correct = 0
    # tt_total = 0
    # model_total_correct = 0
    # model_total = 0
    #
    # for t in types:
    #     hh_correct, hh_num, tt_correct, tt_num, model_correct, model_num = eval_data(path.format(t))
    #     hh_total_correct += hh_correct
    #     hh_total += hh_num
    #     tt_total_correct += tt_correct
    #     tt_total += tt_num
    #     model_total_correct += model_correct
    #     model_total += model_num
    #
    # print(f"hhblits: {hh_total_correct}/{hh_total}  acc:{hh_total_correct / hh_total * 100:.2f}%", end=' ')
    # print(f"twintowers: {tt_total_correct}/{tt_total}  acc:{tt_total_correct / tt_total * 100:.2f}%", end=' ')
    # print(f"model: {model_total_correct}/{model_total}  acc:{model_total_correct / model_total * 100:.2f}%", end=' ')
    
    # for record in SeqIO.parse("CASP14Sequence.faa", 'fasta'):
    #     id = record.id
    #     with open(f"af2_hhblits/{id}_hhblits.fasta", 'w') as f:
    #         f.write(f">{record.description}\n{str(record.seq)}\n")
    
    # path = "/sujin/TwinTowers/task_eval/af2_twintowers/faa"
    # with open(f"{path}/test_seqs.faa", 'w') as f:
    #     for file in os.listdir(path):
    #         file_path = f"{path}/{file}"
    #         for record in SeqIO.parse(file_path, 'fasta'):
    #             f.write(f">{record.description}\n{str(record.seq)}\n")

    
    hhblits = "/sujin/envs/LTEnjoy/bin/hhblits"
    db_path = "/sujin/dataset/uniclust30_2020_06/UniRef30_2020_06"
    path = "af2_hhblits/faa"
    for file in os.listdir(path):
        print(file)
        if file[-1] == 'a':
            file_path = f"{path}/{file}"
            out = f"af2_hhblits/{file.replace('faa', 'a3m')}"
            if not os.path.exists(out):
                    os.system(f"{hhblits} -i {file_path} -n 3 -d {db_path} -cpu 12 -oa3m {out}")
    
    # path = "af2_hhblits"
    # for file in os.listdir(path):
    #     if file[-1] == 'm':
    #         file_path = f"{path}/{file}"
    #         out = f"{path}/{file.replace('.a3m', '_hhblits_formatted.a3m')}"
    #         parse_a3m(file_path, out)

    # path = "twintowers/result"
    # rename_by_id(path, 'faa')
    
    # path = "twintowers/90#1CF7_4_B_twintowers.a3m"
    # path = 'query_filtered_out.a3m'
    # neff = calc_neff(path)
    # print(neff)
    
    # print("        hhblits   ours")
    # for i, record in enumerate(SeqIO.parse("CASP14Sequence.faa", 'fasta')):
    #     id = record.id
    #     hh_path = f"af2_hhblits/{id}_hhblits_formatted.a3m"
    #     tt_path = f"af2_twintowers/{id}_twintowers.a3m"
    #
    #     if os.path.exists(hh_path) and os.path.exists(tt_path):
    #         hh_neff = calc_neff(hh_path)
    #         tt_neff = calc_neff(tt_path)
    #         print(f"No.{i} {id}   {hh_neff:.4f} {tt_neff: .4f}")
    
    # path = 'twintowers'
    # print("        hhblits   ours")
    # for i, file in enumerate(os.listdir(path)):
    #     if '.a3m' in file:
    #         hh_path = f"hhblits/{file.replace('twintowers', 'hhblits_formatted')}"
    #         tt_path = f"twintowers/{file}"
    #         hh_neff = calc_neff(hh_path)
    #         tt_neff = calc_neff(tt_path)
    #         print(f"No.{i}   {file} {hh_neff: .4f}  {tt_neff: .4f}")
    
    # name_dict = {'uniclust30_result.tsv': 'scope 1500w数据库采样负样本',
    #              'uniref100_result.tsv': 'scope 2亿数据库采样负样本',
    #              'pku_result.tsv': '北大数据集 2亿数据库采样负样本',
    #              'esm1b_result.tsv': '原始ESM1b'}

    # dir = 'trRosetta'
    # for file in os.listdir(dir):
    #     if not '_result.tsv' in file:
    #         continue
    #
    #     path = f"{dir}/{file}"
    #     if not os.path.exists(path):
    #         continue
    #
    #     data = pd.read_csv(path, sep='\t')
    #     type_num = len(data.groupby('query_id'))
    #     N = 0
    #     P = 1500 * 10
    #     TP = 0
    #     for query, hits in data.groupby('query_id'):
    #         q_id = query.split('_')[0]
    #
    #         for i, hit in enumerate(hits['target_id'].values):
    #             N += 1
    #             if hit.split('_')[0] == q_id:
    #                 TP += 1
    #
    #         # print(f"id:{query:<15}MAP:{MAP:.4f}")
    #     print(f"{file:<40} recall: {TP}/{P:<10} {TP/P*100:.2f}%     precision: {TP}/{N:<10} {TP/N*100:.2f}%")
    #
    # path = "af2_twintowers/result"
    # rename_by_id(path, 'a3m')
    
    # path = 'hhblits'
    # for file in os.listdir(path):
    #     if '.a3m' in file:
    #         hh_path = f"{path}/{file}"
    #         tt_path = 'twintowers/' + file.replace('.a3m', '_twintowers.faa')
    #
    #         if not os.path.exists(tt_path):
    #             continue
    #
    #         hh_num = len(list(SeqIO.parse(f"{hh_path}", 'fasta')))
    #         tt_num = len(list(SeqIO.parse(tt_path, 'fasta')))
    #
    #         if tt_num != hh_num:
    #             os.remove(tt_path)
    #             print(tt_num, hh_num)

            # seqs = list(SeqIO.parse(tt_path, 'fasta'))[:hh_num]

            # with open(tt_path, 'w') as f:
            #     for seq in seqs:
            #         f.write(f">{seq.description}\n{str(seq.seq)}\n")
            
    # from Results.format import align
    # path = 'twintowers'
    # for file in os.listdir(path):
    #     if file[-3:] == 'faa':
    #         faa_path = f"{path}/{file}"
    #         a3m_path = faa_path.replace('faa', 'a3m')
    #         if os.path.exists(a3m_path):
    #             continue
    #
    #         align(faa_path, faa_path.replace('faa', 'a3m'), 'a3m')

    # from model import MSARetrieveModel
    # import torch
    #
    # device = 'cuda'
    # model = MSARetrieveModel()
    # path = '../PretrainedModels/esm1b_t33_650M_UR50S.pt'
    # model.load_state_dict(torch.load(path, map_location='cpu'))
    # model.to(device)
    #
    # from LoadingData.tokenizer import Tokenizer
    #
    # BATCH = 64
    # tokenizer = Tokenizer()
    #
    # types = ['test', 'valid']
    # path = "/sujin/dataset/contact/proteinnet/proteinnet_{}.json"
    #
    # total = 0
    # total_correct = 0
    # for t in types:
    #     file_path = path.format(t)
    #     data = json.load(open(file_path, 'r'))
    #
    #     total_correct = 0
    #     total = 0
    #     for seq in data:
    #         id = seq['id']
    #
    #         label = create_label(seq)
    #         query_ids = tokenizer.batch_encode([seq['primary']], padding=True)['input_ids'].to(device)
    #
    #         with torch.no_grad():
    #             pred = model.esm1b.predict_contacts(query_ids).squeeze(0).to('cpu').numpy()
    #
    #         correct, num = contact_eval(pred, label, divisor=5, dist_tpye='long')
    #         total_correct += correct
    #         total += num
    #
    # print(f"{total_correct}/{total} {total_correct/total*100:.2f}%")
    
    # import torchvision
    # import torch
    # model = torchvision.models.resnet50()
    # # model.conv1 = torch.nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # model.load_state_dict(torch.load("../PretrainedModels/resnet50-19c8e357.pth"))
    # # print(model)
    #
    # from model import MSARetrieveModel
    # import torch
    # a = torch.randn(5, 5)
    # device = 'cuda'
    # esm = MSARetrieveModel().to(device)
    # path = '../PretrainedModels/esm1b_t33_650M_UR50S.pt'
    # esm.load_state_dict(torch.load(path, map_location='cpu'))
    #
    # from LoadingData.tokenizer import Tokenizer
    # tokenizer = Tokenizer()
    #
    # seq = "VTVDDLVEGIAFSITHDSENPNIVYLKSLMPSSYQVCWQHPQGRSQEREVTLQMPFEGKYEVTFGVQTRGGIVYGNPATFTIDSFCADFVN"
    # # seq2 = "LTSADLQEGTAFTIEADASNPNLIHLTN-KLSGYDAFWSHPgvgTGHSTGNNVDLKIAFQGKYPVVFGVRTPQGMIYSDTTWVDINTFCADFVS"
    # seq = seq.replace('-', '').upper()
    # # seq2 = seq2.replace('-', '').upper()
    # ori = seq
    # # print(seq)
    # n = 100
    # ratio = np.arange(0.2, 1, 0.1)
    # print(ratio)
    # with open("test.a3m", 'w') as f:
    #     f.write(f">ori\n{ori}\n")
    #     print(ori)
    #     for i in tqdm(range(n)):
    #         for j in ratio:
    #             seq = ori
    #             mutation_num = int(len(seq) * j)
    #             mutation_loc = torch.tensor(random.sample(range(len(seq)), mutation_num))
    #             for loc in mutation_loc:
    #                 with torch.no_grad():
    #                     info_dict = tokenizer.batch_encode([seq], padding=True)
    #                     query_ids = info_dict['input_ids'].to(device)
    #                     query_ids[0] = tokenizer.add_mask_token(query_ids[0], loc+1)
    #                     lengths = info_dict['lengths']
    #                     vec = esm.get_lm_head(query_ids, lengths)[:, 1:-1]
    #                     l = random.choice(vec[0][loc].argsort(descending=True)[:2])
    #                     token = tokenizer.convert_id_to_token(l)
    #                     seq = seq[:loc] + token + seq[loc+1:]
    #
    #             f.write(f">{i}\n{seq}\n")
    #             print(seq)
        
    # a, b = vec
    # print(torch.norm(a - b, 2))
    # a = a.numpy().astype(np.float32)
    # b = b.numpy().astype(np.float32)
    # print(np.linalg.norm(a-b, 2))
    #
    # import matplotlib.pyplot as plt
    # plt.imshow(a.to('cpu').numpy(), 'gray')
    # plt.show()
    # plt.imshow(b.to('cpu').numpy(), 'gray')
    # plt.show()


    # from model import ContactMapModel
    # model = ContactMapModel()
    # maps = []
    # for map, length in tuple(zip(pic, lengths)):
    #     img = model.map_transform(map[:length, :length])
    #     maps.append(img[:, :, 0])
    #
    # a, b = maps
    # print(a)
    # print(b)
    # print(torch.norm(a-b, 2))
    # import matplotlib.pyplot as plt
    # plt.imshow(a, 'gray')
    # plt.show()
    # plt.imshow(b, 'gray')
    # plt.show()
    #
    # import cv2
    # img = cv2.dct(a.numpy())[0: 8, 0: 8]
    # print(img.shape)
    # print(img)
    #
    # hash_a = np.where(img > img.mean(), 1, 0).flatten()
    # print(hash_a)
    #
    # img = cv2.dct(b.numpy())[0: 8, 0: 8]
    # hash_b = np.where(img > img.mean(), 1, 0).flatten()
    # print(hash_b)
    #
    # res = np.bitwise_xor(hash_b, hash_a)
    # print(res)
    # print(res.sum())
    # pic = pic[:, None, :, :].repeat(1, 3, 1, 1)
    # print(pic.size())
    # with torch.no_grad():
    #     res = model(pic)
    # print(res.size())
    # a = res[0]
    # b = res[1]
    # print(torch.norm(a-b, 2))
    # import matplotlib.pyplot as plt
    # from model import ContactMapModel
    # pic = np.load("../pic.npy")
    # model = ContactMapModel()
    # img = model.map_transform(pic)
    # print(img.shape)
    # print(pic)
    # print(img[:, :, 0])
    # cv2.imshow('111', pic)
    # from matplotlib import pyplot as plt
    # plt.imshow(pic, 'gray', vmin=0, vmax=255)
    # plt.show()
    # plt.imshow(img, 'gray', vmin=0, vmax=255)
    # plt.show()
    #
    # from model import AutoEncoder, MSARetrieveModel
    #
    # model = MSARetrieveModel()
    # path = '../PretrainedModels/esm1b_t33_650M_UR50S.pt'
    # model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    # model.encoder.load_state_dict(torch.load('../PretrainedModels/AutoEncoder_t30.pt', map_location='cpu'))
    #
    # torch.save(model.state_dict(), '../PretrainedModels/esm1b_t33_650M_UR50S_AE.pt')
    
    # import matplotlib.pyplot as plt
    #
    # auto = AutoEncoder()
    # model = MSARetrieveModel()
    # data = np.load("../pic.npy")
    # data = model.map_transform(data)
    # # torch.save(torch.tensor(data)[None, None, :, :], '/sujin/TwinTowers/Data/autoencoder_train.pt')
    # plt.imshow(data, 'gray')
    # plt.show()
    #
    # data = torch.tensor(data)[:64, :64][None, None, :, :]
    #
    # with torch.no_grad():
    #     res = auto(data)
    # print(res.size())
    # plt.imshow(res.numpy()[0, 0], 'gray')
    # plt.show()
    
    # a3m = "twintowers/result/TBM#T0872.a3m"
    # a3m = "hhblits/TBM#T0873.a3m"
    # path = '/sujin/dataset/trRosetta/training_set/a3m/train'
    # # path = 'twintowers/result'
    #
    # pattern = r'[a-z]+'
    #
    # for file in os.listdir(path):
    #     if '.a3m' in file:
    #         a3m = f"{path}/{file}"
    #         records = list(SeqIO.parse(a3m, 'fasta'))
    #         step = int(len(records) / 100)
    #         query = str(records[0].seq)
    #         # records = [records[i] for i in range(step, len(records), step)]
    #         identities = np.empty(len(records))
    #
    #         for i, record in enumerate(records):
    #             seq = re.sub(pattern, '', str(record.seq))
    #             cnt = 0
    #             total = 0
    #             for q, c in tuple(zip(query, seq)):
    #                 if c != '-':
    #                     total += 1
    #                     cnt += int(q == c)
    #             identities[i] = cnt / total * 100
    #
    #         identities = np.sort(np.round(identities))
    #         for i in range(len(records)):
    #             print(identities[-i])
    #         print(len(identities))
    #         break
    
    # from model import MSATransformerRetrieveModel, MSARetrieveModel
    # from LoadingData.tokenizer import Tokenizer
    # device = 'cuda'
    # model = MSATransformerRetrieveModel()
    # path = "/sujin/TwinTowers/PretrainedModels/esm_msa1_t12_100M_UR50S.pt"
    # model.load_state_dict(torch.load(path))
    # model.eval()
    # model.to(device)
    #
    # esm1b = MSARetrieveModel()
    # path = "/sujin/TwinTowers/PretrainedModels/esm1b_t33_650M_UR50S.pt"
    # esm1b.load_state_dict(torch.load(path))
    # esm1b.eval()
    # esm1b.to(device)
    #
    # tokenizer = Tokenizer()
    #
    # path = '/sujin/dataset/trRosetta/training_set/a3m/test'
    # for file in os.listdir(path):
    #     file_path = f"{path}/{file}"
    #     break
    #
    # seqs = []
    # for record in SeqIO.parse(file_path, 'fasta'):
    #     seq = re.sub(r'[a-z]+', '', str(record.seq))
    #     seqs.append(seq)
    #
    # random.seed(2022)
    # input_seqs = seqs[0:1] + ['VTVDDLVEGIAFSITHDSENPNIVYLKSLMPSSYQVCWQHPQGRSQEREVTLQMPFEGKYEVTFGVQTRVYGNPATFT']
    # input_ids = tokenizer.batch_encode(input_seqs, padding=True)['input_ids'].to(device)
    #
    # with torch.no_grad():
    #     outputs = model(input_ids.unsqueeze(0))['vec']
    #     esm1b_out = esm1b(input_ids)['vec']
    #     print(outputs.size())
    #     print(torch.norm(outputs[0, 0:1] - outputs[0], 2, dim=-1))
    #     vec1 = outputs[0, 0]
    #
    #     print(torch.norm(esm1b_out[0:1] - esm1b_out, 2, dim=-1))
    #
    # input_seqs = ['VTVDDLVEGIAFSITHDSENPNIVYLKSLMPSSYQVCWQHPQGRSQEREVTLQMPFEGKYEVTFGVQTRVYGNPATFT'] + seqs[0:1]
    # input_ids = tokenizer.batch_encode(input_seqs, padding=True)['input_ids'].to(device)
    #
    # with torch.no_grad():
    #     outputs = model(input_ids.unsqueeze(0))['vec']
    #     vec2 = outputs[0, 0]
    # print(vec1, vec2)
    # print(torch.norm(vec1-vec2, 2))
    #