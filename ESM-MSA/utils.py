# 自己平常写代码时为了方便实现某些功能而写的工具函数
import pandas as pd
import torch
import os
import numpy as np
import random
import time
import matplotlib.pylab as plt
from tqdm import tqdm
from Bio import SeqIO
from math import ceil


# 统计时间的函数
class TimeCounter:
    def __init__(self, text, verbose=True):
        self.text = text
        self.verbose = verbose

    def __enter__(self):
        self.start = time.time()
        if self.verbose:
            print(self.text, flush=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        t = end - self.start
        if self.verbose:
            print(f"Finished. The time is {t:.2f}s.\n", flush=True)


# 进度条
def progress_bar(now: int, total: int, desc: str = '', end='\n'):
    length = 50
    now = now if now <= total else total
    num = now * length // total
    progress_bar = '[' + '#' * num + '_' * (length - num) + ']'
    display = f'{desc:} {progress_bar} {int(now/total*100):02d}% {now}/{total}'

    print(f'\r\033[31m{display}\033[0m', end=end, flush=True)


# 设定种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 随机种子
def random_seed():
    torch.seed()
    torch.cuda.seed()
    np.random.seed()
    random.seed()
    torch.backends.cudnn.deterministic = False


def truncate(seq, t):
    if len(seq) <= t:
        return seq
    
    else:
        st = random.randint(0, len(seq) - t)
        return seq[st: st + t]


def imgToSize(img, size):
    ''' imgToSize()
	# ----------------------------------------
	# Function:   将图像等比例缩放到 512x512 大小
	#             根据图像长宽不同分为两种缩放方式
	# Param img:  图像 Mat
	# Return img: 返回缩放后的图片
	# Example:    img = imgToSize(img)
	# ----------------------------------------
	'''
    # 测试点
    # cv2.imshow('metaImg.jpg', img)
    
    imgHeight, imgWidth = img.shape[:2]
    
    # cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src 原图像，dsize 输出图像的大小，
    # img = cv2.resize(img, (512,512))
    zoomHeight = size
    zoomWidth = int(imgWidth * size / imgHeight)
    img = cv2.resize(img, (zoomWidth, zoomHeight))
    
    # 测试点
    # cv2.imshow('resizeImg', img)
    
    # 如果图片属于 Width<Height，那么宽度将达不到 512
    if imgWidth >= imgHeight:
        # 正常截取图像
        w1 = (zoomWidth - size) // 2
        # 图像坐标为先 Height，后 Width
        img = img[0:size, w1:w1 + size]
    else:
        # 如果宽度小于 512，那么对两侧边界填充为全黑色
        # 根据图像的边界的像素值，向外扩充图片，每个方向扩充50个像素，常数填充：
        # dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]])
        # dst = cv2.copyMakeBorder(img,50,50,50,50, cv2.BORDER_CONSTANT,value=[0,255,0])
        # 需要填充的宽度为 512-zoomWidth
        left = (size - zoomWidth) // 2
        # 避免余数取不到
        right = left + 1
        img = cv2.copyMakeBorder(img, 0, 0, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        img = img[0:size, 0:size]
    
    # 测试点
    # cv2.imshow('size512', img)
    
    return img


if __name__ == '__main__':
    # file_names = [f"{i}_{i+100}" for i in range(0, 1100, 100)]
    # files = []
    # path = "/sujin/dataset/uniclust30"
    # for name in file_names:
    #     files.append(open(f"{path}/UniRef30_2020_06_{name}", 'w'))
    #
    # fasta_path = "/sujin/dataset/uniclust30/UniRef30_2020_06.faa"
    # with open(fasta_path, 'r') as r:
    #     desc = r.readline()
    #     while desc:
    #         seq = r.readline()
    #         if len(seq) - 1 > 1022:
    #             desc = r.readline()
    #             continue
    #
    #         index = int((len(seq) - 1) / 100)
    #         files[index].write(desc)
    #         files[index].write(seq)
    #
    #         desc = r.readline()
    # from math import ceil
    # from LoadingData.data_construction import construct_pointer
    # part_dict = {"300_400": 4, "400_500": 3, "500_600": 2}
    #
    # for name, part_num in part_dict.items():
    #     path = f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}.faa"
    #     npy = f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}.npy"
    #
    #     vec = np.load(npy)
    #     total = vec.shape[0]
    #     part_len = ceil(total / part_num)
    #     print(vec.shape)
    #     records = SeqIO.parse(path, 'fasta')
    #     for i in range(part_num):
    #         st = i * part_len
    #         ed = st + part_len
    #         if ed > vec.shape[0]:
    #             ed = vec.shape[0]
    #
    #         fasta_path = f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}_part{i}.faa"
    #         np.save(f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}_part{i}.npy", vec[st: ed])
    #         with open(fasta_path, "w") as f:
    #             for _ in range(st, ed):
    #                 record = next(records)
    #                 f.write(f">{record.description}\n{str(record.seq)}\n")
    #
    #         pointer_path = f"/sujin/dataset/uniclust30/UniRef30_2020_06_{name}/UniRef30_2020_06_{name}_part{i}_pointer.tsv"
    #
    #         if not os.path.exists(pointer_path):
    #             construct_pointer(fasta_path, pointer_path)

    from dynamic_sampling import truncate
    # path = "/sujin/dataset/trRosetta/training_set/a3m"
    # query = open("task_eval/trRosetta/trRosetta_test.faa", 'w')
    # db = open('task_eval/trRosetta/trRosetta_test_db.faa', 'w')
    #
    # for i, file in enumerate(tqdm(os.listdir(path))):
    #     records = list(SeqIO.parse(f"{path}/{file}", 'fasta'))
    #     samples = random.sample(records[1:], 10)
    #
    #     query_seq = truncate(str(records[0].seq), 1022)
    #
    #     query.write(f">{i}_{records[0].id}\n{query_seq}\n")
    #
    #     for sample in samples:
    #         seq = truncate(str(sample.seq).replace('-', '').upper(), 1022)
    #         db.write(f">{i}_{sample.id}\n{seq}\n")
   
    # with open('task_eval/trRosetta/trRosetta_test_db1.faa', 'w') as f:
    #     with open('task_eval/trRosetta/trRosetta_test_db.faa', 'r') as r:
    #         line = r.readline()
    #         while line:
    #             line = truncate(line[:-1].replace('-', '').upper(), 1022)
    #             f.write(line + '\n')
    #             line = r.readline()
    
    # with open("task_eval/trRosetta/trRosetta_test_db.faa", 'r') as f:
    #     print(f.readline(), f.readline())
    # for record in SeqIO.parse("task_eval/trRosetta/trRosetta_test_db.faa", 'fasta'):
    #     print(record.seq)
    #     if 'p' in record.seq:
    #         print(record.seq)
    #         break
    
    # setup_seed(2021)
    # # path = "/sujin/dataset/trRosetta/training_set/a3m/test"
    # path = "/sujin/dataset/MSA_Uniref50/Example_100G/0078"
    # files = os.listdir(path)
    #
    # query_seqs = open("Data/pku_test.faa", 'w')
    # query_db = open("Data/pku_test_db.faa", 'w')
    #
    # cnt = 0
    # for i, file in tqdm(enumerate(files)):
    #     records = list(SeqIO.parse(f"{path}/{file}", 'fasta'))
    #     if len(records) < 10:
    #         continue
    #     samples = random.sample(records[1:], 10)
    #
    #     query_seq = truncate(str(records[0].seq), 1022)
    #
    #     query_seqs.write(f">{i}_{records[0].id}\n{query_seq}\n")
    #
    #     for sample in samples:
    #         seq = truncate(str(sample.seq).replace('-', '').upper(), 1022)
    #         query_db.write(f">{i}_{sample.id}\n{seq}\n")
    #
    #     cnt += 1
    #     if cnt == 1500:
    #         break
    
    # path = "/sujin/dataset/trRosetta/training_set/a3m/train"
    # with open("/sujin/dataset/trRosetta/training_set/a3m/trRosetta_train_id40_index.tsv", 'w') as f:
    #     f.write('st\tnum\n')
    #     st = 0
    #     for i, file in tqdm(enumerate(os.listdir(path))):
    #         cnt = 0
    #         for record in SeqIO.parse(f"{path}/{file}", 'fasta'):
    #             if len(str(record.seq).replace('-', '')) <= 512:
    #                 cnt += 1
    #
    #         f.write(f"{st}\t{cnt}\n")
    #         st += cnt
    
    # names = [f'{i}_{i+100}' for i in range(0, 1100, 100)]
    #
    # for name in tqdm(names):
    #     path = f'/sujin/dataset/uniclust30/UniRef30_2020_06_{name}'
    #     seqs = []
    #     for file in os.listdir(path):
    #         if '.faa' in file:
    #             seqs += list(SeqIO.parse(f"{path}/{file}", 'fasta'))
    #
    #     n = ceil(len(seqs) / 10)
    #
    #     for i in range(10):
    #         st = i * n
    #         with open(f"{path}/UniRef30_2020_06_{name}_{i}.faa", 'w') as f:
    #             for seq in seqs[st: st+n]:
    #                 f.write(f">{seq.description}\n{str(seq.seq)}\n")

    # path = "/sujin/dataset/trRosetta/training_set/a3m/train"
    # cnt = 0
    # for file in tqdm(os.listdir(path)):
    #     if ".a3m" in file:
    #         cnt += 1
    #         a3m = f"{path}/{file}"
    #         faa = a3m.replace(".a3m", ".faa")
    #         if os.path.exists(faa):
    #             continue
    #
    #         with open(faa, 'w') as f:
    #             for record in SeqIO.parse(a3m, 'fasta'):
    #                 id = f"{cnt}_{record.id}"
    #                 f.write(f">{id}\n{str(record.seq).replace('-', '').upper()}\n")

    # path = "/sujin/dataset/trRosetta/training_set/a3m/train"
    # cd_hit = "/sujin/software/cd-hit-v4.6.7-2017-0501/cd-hit"
    # for file in tqdm(os.listdir(path)):
    #     if ".faa" in file and "id40" not in file:
    #         faa = f"{path}/{file}"
    #         id40 = faa.replace(".faa", "_id40.faa")
    #         os.system(f"{cd_hit} -i {faa} -o {id40} -c 0.4 -n 2 -T {127} -M 16000 >/dev/null 2>&1")

    # path = "/sujin/dataset/trRosetta/training_set/a3m/train"
    # cnt = 0
    # with open("/sujin/dataset/trRosetta/training_set/a3m/trRosetta_train_id40.faa", 'w') as f:
    #     for file in tqdm(os.listdir(path)):
    #         if "id40" in file:
    #             faa = f"{path}/{file}"
    #             for record in SeqIO.parse(faa, 'fasta'):
    #                 if len(str(record.seq)) > 512:
    #                     continue
    #                 seq = str(record.seq)
    #                 f.write(f">{record.description} \n{seq}\n")
    
    # faa = "/sujin/dataset/trRosetta/training_set/a3m/trRosetta_train_id40.faa"
    # index = "/sujin/dataset/trRosetta/training_set/a3m/trRosetta_train_id40_index.tsv"
    # # pointer = "/sujin/dataset/trRosetta/training_set/a3m/trRosetta_train_id40_pointer.tsv"
    # # from LoadingData.data_construction import construct_pointer
    # # construct_pointer(faa, pointer)
    # with open(faa, 'r') as r:
    #     with open(index, 'w') as w:
    #         w.write('st\tnum\n')
    #         st = 0
    #         cnt = 1
    #         now = -1
    #         for record in tqdm(SeqIO.parse(faa, 'fasta')):
    #             id = int(record.id.split('_')[0])
    #             if now != id:
    #                 if now == -1:
    #                     now = id
    #
    #                 else:
    #                     now = id
    #                     w.write(f"{st}\t{cnt}\n")
    #                     st += cnt
    #                     cnt = 1
    #
    #             else:
    #                 cnt += 1
    #
    #         w.write(f"{st}\t{cnt}\n")
    
    # tt_path = 'task_eval/query_0.faa'
    # hh_path = 'task_eval/hhblits/40#1X51_1_A.a3m'
    #
    # tt_ids = []
    # hh_ids = []
    #
    # for record in SeqIO.parse(tt_path, 'fasta'):
    #     tt_ids.append(record.id)
    #
    # for record in SeqIO.parse(hh_path, 'fasta'):
    #     hh_ids.append(record.id)
    #
    # common = set(tt_ids) & set(hh_ids)
    #
    # print(len(common))

    # import cv2
    # from model import MSARetrieveModel, ContactMapModel, AutoEncoder
    # device = 'cuda'
    # model = MSARetrieveModel()
    # path = 'PretrainedModels/esm1b_t33_650M_UR50S.pt'
    # # model_path = "PretrainedModels/esm1b_t33_650M_UR50S.pt"
    # model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)
    # # model.encoder.load_state_dict(torch.load("PretrainedModels/AutoEncoder.pt", map_location='cpu'))
    # model.to(device)
    # model.eval()
    #
    # encoder = AutoEncoder()
    # # encoder.load_state_dict(torch.load('PretrainedModels/AutoEncoder_t8.pt', map_location='cpu'), strict=False)
    # encoder.to(device)
    # encoder.eval()
    #
    # from LoadingData.tokenizer import Tokenizer
    # BATCH = 64
    # tokenizer = Tokenizer()
    # from PIL import Image
    # # path = '/sujin/TwinTowers/task_eval/hhblits'
    # # path = 'task_eval/twintowers/result'
    # # path = 'task_eval/trRosetta/result'
    # path = '/sujin/dataset/trRosetta/training_set/a3m/test'
    # # path = '/sujin/TwinTowers/task_eval/trRosetta/result'
    # file_list = os.listdir(path)
    # for file in os.listdir(path):
    #     if 'a3m' in file:
    #         print(file)
    #         # file = file_list[10]
    #         a3m = f"{path}/{file}"
    #         # a3m = 'task_eval/hhblits/10#2IJR_1_A.a3m'
    #         seqs = []
    #         ids = []
    #         record = next(SeqIO.parse(a3m, 'fasta'))
    #         # print(len(str(record.seq)))
    #         # continue
    #         for record in SeqIO.parse(a3m, 'fasta'):
    #             seqs.append(str(record.seq).replace('-', '').upper())
    #             ids.append(record.id)
    #
    #         def pHash(img):
    #             img = cv2.dct(img.astype(float))[0: 8, 0: 8]
    #             return np.where(img > img.mean(), 1, 0).flatten()
    #
    #         with torch.no_grad():
    #             query = seqs[0]
    #             # query = 'VTVDDLVEGIAFSITHDSENPNIVYLKSLMPSSYQVCWQHPQGRSQEREVTLQMPFEGKYEVTFGVQTRGGIVYGNPATFTIDSFCADFVN'
    #             info_dict = tokenizer.batch_encode([query], padding=True)
    #             query_ids = info_dict['input_ids'].to(device)
    #             lengths = info_dict['lengths']
    #
    #             # ori_pic = model.esm1b.predict_contacts(query_ids, lengths)[0].to('cpu').numpy()
    #             # ori_dist = cmodel(ori_pic, lengths)
    #             ori_vec = model(query_ids)['vec']
    #             pic = model.esm1b.predict_contacts(query_ids, lengths=lengths)
    #             a = pic[0]
    #             ori_pic = model.get_contact_map(query_ids, lengths)
    #             # plt.imshow(ori_pic[0].to('cpu').numpy(), cmap='gray', interpolation='none')
    #             # plt.show()
    #             # ori_hash = pHash(ori_pic)
    #             # img = torch.where(ori_pic[0] > 0.5)
    #             v = ori_pic[0].flatten().topk(200)[0][-1]
    #             img = torch.where(ori_pic[0] >= v, 1., 0.)
    #             # ori_vec = encoder.encode(img[None, None, :, :])
    #             # ori_hash = pHash(ori_pic[0].to('cpu').numpy())
    #             # plt.imshow(img.to('cpu').numpy(), cmap='gray')
    #             # plt.show()
    #             pred = encoder(img[None, None, :, :])
    #             ori_img = ori_pic[0].to('cpu').numpy()
    #             # ori_hash = pHash(img.to('cpu').numpy())
    #             # print(ori_hash)
    #             plt.imshow(pred[0, 0].to('cpu').numpy(), cmap='gray')
    #             # plt.show()
    #
    #             img = Image.fromarray(img.to('cpu').numpy())
    #             import imagehash
    #             hash = imagehash.phash(img, 8)
    #             print(hash, type(hash))
    #
    #
    #             for i in range(30):
    #                 query = seqs[i+100]
    #                 info_dict = tokenizer.batch_encode([query], padding=True)
    #                 query_ids = info_dict['input_ids'].to(device)
    #                 lengths = info_dict['lengths']
    #
    #                 # hit = model.esm1b.predict_contacts(query_ids).squeeze(0).to('cpu').numpy()
    #                 # print(query)
    #                 pic = model.esm1b.predict_contacts(query_ids, lengths=lengths)
    #                 hit = model.get_contact_map(query_ids, lengths)
    #                 v = hit[0].flatten().topk(200)[0][-1]
    #                 img = torch.where(hit[0] >= v, 1., 0.)
    #                 img = Image.fromarray(img.to('cpu').numpy())
    #                 hit_hash = imagehash.phash(img, 8)
    #                 # plt.imshow(hit, cmap='gray', interpolation='none')
    #                 # plt.title(ids[i])
    #                 # plt.show()
    #                 # hit_dist = cmodel(hit, lengths)
    #                 # hit_hash = pHash(hit)
    #                 # print(f">{ids[i]}, {torch.norm(ori_pic[0] - hit[0], 2)}")
    #                 # print(f">{ids[i]}, {torch.norm(ori_vec-hit_vec, 2)}")
    #                 # print(hit_vec.norm(2))
    #                 # print(f">{ids[i]}, {np.bitwise_xor(ori_hash, hit_hash).sum()}")
    #                 pic = model.esm1b.predict_contacts(query_ids, lengths=lengths)
    #                 b = pic[0]
    #                 v = hit[0].flatten().topk(200)[0][-1]
    #                 img = torch.where(hit[0] >= v, 1., 0.)
    #                 hit_vec = model(query_ids)['vec']
    #                 print(f">{ids[i]}, {len(query)}, {torch.norm(ori_vec - hit_vec, 2)}")
    #                 # hit_vec = encoder.encode(img[None, None, :, :])
    #                 plt.imshow(img.to('cpu').numpy(), cmap='gray')
    #                 plt.title(ids[i])
    #                 # plt.show()
    #                 pred = encoder(img[None, None, :, :])
    #                 hit_img = hit[0].to('cpu').numpy()
    #                 img = Image.fromarray(img.to('cpu').numpy())
    #
    #                 hit_hash = imagehash.phash(img, 8)
    #                 # print(hit_hash.hash.flatten().shape)
    #                 # print(f">{ids[i]}, {hash - hit_hash}")
    #                 # print(f">{ids[i]}, {torch.mm(ori_vec, hit_vec.T)}")
    #                 # hit_hash = pHash(img.to('cpu').numpy())
    #                 # cost, _, flow = cv2.EMD(ori_img.astype(np.float32), hit_img, cv2.DIST_L1)
    #                 # print(f">{ids[i]}, {cost}")
    #                 # plt.imshow(pred[0, 0].to('cpu').numpy(), cmap='gray')
    #                 # plt.show()
    #             # for i in range(10):
    #             #     query = seqs[i]
    #             #     info_dict = tokenizer.batch_encode([query], padding=True)
    #             #     query_ids = info_dict['input_ids'].to(device)
    #             #     lengths = info_dict['lengths']
    #             #
    #             #     # hit = model.esm1b.predict_contacts(query_ids).squeeze(0).to('cpu').numpy()
    #             #     hit = model(query_ids, lengths=None)['vec'].squeeze(0)
    #             #     print(f">{ids[-i]}, {torch.norm(ori - hit, 2)}")
    #             #     # print(query)
    #             #     hit = model.esm1b.predict_contacts(query_ids).squeeze(0).to('cpu').numpy()
    #             #     hit = np.round(hit * 255)
    #             #     plt.imshow(hit, cmap='gray', interpolation='none', vmin=0, vmax=255)
    #             #     plt.title(ids[i])
    #             #     plt.show()
    #             break
                # cnt = 0
                #
                # vec = np.empty((len(seqs), 1280))
                #
                # # np.save("vec.npy", vec)
                # print(len(seqs), cnt)
    
    # import faiss
    # vec = np.load("vec.npy").astype(np.float32)
    # dim, measure = 1280, faiss.METRIC_L2
    # param = f'HNSW64, Flat'
    # index = faiss.index_factory(dim, param, measure)
    # if not index.is_trained:
    #     index.train(vec)
    # index.add(vec)
    # ori = vec[0:1]
    #
    # dist, indices = index.search(ori, index.ntotal)
    # print(dist[0])
    # print((dist[0] <= 1).sum())
    
    # path = "task_eval/twintowers/result"
    # hmmer = "/sujin/software/hmmer-3.3.2/src/jackhmmer"
    # reformat = "/sujin/envs/LTEnjoy/scripts/reformat.pl"
    # for file in os.listdir(path):
    #     faa = f"{path}/{file}"
    #     if os.path.exists(faa.replace('faa', 'a3m')):
    #         continue
    #
    #     query = next(SeqIO.parse(faa, 'fasta'))
    #     with open(f'{path}/temp', 'w') as f:
    #         f.write(f">{query.description}\n{str(query.seq)}\n")
    #
    #     a3m = faa.replace('.faa', '.a3m')
    #     os.system(f"{hmmer} -N 3 -A {path}/res {path}/temp {faa} && {reformat} sto a3m {path}/res {faa.replace('faa', 'a3m')}")

    # path = '/sujin/dataset/trRosetta/training_set/a3m/train'
    # for file in os.listdir(path):
    #     if 'faa' in file:
    #         faa = f"{path}/{file}"
    #         out = faa.replace('.faa', '_1024.faa')
    #
    #         seqs = list(SeqIO.parse(faa, 'fasta'))
    #         print(len(seqs))
    #         step = len(seqs) / 1024
    #         step = ceil(step) if step < 1 else int(step)
    #         seqs = [seqs[i] for i in range(0, len(seqs), step)]
    #
    #         print(len(seqs))
    
    # pic = np.load("pic.npy")
    # print(pic.shape)
    # res = cv2.resize(pic, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    # res[res < 0] = 0
    # print(res.shape)
    # plt.imshow(pic, 'gray', vmin=0, vmax=1)
    # plt.show()
    # print(pic)
    # print(res)
    # plt.imshow(np.power(res, 0.9), 'gray', vmin=0, vmax=1)
    # plt.show()
    
    device = 'cuda'
    setup_seed(2022)
    from model import MSARetrieveModel, FullyConnectedModel, DirectedEvolutionModel
    from LoadingData.tokenizer import Tokenizer
    esm1b = MSARetrieveModel()
    # esm1b.load_state_dict(torch.load(model_path, map_location='cpu'))

    top_model = FullyConnectedModel(input_size=1280,
                                    hidden_size=1280,
                                    hidden_layer_num=0,
                                    label_num=1)
    
    model_path = "PretrainedModels/DirectedEvolutionModel_t9.pt"
    model = DirectedEvolutionModel(esm1b, top_model)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    tokenizer = Tokenizer()
    
    model.to(device)
    seq = "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTELEVLFQGPLDPNSMATYEVLCEVARKLGTDDREVVLFLLNVFIPQPTLAQLIGALRALKEEGRLTFPLLAECLFRAGRRDLLRDLLHLDPRFLERHLAGTMSYFSPYQLTVLHVDGELCARDIRSLIFLSKDTIGSRSTPQTFLHWVYCMENLDLLGPTDVDALMSMLRSLSRVDLQRQVQTLMGLHLSGPSHSQHYRHTPLEHHHHHH"
    # 39 40 41 54
    test_list, test_fitness = model.evolve(device, seq, [39, 41, 54])
    test_fitness.sort(reverse=True)
    print(test_fitness)
    # name = "BLAT_ECOLX_Ranganathan2015"
    # path = f"/sujin/TwinTowers/task_eval/mutation/datasets/{name}/{name}_twintowers.npy"
    # data = np.load(path)
    # indices = data[0].argsort()
    #
    # print(data[0][indices])
    # print(data[1][indices])