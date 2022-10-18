# -*-coding:utf-8-*-
"""
这是tensorflow版本的GREMLIN实现
参考代码：https://github.com/sokrypton/GREMLIN_CPP/blob/master/GREMLIN_TF.ipynb
"""
import os

import numpy as np
import tensorflow as tf
import re
from Bio import SeqIO
tf.disable_eager_execution()
import matplotlib.pylab as plt
from scipy import stats
from scipy.spatial.distance import pdist,squareform
import pandas as pd


# note: if you are modifying the alphabet
# make sure last character is "-" (gap)
################
alphabet = "ARNDCQEGHILKMFPSTWYV-"
states = len(alphabet)
a2n = {}
for a,n in zip(alphabet,range(states)):
  a2n[a] = n
################

def aa2num(aa):
  '''convert aa into num'''
  if aa in a2n: return a2n[aa]
  else: return a2n['-']


# from fasta
def parse_fasta(filename, limit=-1):
    '''function to parse fasta'''
    header = []
    sequence = []
    lines = open(filename, "r")
    for line in lines:
        line = line.rstrip()
        if line[0] == ">":
            if len(header) == limit:
                break
            header.append(line[1:])
            sequence.append([])
        else:
            sequence[-1].append(line)
    lines.close()
    sequence = [''.join(seq) for seq in sequence]
    return np.array(header), np.array(sequence)


def filt_gaps(msa, gap_cutoff=0.5):
    '''filters alignment to remove gappy positions'''
    tmp = (msa == states - 1).astype(np.float64)
    non_gaps = np.where(np.sum(tmp.T, -1).T / msa.shape[0] < gap_cutoff)[0]
    print(non_gaps)
    print(msa.shape)
    return msa[:, non_gaps], non_gaps


def get_eff(msa, eff_cutoff=0.8):
    '''compute effective weight for each sequence'''
    ncol = msa.shape[1]

    # pairwise identity
    a = pdist(msa, "hamming")
    b = squareform(pdist(msa, "hamming"))
    msa_sm = 1.0 - squareform(pdist(msa, "hamming"))

    # weight for each sequence
    msa_w = (msa_sm >= eff_cutoff).astype(np.float)
    msa_w = 1 / np.sum(msa_w, -1)

    return msa_w


def mk_msa(seqs):
    '''converts list of sequences to msa'''

    msa_ori = []
    for seq in seqs:
        msa_ori.append([aa2num(aa) for aa in seq])
    msa_ori = np.array(msa_ori)
    # msa_ori的维度是[num_seqs,seq_len]
    # remove positions with more than > 50% gaps
    """
    msa_ori的维度是[num_seqs,seq_len]，eg [817,62]
    如果在某一列的gap的数量大于50%，就把这一列消去，eg 消去最后一列，维度变为[817,61]
    msa 是把之前的某一列gap的数量大于50%消去,其他和msa_ori没有区别
    v_idx = [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23, 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47, 48 49 50 51 52 53 54 55 56 57 58 59 60]
    """
    msa, v_idx = filt_gaps(msa_ori, 0.5)

    # compute effective weight for each sequence
    """
    1.求任意两个向量之间的hamming距离 -> num_seqs*（num_seqs-1)/2
    2.然后求向量与向量之间的hamming距离方阵 -> [num_seqs,num_seqs]
    3. 1-hamming距离方阵，>=0.8为1
    4. 1/np.sum(3的方阵，-1）-> [num_seqs]
    """
    msa_weights = get_eff(msa, 0.8)

    # compute effective number of sequences
    """
    num_seqs 序列长度，去除掉某一列的gap数量大于50%
    """
    ncol = msa.shape[1]  # length of sequence
    w_idx = v_idx[np.stack(np.triu_indices(ncol, 1), -1)]

    return {"msa_ori": msa_ori,
            "msa": msa,
            "weights": msa_weights,
            "neff": np.sum(msa_weights),
            "v_idx": v_idx,
            "w_idx": w_idx,
            "nrow": msa.shape[0],
            "ncol": ncol,
            "ncol_ori": msa_ori.shape[1]}


# external functions

def sym_w(w):
    '''symmetrize input matrix of shape (x,y,x,y)'''
    x = w.shape[0]
    """
    去除对角线上的元素
    """
    w = w * np.reshape(1 - np.eye(x), (x, 1, x, 1))
    '''
    A + A^T
    '''
    w = w + tf.transpose(w, [2, 3, 0, 1])
    return w


def opt_adam(loss, name, var_list=None, lr=1.0, b1=0.9, b2=0.999, b_fix=False):
    # adam optimizer
    # Note: this is a modified version of adam optimizer. More specifically, we replace "vt"
    # with sum(g*g) instead of (g*g). Furthmore, we find that disabling the bias correction
    # (b_fix=False) speeds up convergence for our case.

    if var_list is None: var_list = tf.trainable_variables()
    gradients = tf.gradients(loss, var_list)
    if b_fix: t = tf.Variable(0.0, "t")
    opt = []
    for n, (x, g) in enumerate(zip(var_list, gradients)):
        if g is not None:
            ini = dict(initializer=tf.zeros_initializer, trainable=False)
            mt = tf.get_variable(name + "_mt_" + str(n), shape=list(x.shape), **ini)
            vt = tf.get_variable(name + "_vt_" + str(n), shape=[], **ini)

            mt_tmp = b1 * mt + (1 - b1) * g
            vt_tmp = b2 * vt + (1 - b2) * tf.reduce_sum(tf.square(g))
            lr_tmp = lr / (tf.sqrt(vt_tmp) + 1e-8)

            if b_fix: lr_tmp = lr_tmp * tf.sqrt(1 - tf.pow(b2, t)) / (1 - tf.pow(b1, t))

            opt.append(x.assign_add(-lr_tmp * mt_tmp))
            opt.append(vt.assign(vt_tmp))
            opt.append(mt.assign(mt_tmp))

    if b_fix: opt.append(t.assign_add(1.0))
    return (tf.group(opt))


def GREMLIN(msa, opt_type="adam", opt_iter=1000, opt_rate=1.0, batch_size=None):
    ##############################################################
    # SETUP COMPUTE GRAPH
    ##############################################################
    # kill any existing tensorflow graph
    tf.reset_default_graph()

    ncol = msa["ncol"]  # length of sequence

    """
    MSA的数据维度
    [num_seqs,seq_len]
    """
    # msa (multiple sequence alignment)
    MSA = tf.placeholder(tf.int32, shape=(None, ncol), name="msa")

    # one-hot encode msa
    OH_MSA = tf.one_hot(MSA, states)

    # msa weights
    MSA_weights = tf.placeholder(tf.float32, shape=(None,), name="msa_weights")

    # 1-body-term of the MRF
    V = tf.get_variable(name="V",
                        shape=[ncol, states],
                        initializer=tf.zeros_initializer)

    # 2-body-term of the MRF
    W = tf.get_variable(name="W",
                        shape=[ncol, states, ncol, states],
                        initializer=tf.zeros_initializer)

    # symmetrize W
    W = sym_w(W)

    def L2(x):
        return tf.reduce_sum(tf.square(x))

    ########################################
    # V + W
    ########################################
    VW = V + tf.tensordot(OH_MSA, W, 2)

    # hamiltonian
    H = tf.reduce_sum(tf.multiply(OH_MSA, VW), axis=(1, 2))
    # local Z (parition function)
    Z = tf.reduce_sum(tf.reduce_logsumexp(VW, axis=2), axis=1)

    # Psuedo-Log-Likelihood
    PLL = H - Z

    # Regularization
    L2_V = 0.01 * L2(V)
    L2_W = 0.01 * L2(W) * 0.5 * (ncol - 1) * (states - 1)

    # loss function to minimize
    loss = -tf.reduce_sum(PLL * MSA_weights) / tf.reduce_sum(MSA_weights)
    loss = loss + (L2_V + L2_W) / msa["neff"]

    ##############################################################
    # MINIMIZE LOSS FUNCTION
    ##############################################################
    if opt_type == "adam":
        opt = opt_adam(loss, "adam", lr=opt_rate)

    # generate input/feed
    def feed(feed_all=False):
        if batch_size is None or feed_all:
            return {MSA: msa["msa"], MSA_weights: msa["weights"]}
        else:
            idx = np.random.randint(0, msa["nrow"], size=batch_size)
            return {MSA: msa["msa"][idx], MSA_weights: msa["weights"][idx]}

    # optimize!
    with tf.Session() as sess:
        # initialize variables V and W
        sess.run(tf.global_variables_initializer())

        # initialize V
        msa_cat = tf.keras.utils.to_categorical(msa["msa"], states)
        pseudo_count = 0.01 * np.log(msa["neff"])
        V_ini = np.log(np.sum(msa_cat.T * msa["weights"], -1).T + pseudo_count)
        V_ini = V_ini - np.mean(V_ini, -1, keepdims=True)
        sess.run(V.assign(V_ini))

        # compute loss across all data
        get_loss = lambda: round(sess.run(loss, feed(feed_all=True)) * msa["neff"], 2)
        print("starting", get_loss())

        if opt_type == "lbfgs":
            lbfgs = tf.contrib.opt.ScipyOptimizerInterface
            opt = lbfgs(loss, method="L-BFGS-B", options={'maxiter': opt_iter})
            opt.minimize(sess, feed(feed_all=True))

        if opt_type == "adam":
            for i in range(opt_iter):
                sess.run(opt, feed())
                if (i + 1) % int(opt_iter / 10) == 0:
                    print("iter", (i + 1), get_loss())

        # save the V and W parameters of the MRF
        V_ = sess.run(V)
        W_ = sess.run(W)

    # only return upper-right triangle of matrix (since it's symmetric)
    tri = np.triu_indices(ncol, 1)
    W_ = W_[tri[0], :, tri[1], :]

    mrf = {"v": V_,
           "w": W_,
           "v_idx": msa["v_idx"],
           "w_idx": msa["w_idx"]}

    return mrf


###################
def normalize(x):
    x = stats.boxcox(x - np.amin(x) + 1.0)[0]
    x_mean = np.mean(x)
    x_std = np.std(x)
    return ((x - x_mean) / x_std)


def get_mtx(mrf):
    '''get mtx given mrf'''

    # l2norm of 20x20 matrices (note: we ignore gaps)
    raw = np.sqrt(np.sum(np.square(mrf["w"][:, :-1, :-1]), (1, 2)))
    raw_sq = squareform(raw)

    # apc (average product correction)
    ap_sq = np.sum(raw_sq, 0, keepdims=True) * np.sum(raw_sq, 1, keepdims=True) / np.sum(raw_sq)
    apc = squareform(raw_sq - ap_sq, checks=False)

    mtx = {"i": mrf["w_idx"][:, 0],
           "j": mrf["w_idx"][:, 1],
           "raw": raw,
           "apc": apc,
           "zscore": normalize(apc)}
    return mtx


def plot_mtx(mtx, key="zscore", vmin=1, vmax=3):
    '''plot the mtx'''
    plt.figure(figsize=(5, 5))
    plt.imshow(squareform(mtx[key]), cmap='Blues', interpolation='none', vmin=vmin, vmax=vmax)
    plt.grid(False)
    plt.colorbar()
    plt.show()


def rm_lowercase(path, out):
    pattern = r'[a-z]+'
    with open(out, 'w') as f:
        for record in SeqIO.parse(path, 'fasta'):
            seq = re.sub(pattern, '', str(record.seq))
            f.write(f">{record.description}\n{seq}\n")


def parse(path):
    pattern = r'[a-z]+'
    names = []
    seqs = []
    for record in SeqIO.parse(path, 'fasta'):
        seq = re.sub(pattern, '', str(record.seq))
        names.append(record.description)
        seqs.append(seq)
    
    return names, seqs


if __name__ == '__main__':
    # ===============================================================================
    # PREP MSA
    # ===============================================================================
    # parse fasta
    # path = "task_eval/twintowers/"
    #
    # for file in os.listdir(path):
    #     # if "20#2W6K_1_A" not in file:
    #     #     continue
    #     if file[-1] == 'm':
    #         if os.path.exists(f"{path}/{file.replace('.a3m', '_mtx.npy')}"):
    #             continue
    #
    #         names, seqs = parse(f"{path}/{file}")
    #
    #         # process input sequences
    #         msa = mk_msa(seqs)
    #         np.save(f"{path}/{file.replace('.a3m', '_nogap')}", msa["v_idx"])
    #         # ===============================================================================
    #         # RUN GREMLIN
    #         # ===============================================================================
    #         # Note: the original GREMLIN uses the "lbfgs" optimizer which is EXTREMELY slow
    #         # in tensorflow. The modified adam optimizer is much faster, but may
    #         # require adjusting number of iterations (opt_iter) to converge to the same
    #         # solution. To switch back to the original, set opt_type="lbfgs".
    #         # ===============================================================================
    #         mrf = GREMLIN(msa)
    #
    #
    #         mtx = get_mtx(mrf)
    #         np.save(f"{path}/{file.replace('.a3m', '_mtx')}", mtx["zscore"])
    #         plot_mtx(mtx)
    
    # path = 'task_eval/query.faa'
    # records = list(SeqIO.parse(path, 'fasta'))
    # with open(path, 'w') as f:
    #     for record in records:
    #         f.write(f">{record.description}\n{str(record.seq).replace('-', '').upper()}\n")

    
    names, seqs = parse(f"task_eval/test.a3m")

    # process input sequences
    msa = mk_msa(seqs)
    np.save(f"task_eval/twintowers/TBM#T0872_twintowers_nogap", msa["v_idx"])
    mrf = GREMLIN(msa)


    mtx = get_mtx(mrf)
    np.save(f"task_eval/twintowers/TBM#T0872_twintowers_mtx", mtx["zscore"])
    plot_mtx(mtx)
    plt.figure(figsize=(5, 5))

    # v = np.load(f"task_eval/twintowers/40#2OCT_1_A_twintowers_mtx.npy")
    # gap = np.load(f"task_eval/hhblits/40#2OCT_1_A_nogap.npy")
    # print(gap.shape)
    # plt.imshow(squareform(v), cmap='Blues', interpolation='none', vmin=1, vmax=3)
    # plt.show()
    # for file in os.listdir("task_eval/twintowers"):
    #     # if not '90#1CF7' in file:
    #     #     continue
    #
    #     if "mtx.npy" in file:
    #         # if file != 'TBM#T0922_twintowers_mtx.npy':
    #         #     continue
    #         v = np.load(f"task_eval/twintowers/{file}")
    #         plt.title(file)
    #         plt.imshow(squareform(v), cmap='Blues', interpolation='none', vmin=1, vmax=3)
    #         plt.grid(False)
    #         plt.colorbar()
    #         plt.show()
    #         plt.cla()

    # def parse_a3m(path, out):
    #     pattern = r'[a-z]+'
    #     with open(out, 'w') as f:
    #         for record in SeqIO.parse(path, 'fasta'):
    #             f.write(f">{record.description}\n{re.sub(pattern, '', str(record.seq))}\n")
    # path = "task_eval/af2_hhblits"
    # for file in os.listdir(path):
    #     if 'hhblits.a3m' in file:
    #         file_path = f"{path}/{file}"
    #         out = f"{path}/{file.replace('hhblits.a3m', 'hhblits_formatted.a3m')}"
    #         parse_a3m(file_path, out)