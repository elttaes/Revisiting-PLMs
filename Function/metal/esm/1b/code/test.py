from Bio import SeqIO
import gzip
import random



def read_fasta(fn_fasta):
    aa = set(['R', 'X', 'S', 'G', 'W', 'I', 'Q', 'A', 'T', 'V', 'K', 'Y', 'C', 'N', 'L', 'F', 'D', 'M', 'P', 'H', 'E'])
    prot2seq = {}
    if fn_fasta.endswith('gz'):
        handle = gzip.open(fn_fasta, "rt")
    else:
        handle = open(fn_fasta, "rt")

    for record in SeqIO.parse(handle, "fasta"):
        seq = str(record.seq)
        prot = record.id
        pdb, chain = prot.split('_') if '_' in prot else prot.split('-')
        prot = pdb.upper() + '-' + chain
        if len(seq) >= 60 and len(seq) <= 1000:
            if len((set(seq).difference(aa))) == 0:
                prot2seq[prot] = seq

    return prot2seq


prot2seq = read_fasta('../data/pdb_seqres.txt')

with open('../data/label.txt', 'r') as f:
    data = f.readlines()

ids = []
for line in data:
    id = line.split('\t')[0]
    ids.append(id)
random.shuffle(ids)

new_data = []
n = 0
for id in ids:
    if n >= 3000:
        break
    if id.upper() + '-A' in prot2seq.keys():
        new_data.append(id + '\t' + prot2seq[id.upper() + '-A'] + '\t' + '1' + '\n')
        n += 1

for key in prot2seq.keys():
    if key not in ids:
        new_data.append(key.split('-')[0].lower() + '\t' + prot2seq[key] + '\t' + '0' + '\n')
        if len(new_data) >= 6000:
            break

random.shuffle(new_data)
with open('../../GNN_supervised_learning/data/data_6000.txt', 'a') as f:
    for line in new_data:
        f.write(line)
