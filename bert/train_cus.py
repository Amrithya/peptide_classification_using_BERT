import numpy as np

task_name = 'neoag_train'

# read data
seqs, labels = [], []
with open('./data/neoag_train.csv', 'r') as f:
    for line in f.readlines()[1:]:
        seq, label = line.strip().split(',')
        seqs.append(seq)
        labels.append(int(label))

MAX_LEN = max(map(len, seqs))

# convert to tokens
mapping = dict(zip(
    ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L',
    'A','G','V','E','S','I','K','R','D','T','P','N',
    'Q','F','Y','M','H','C','W'],
    range(30)
))

pos_data, neg_data = [], []
for i in range(len(seqs)):
    seq = [mapping[c] for c in seqs[i]] 
    seq.extend([0] * (MAX_LEN - len(seq)))  
    if labels[i] == 1:
        pos_data.append(seq)
    else:
        neg_data.append(seq)

pos_data = np.array(pos_data)
neg_data = np.array(neg_data)

np.savez(
    f'./data/{task_name}-positive.npz',
    arr_0=pos_data
)
np.savez(
    f'./data/{task_name}-negative.npz',
    arr_0=neg_data
)