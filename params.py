import json

dataset = 'data/assist09/'
# dataset = 'data/mooper/'
# dataset = 'data/CSEDM-F/'
# dataset = 'data/junyi/'
# dataset = 'data/math1/'
# dataset = 'data/NIPS20/'
# dataset = 'data/PISA2015/'
# dataset = 'data/MOOCRadar-middle/'
# dataset = 'data/Q-free/mooper/'
# dataset = 'data/Q-free/mopper_add/'
batch_size = 128
lr = 0.002
epoch = 100

# src = dataset + 'enhanced_train_d_updated_unlimited.json'
# tgt = dataset + 'enhanced_val_d_updated_unlimited.json'
src = dataset + 'train.json'
tgt = dataset + 'val.json'

test = tgt
all = dataset + 'slice_d.json'

with open(dataset + 'config.txt') as f:
    f.readline()
    un, en, kn = f.readline().split(',')
    un, en, kn = int(un), int(en), int(kn)
latent_dim = kn



kn_select = 21
pass
