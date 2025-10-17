import torch
from torch.utils.data import TensorDataset, DataLoader
import json
import params
import random
import numpy as np

def my_collate(batch):
    input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
    for log in batch:
        if log['knowledge_code']==[]:
            knowledge_emb = [1.0] * params.kn
        else:
            knowledge_emb = [0.] * params.kn
            for knowledge_code in log['knowledge_code']:
                knowledge_emb[knowledge_code-1] = 1.0
        y = log['score']
        input_stu_ids.append(log['user_id']-1)
        input_exer_ids.append(log['exer_id']-1)
        input_knowledge_embs.append(knowledge_emb)
        ys.append(y)

    return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

def CD_DL():
    with open('../data/{0}/{1}.json'.format(params.data_type, params.src)) as i_f:
        src_dataset = json.load(i_f)
    with open('../data/{0}/{1}.json'.format(params.data_type, params.tgt)) as i_f:
        tgt_dataset = json.load(i_f)
    src_DL = DataLoader(dataset=src_dataset, batch_size=128, shuffle=True, collate_fn=my_collate)
    tgt_DL = DataLoader(dataset=tgt_dataset, batch_size=128, shuffle=True, collate_fn=my_collate)
    return src_DL, tgt_DL

