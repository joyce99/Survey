import torch
import numpy as np
import json
import torch.nn.functional as F


# data_file = '../data/junyi/log_data.json'
# u_l = []
# e_l=[]
# k_l=[]
# temp_list = []
# with open(data_file, encoding='utf8') as i_f:
#     data = json.load(i_f)
# k_from_e = ''
# e_from_k = ''
# for stu in data:
#     u_l.append(stu['user_id'])
#     for log in stu['logs']:
#         e_l.append(log['exer_id'] - 1)
#         for k in log['knowledge_code']:
#             k_l.append(k-1)
#
# u_s=set(u_l) #10000 1-10001
# e_s=set(e_l) #706
# k_s=set(k_l) #706


# with open('../data/mooper/idx1_slice.json', encoding='utf8') as i_f:
#     data = json.load(i_f)
# with open('../data/mooper/A.json', encoding='utf8') as i_f:
#     n_data = json.load(i_f)

# count_list=[0]*len(n_data)
# for d in data:
#     for i in range(len(n_data)):
#         if d == n_data[i]:
#             count_list[i] += 1
#
# count = 0
# for i in count_list:
#     count+=i


# s1= torch.rand((2,3))
# s2 = s1.repeat(1,3)
# s3 = s2.reshape(2,3,3)
#
# k1 = torch.rand((3,3))
# k2 = k1.repeat(2,1)
# k3 = k2.reshape(2,3,3)


# import numpy as np
# import torch
# from torch import nn
# from torch.nn import init
#
#
# class ScaledDotProductAttention(nn.Module):
#     def __init__(self, d_model, d_k, d_v, h, dropout=.1):
#         super(ScaledDotProductAttention, self).__init__()
#         self.fc_q = nn.Linear(d_model, h * d_k) #512 - 512*8
#         self.fc_k = nn.Linear(d_model, h * d_k) #512 - 512*8
#         self.fc_v = nn.Linear(d_model, h * d_v) #512 - 512*8
#         self.fc_o = nn.Linear(h * d_v, d_model) #512*8 - 512
#         self.dropout = nn.Dropout(dropout)
#
#         self.d_model = d_model
#         self.d_k = d_k
#         self.d_v = d_v
#         self.h = h
#
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
#         '''
#         Computes
#         :param queries: Queries (b_s, nq, d_model)
#         :param keys: Keys (b_s, nk, d_model)
#         :param values: Values (b_s, nk, d_model)
#         :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
#         :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
#         :return:
#         '''
#         b_s, nq = queries.shape[:2]
#         nk = keys.shape[1]
#
#         q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
#         k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
#         v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
#
#         att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
#         if attention_weights is not None:
#             att = att * attention_weights
#         if attention_mask is not None:
#             att = att.masked_fill(attention_mask, -np.inf)
#         att = torch.softmax(att, -1)
#         att = self.dropout(att)
#
#         out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
#         out = self.fc_o(out)  # (b_s, nq, d_model)
#         return out
#
#
# if __name__ == '__main__':
#     input = torch.randn(50, 49, 512)
#     sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
#     output = sa(input, input, input)
#     print(output.shape)

import json
import random

# random.seed(0)
# exer_index = list(range(1,17747))
# random.shuffle(exer_index)
# exer_1 = exer_index[:int(0.5*len(exer_index))]
# exer_2 = exer_index[int(0.5*len(exer_index)):]
#
#
# with open('../data/assist09/log_data.json', encoding='utf8') as i_f:
#     stus = json.load(i_f)
# # 1. delete students who have fewer than min_log response logs
# stu_i = 0
# while stu_i < len(stus):
#     if stus[stu_i]['log_num'] < 15:
#         del stus[stu_i]
#         stu_i -= 1
#     stu_i += 1
# # 2. divide dataset into train_set, val_set and test_set
# train_slice, train_set, val_set, test_set = [], [], [], []
# for stu in stus:
#     user_id = stu['user_id']
#     stu_train = {'user_id': user_id}
#     stu_test = {'user_id': user_id}
#     logs1, logs2 = [], []
#     for log in stu['logs']:
#         if log["exer_id"] in exer_1:
#             logs1.append(log)
#         else:
#             logs2.append(log)
#     stu_train['log_num'] = len(logs1)
#     stu_train['logs'] = logs1
#     stu_test['log_num'] = len(logs2)
#     stu_test['logs'] = logs2
#     train_slice.append(stu_train)
#     test_set.append(stu_test)
#     # shuffle logs in train_slice together, get train_set
#     for log in stu_train['logs']:
#         train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
#                           'knowledge_code': log['knowledge_code']})
# # random.shuffle(train_set)
# with open('../data/assist09/train_slice.json', 'w', encoding='utf8') as output_file:
#     json.dump(train_slice, output_file, indent=4, ensure_ascii=False)
# with open('../data/assist09/train_set.json', 'w', encoding='utf8') as output_file:
#     json.dump(train_set, output_file, indent=4, ensure_ascii=False)
# with open('../data/assist09/test_set.json', 'w', encoding='utf8') as output_file:
#     json.dump(test_set, output_file, indent=4, ensure_ascii=False)

# with open('../data/assist09/test_set.json', encoding='utf8') as i_f:
#     stus = json.load(i_f)
# tgt_set=[]
# for stu in stus:
#     user_id = stu['user_id']
#     stu_train = {'user_id': user_id}
#     logs = []
#     for log in stu['logs']:
#         logs.append(log)
#     stu_train['logs'] = logs
#     for log in stu_train['logs']:
#         tgt_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
#                       'knowledge_code': log['knowledge_code']})
#
# with open('../data/assist09/tgt.json', 'w', encoding='utf8') as output_file:
#     json.dump(tgt_set, output_file, indent=4, ensure_ascii=False)

# def ST2slice():
#     with open('../data/assist09/test_set.json', encoding='utf8') as i_f:
#         stus = json.load(i_f)
#     # 1. delete students who have fewer than min_log response logs
#     stu_i = 0
#     while stu_i < len(stus):
#         if stus[stu_i]['log_num'] < 5:
#             del stus[stu_i]
#             stu_i -= 1
#         stu_i += 1
#     # 2. divide dataset into train_set, val_set and test_set
#     train_slice, train_set, test_slice, test_set = [], [], [], []
#     for stu in stus:
#         user_id = stu['user_id']
#         stu_train = {'user_id': user_id}
#         stu_test = {'user_id': user_id}
#         train_size = int(stu['log_num'] * 0.8)
#         test_size = stu['log_num'] - train_size
#         logs = []
#         for log in stu['logs']:
#             logs.append(log)
#         random.shuffle(logs)
#         stu_train['log_num'] = train_size
#         stu_train['logs'] = logs[:train_size]
#         stu_test['log_num'] = test_size
#         stu_test['logs'] = logs[-test_size:]
#         train_slice.append(stu_train)
#         test_slice.append(stu_test)
#         # shuffle logs in train_slice together, get train_set
#         for log in stu_train['logs']:
#             train_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
#                               'knowledge_code': log['knowledge_code']})
#         for log in stu_test['logs']:
#             test_set.append({'user_id': user_id, 'exer_id': log['exer_id'], 'score': log['score'],
#                               'knowledge_code': log['knowledge_code']})
#     with open('../data/assist09/tgt_train.json', 'w', encoding='utf8') as output_file:
#         json.dump(train_set, output_file, indent=4, ensure_ascii=False)
#     with open('../data/assist09/tgt_test.json', 'w', encoding='utf8') as output_file:
#         json.dump(test_set, output_file, indent=4, ensure_ascii=False)

def attn(self, queries, keys, values, attention_mask=None, attention_weights=None):
    b_s, nq = queries.shape[:2]
    nk = keys.shape[1]

    q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
    k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
    v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

    att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
    if attention_weights is not None:
        att = att * attention_weights
    if attention_mask is not None:
        att = att.masked_fill(attention_mask, -np.inf)
    att = torch.softmax(att, -1)
    att = self.dropout(att)

    out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
    out = self.fc_o(out)  # (b_s, nq, d_model)
    return out


pass
