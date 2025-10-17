import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion import Fusion


class Net(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        # self.directed_g = local_map['directed_g'].to(self.device)
        # self.undirected_g = local_map['undirected_g'].to(self.device)
        # self.k_from_e = local_map['k_from_e'].to(self.device)
        # self.e_from_k = local_map['e_from_k'].to(self.device)
        # self.u_from_e = local_map['u_from_e'].to(self.device)
        # self.e_from_u = local_map['e_from_u'].to(self.device) #全部放到GPU，几份图了，需要这一步吗

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim) #10000的onehot变为835熟练度
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim) #维度没变
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim) #835个onehot变为835困难度

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device) #也是835[0,1,....]
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device) #这个是10000个
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device) #835

        self.FusionLayer1 = Fusion(args, local_map) #这好像是独立的
        self.FusionLayer2 = Fusion(args, local_map)

        self.prednet_full1 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * args.knowledge_n, 1)

        # initialization
        for name, param in self.named_parameters(): #super里的named_parameters()参数
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device) #835
        exer_emb = self.exercise_emb(self.exer_index).to(self.device) #835
        kn_emb = self.knowledge_emb(self.k_index).to(self.device) #835
        # 所有嵌入已放入显存(还没用上载入id，是全部的)

        # Fusion layer 1
        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        # Fusion layer 2
        self.kn_emb2, self.exer_emb2, self.all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)
        #重点注意一下，怎么全融合了？******************************************************************************
        #这里可以加速，验证、测试时没有必要每次都聚合求全，想办法放到self里

        # get batch student data
        batch_stu_emb = self.all_stu_emb2[stu_id] # 32 123   256x835
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1]) #256x835x835

        # get batch exercise data
        batch_exer_emb = self.exer_emb2[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = self.kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], self.kn_emb2.shape[0], self.kn_emb2.shape[1]) #区别？256x835x835

        # Cognitive diagnosis
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2))) #到底是哪个维度？什么大小的tensor？256x835x835
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2))) #256x835x1670-->256x835x835
        o = torch.sigmoid(self.prednet_full3(preference - diff)) #最后256x835x1???

        #kn_r 256x835
        temp_out = o * kn_r.unsqueeze(2)
        sum_out = torch.sum(temp_out, dim = 1) #256x1??? 能力大过难度值 *代表点乘
        count_of_concept = torch.sum(kn_r, dim = 1).unsqueeze(1) #256x1
        output = sum_out / count_of_concept #平均超标值
        return output, temp_out.squeeze(2)

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper) # 为何拼接层也要正向？嵌入代表了什么？
        self.prednet_full3.apply(clipper)

    def pred(self, stu_id, exer_id, kn_r):
        batch_stu_emb = self.all_stu_emb2[stu_id] # 32 123   256x835
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1]) #256x835x835

        batch_exer_emb = self.exer_emb2[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        kn_vector = self.kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], self.kn_emb2.shape[0], self.kn_emb2.shape[1]) #区别？256x835x835

        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2))) #到底是哪个维度？什么大小的tensor？256x835x835
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2))) #256x835x1670-->256x835x835
        o = torch.sigmoid(self.prednet_full3(preference - diff)) #最后256x835x1???

        #kn_r 256x835
        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim = 1) #256x1??? 能力大过难度值 *代表点乘
        count_of_concept = torch.sum(kn_r, dim = 1).unsqueeze(1) #256x1
        output = sum_out / count_of_concept #平均超标值
        return output

    def get_emb(self):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        self.kn_emb2, self.exer_emb2, self.all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)

    def tl_temp(self, stu_id, exer_id, kn_r):
        batch_stu_emb = self.all_stu_emb2[stu_id]
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])

        batch_exer_emb = self.exer_emb2[exer_id]
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        kn_vector = self.kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], self.kn_emb2.shape[0], self.kn_emb2.shape[1])

        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        o = torch.sigmoid(self.prednet_full3(preference - diff))

        temp_out = o * kn_r.unsqueeze(2)
        return temp_out.squeeze(2)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)


class PredNet(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256

        super(PredNet, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)

        self.FusionLayer1 = Fusion(args, local_map)
        self.FusionLayer2 = Fusion(args, local_map)

        self.prednet_full1 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        self.prednet_full2 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        self.prednet_full3 = nn.Linear(1 * args.knowledge_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r):
        # get batch student data
        batch_stu_emb = self.all_stu_emb2[stu_id] # 32 123   256x835
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1]) #256x835x835

        # get batch exercise data
        batch_exer_emb = self.exer_emb2[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = self.kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], self.kn_emb2.shape[0], self.kn_emb2.shape[1]) #区别？256x835x835

        # Cognitive diagnosis
        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2))) #到底是哪个维度？什么大小的tensor？256x835x835
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2))) #256x835x1670-->256x835x835
        o = torch.sigmoid(self.prednet_full3(preference - diff)) #最后256x835x1???

        #kn_r 256x835
        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim = 1) #256x1??? 能力大过难度值 *代表点乘
        count_of_concept = torch.sum(kn_r, dim = 1).unsqueeze(1) #256x1
        output = sum_out / count_of_concept #平均超标值
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper) # 为何拼接层也要正向？嵌入代表了什么？
        self.prednet_full3.apply(clipper)

    def get_emb(self):
        all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        kn_emb1, exer_emb1, all_stu_emb1 = self.FusionLayer1(kn_emb, exer_emb, all_stu_emb)
        self.kn_emb2, self.exer_emb2, self.all_stu_emb2 = self.FusionLayer2(kn_emb1, exer_emb1, all_stu_emb1)
