import torch
import torch.nn as nn
from fusion import Fusion


class Net(nn.Module):
    def __init__(self, args, local_map, flag=0):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256

        super(Net, self).__init__()

        # network structure
        self.student_q = nn.Embedding(self.emb_num, self.knowledge_dim)
        self.exercise_k = nn.Embedding(self.exer_n, 1)
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim)
        self.knowledge_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)
        self.exercise_emb = nn.Embedding(self.exer_n, self.knowledge_dim)

        self.k_index = torch.LongTensor(list(range(self.stu_dim))).to(self.device)
        self.stu_index = torch.LongTensor(list(range(self.emb_num))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)

        self.FusionLayer1 = Fusion(args, local_map, flag=flag)
        self.FusionLayer2 = Fusion(args, local_map, flag=flag)

        self.prednet_full1 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(2 * args.knowledge_n, args.knowledge_n, bias=False)
        # self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(1 * args.knowledge_n, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, exer_id, kn_r):
        self.get_emb()
        output = self.pred(stu_id, exer_id, kn_r)
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def pred(self, stu_id, exer_id, kn_r):
        batch_stu_emb = self.all_stu_emb[stu_id]
        batch_exer_emb = self.exer_emb[exer_id]
        stu_q = self.student_q(stu_id)
        exer_k = self.exercise_k(exer_id)
        disc_1 = torch.sigmoid(exer_k)
        # batch_disc = disc_1.repeat(1, self.kn_emb.shape[0]).reshape(batch_stu_emb.shape[0], self.kn_emb.shape[0], self.kn_emb.shape[1])
        batch_disc = disc_1.repeat(self.kn_emb.shape[1], self.kn_emb.shape[0]).reshape(batch_stu_emb.shape[0], self.kn_emb.shape[0], self.kn_emb.shape[1])

        #origin
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0], batch_stu_emb.shape[1], batch_stu_emb.shape[1])
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0], batch_exer_emb.shape[1], batch_exer_emb.shape[1])
        kn_vector = self.kn_emb.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], self.kn_emb.shape[0], self.kn_emb.shape[1])

        preference = torch.sigmoid(self.prednet_full1(torch.cat((batch_stu_vector, kn_vector), dim=2)))
        diff = torch.sigmoid(self.prednet_full2(torch.cat((batch_exer_vector, kn_vector), dim=2)))
        input_x = (preference - diff) * batch_disc
        # input_x = preference - diff

        o = torch.sigmoid(self.prednet_full3(input_x))

        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim = 1)
        count_of_concept = torch.sum(kn_r, dim = 1).unsqueeze(1)
        output = sum_out / count_of_concept
        return output

    def get_emb(self):
        self.all_stu_emb = self.student_emb(self.stu_index).to(self.device)
        self.exer_emb = self.exercise_emb(self.exer_index).to(self.device)
        self.kn_emb = self.knowledge_emb(self.k_index).to(self.device)

        self.kn_emb, self.exer_emb, self.all_stu_emb = self.FusionLayer1(self.kn_emb, self.exer_emb, self.all_stu_emb)
        self.kn_emb, self.exer_emb, self.all_stu_emb = self.FusionLayer2(self.kn_emb, self.exer_emb, self.all_stu_emb)


class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

