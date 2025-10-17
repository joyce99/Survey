import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphLayer import GraphLayer


class Fusion(nn.Module):
    def __init__(self, args, local_map, flag):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n  # 835
        self.exer_n = args.exer_n  # 835
        self.emb_num = args.student_n  # 10000
        self.stu_dim = self.knowledge_dim  # 835
        self.flag = flag
        # graph structure
        self.k_from_e = local_map['k_from_e'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)
        self.u_from_e = local_map['u_from_e'].to(self.device)
        self.e_from_u = local_map['e_from_u'].to(self.device) #dgl图

        super(Fusion, self).__init__()

        self.k_from_e = GraphLayer(self.k_from_e, args.knowledge_n, args.knowledge_n)  # src: e #这是一个class
        self.e_from_k = GraphLayer(self.e_from_k, args.knowledge_n, args.knowledge_n)  # src: k

        self.u_from_e = GraphLayer(self.u_from_e, args.knowledge_n, args.knowledge_n)  # src: e
        self.e_from_u = GraphLayer(self.e_from_u, args.knowledge_n, args.knowledge_n)  # src: u

        self.e_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.e_attn_fc2 = nn.Linear(2 * args.knowledge_n, 1, bias=True)  # 练习注意力？

    def forward(self, kn_emb, exer_emb, all_stu_emb):
        if self.flag == 0:
            e_k_graph = torch.cat((exer_emb, kn_emb), dim=0) #693+377？1070x835
            k_from_e_graph = self.k_from_e(e_k_graph)
            e_from_k_graph = self.e_from_k(e_k_graph) #k、e间


            e_u_graph = torch.cat((exer_emb, all_stu_emb), dim=0) #835+835
            u_from_e_graph = self.u_from_e(e_u_graph)
            e_from_u_graph = self.e_from_u(e_u_graph) #u、e间

            # update concepts
            A = kn_emb
            D = k_from_e_graph[self.exer_n:] #练习传来的吗？后835？
            kn_emb = A + D

            # updated exercises
            A = exer_emb
            B = e_from_k_graph[0: self.exer_n] #取前835
            C = e_from_u_graph[0: self.exer_n]
            concat_e_1 = torch.cat([A, B], dim=1)
            concat_e_2 = torch.cat([A, C], dim=1)
            score1 = self.e_attn_fc1(concat_e_1)
            score2 = self.e_attn_fc2(concat_e_2)
            score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)  # dim = 1, 按行SoftMax, 行和为1
            exer_emb = exer_emb + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C

            # updated students
            all_stu_emb = all_stu_emb + u_from_e_graph[self.exer_n:] #也是后835

            return kn_emb, exer_emb, all_stu_emb
        else:
            e_k_graph = torch.cat((exer_emb, kn_emb), dim=0) #693+377？1070x835
            k_from_e_graph = self.k_from_e(e_k_graph)
            e_from_k_graph = self.e_from_k(e_k_graph) #k、e间

            # update concepts
            A = kn_emb
            D = k_from_e_graph[self.exer_n:] #练习传来的吗？后835？
            kn_emb = A + D

            # updated exercises
            A = exer_emb
            B = e_from_k_graph[0: self.exer_n] #取前835
            exer_emb = A + B

            # updated students
            all_stu_emb = all_stu_emb

            return kn_emb, exer_emb, all_stu_emb
