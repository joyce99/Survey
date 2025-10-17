import json
import torch
import random
import params


class TrainDataLoader(object):
    '''
    data loader for training
    '''
    def __init__(self, dtpye='src'):
        self.batch_size = 128
        self.ptr = 0
        self.data = []

        if dtpye == 'tgt':
            data_file = '../data/{0}/{1}.json'.format(params.data_type, params.tgt)
        else:  # src
            data_file = '../data/{0}/{1}.json'.format(params.data_type, params.src)
        # config_file = '../data/assist09/config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        # with open(config_file) as i_f:
        #     i_f.readline()
        student_n, exercise_n, knowledge_n = params.un, params.en, params.kn
        self.knowledge_dim = int(knowledge_n)
        # self.student_dim = int(student_n)
        # self.exercise_dim = int(exercise_n)

    def next_batch(self):
        if self.is_end():
            return None, None, None, None
        input_stu_ids, input_exer_ids, input_knowledge_embs, ys = [], [], [], []
        for count in range(self.batch_size):
            log = self.data[self.ptr + count]
            if log['knowledge_code']==[]:
                knowledge_emb = [1.0] * self.knowledge_dim
            else:
                knowledge_emb = [0.] * self.knowledge_dim
                for knowledge_code in log['knowledge_code']:
                    knowledge_emb[knowledge_code-1] = 1.0
            y = log['score']
            input_stu_ids.append(log['user_id']-1)
            input_exer_ids.append(log['exer_id']-1)
            input_knowledge_embs.append(knowledge_emb)
            ys.append(y)

        self.ptr += self.batch_size
        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowledge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.ptr + self.batch_size > len(self.data):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0
        random.shuffle(self.data)


class PsliceDL(object):
    def __init__(self, batchsize=128):
        self.batch_size = batchsize
        self.ptr = 0
        self.flag = False

        data_file = '../data/{0}/{1}.json'.format(params.data_type, params.tgt)
        # config_file = '../data/assist09/config.txt'
        with open(data_file, encoding='utf8') as i_f:
            self.data = json.load(i_f)
        # with open(config_file) as i_f:
        #     i_f.readline()
        #     _, _, knowledge_n = i_f.readline().split(',')
        self.knowledge_dim = params.kn

    def next_batch(self):
        input_stu_ids, input_exer_ids, input_knowedge_embs, ys = [], [], [], []
        last_len = self.is_last()
        if last_len == 0:
            for count in range(self.batch_size):
                log = self.data[self.ptr + count]
                knowledge_emb = [0.] * self.knowledge_dim
                if log['knowledge_code']==[]:
                    knowledge_emb = [1.0] * self.knowledge_dim
                else:
                    for knowledge_code in log['knowledge_code']:
                        knowledge_emb[knowledge_code-1] = 1.0
                y = log['score']
                input_stu_ids.append(log['user_id']-1)
                input_exer_ids.append(log['exer_id']-1)
                input_knowedge_embs.append(knowledge_emb)
                ys.append(y)
            self.ptr += self.batch_size
        else:
            for count in range(last_len):
                log = self.data[self.ptr + count]
                knowledge_emb = [0.] * self.knowledge_dim
                if log['knowledge_code']==[]:
                    knowledge_emb = [1.0] * self.knowledge_dim
                else:
                    for knowledge_code in log['knowledge_code']:
                        knowledge_emb[knowledge_code-1] = 1.0
                y = log['score']
                input_stu_ids.append(log['user_id']-1)
                input_exer_ids.append(log['exer_id']-1)
                input_knowedge_embs.append(knowledge_emb)
                ys.append(y)
            self.flag = True

        return torch.LongTensor(input_stu_ids), torch.LongTensor(input_exer_ids), torch.Tensor(input_knowedge_embs), torch.LongTensor(ys)

    def is_end(self):
        if self.flag == True:
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

    def is_last(self):
        if self.ptr + self.batch_size >= len(self.data):
            len_last = len(self.data) - self.ptr
            return len_last
        else:
            return 0


