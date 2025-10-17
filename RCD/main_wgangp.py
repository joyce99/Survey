import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from data_loader import TrainDataLoader, PsliceDL
from model import Net
from utils import CommonArgParser, construct_local_map
import torch.autograd as autograd


DIM = 512
use_cuda = True
BATCH_SIZE = 256
LAMBDA = 0.1

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        main = nn.Sequential(
            nn.Linear(123, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Linear(DIM, 1),
        )
        self.main = main

    def forward(self, inputs):
        output = self.main(inputs)
        return output.view(-1)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if use_cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty



def train(args, local_map, load_epoch = 0):
    src_dl = TrainDataLoader('src')
    tgt_dl = TrainDataLoader('tgt')
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    net = Net(args, local_map)
    if load_epoch != 0:
        load_snapshot(net, 'model/model_epoch'+str(load_epoch))
    net = net.to(device)
    print(net)

    netD = Discriminator()
    netD.apply(weights_init)
    netD = netD.cuda()
    print(netD)
    print('training model...')
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    one = torch.FloatTensor([1])
    mone = one * -1  # 反向
    if use_cuda:
        one = one.cuda()
        mone = mone.cuda()
    loss_function = nn.NLLLoss() #交叉熵去掉softmax、log后的最后对比标签求均值

    for epoch in range(args.epoch_n):
        src_dl.reset()
        tgt_dl.reset()
        running_loss = 0.0
        batch_count = 0
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        while not (src_dl.is_end() or tgt_dl.is_end()):
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = src_dl.next_batch()
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)
            optimizer.zero_grad()
            tgt_stu_ids, tgt_exer_ids, tgt_knowledge_embs, _ = tgt_dl.next_batch()
            tgt_stu_ids, tgt_exer_ids, tgt_knowledge_embs = tgt_stu_ids.to(device), tgt_exer_ids.to(device), tgt_knowledge_embs.to(device)
            optimizer.zero_grad()
            output_1, src_temp = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            _, tgt_temp = net.pred(tgt_stu_ids, tgt_exer_ids, tgt_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)


            netD.zero_grad()

            # train with real
            D_real = netD(src_temp)
            D_real = D_real.mean()  # 均值？期望
            D_real.backward(mone.mean(), retain_graph=True)  # 梯度反转？求导吗 第一次backward

            # train with fake
            D_fake = netD(tgt_temp)
            D_fake = D_fake.mean()
            D_fake.backward(one.mean(), retain_graph=True) #第二次backward

            # train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, src_temp, tgt_temp)
            gradient_penalty.backward() #第三次backward

            D_cost = D_fake - D_real + gradient_penalty
            Wasserstein_D = D_real - D_fake
            optimizerD.step()

            for p in netD.parameters():
                p.requires_grad = False

            G = netD(tgt_temp)
            G = G.mean()
            G.backward(mone.mean(), retain_graph=True) #第一次backward

            loss1 = loss_function(torch.log(output+1e-10), labels)

            loss1.backward() #第二次backward
            optimizer.step()
            net.apply_clipper()

            loss = loss1 + D_cost
            running_loss += loss.item()
            if batch_count % 20 == 19:
                print('[%d, %5d] loss: %.3f' % (epoch + 1 + load_epoch, batch_count + 1, running_loss / 20))
                running_loss = 0.0

        save_snapshot(net, 'model/model_epoch' + str(epoch + 1 + load_epoch))
        rmse, auc = predict(args, net, epoch + load_epoch)
        pass

def predict(args, net, epoch):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = PsliceDL()
    print('predicting model...')
    data_loader.reset()
    net.eval()
    net.get_emb()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    while not data_loader.is_end():
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = data_loader.next_batch()
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output, _ = net.pred(input_stu_ids, input_exer_ids, input_knowledge_embs)
        output = output.view(-1)

        for i in range(len(labels)):
            if (labels[i] == 1 and output[i] > 0.5) or (labels[i] == 0 and output[i] < 0.5):
                correct_count += 1
        exer_count += len(labels)
        pred_all += output.to(torch.device('cpu')).tolist()
        label_all += labels.to(torch.device('cpu')).tolist()

    pred_all = np.array(pred_all)
    label_all = np.array(label_all)
    accuracy = correct_count / exer_count
    rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
    auc = roc_auc_score(label_all, pred_all)
    print('epoch= %d, accuracy= %f, rmse= %f, auc= %f' % (epoch+1, accuracy, rmse, auc))
    with open('result/ncd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f\n' % (epoch+1, accuracy, rmse, auc))

    return rmse, auc

def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()

def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()

if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    train(args, construct_local_map(args), load_epoch=0)
