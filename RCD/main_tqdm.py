import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score
from dataloader_tqdm import CD_DL
from model import Net
from tqdm import tqdm
import sys
from utils import CommonArgParser, construct_local_map

src, tgt = CD_DL()

def train(args, local_map, load_epoch = 0):
    src_dl = src
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    net = Net(args, local_map)
    optimizer = optim.Adam(net.parameters(), lr=0.002)
    if load_epoch != 0:
        load_snapshot(net, 'model/model_epoch'+str(load_epoch))
    net = net.to(device)
    print(net)
    print('training model...')

    loss_function = nn.NLLLoss() #交叉熵去掉softmax、log后的最后对比标签求均值
    for epoch in range(args.epoch_n):
        running_loss = 0.0
        batch_count = 0
        for batch_data in tqdm(src_dl, "Epoch %s" % epoch, file=sys.stdout):
            batch_count += 1
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = batch_data
            input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(device), input_knowledge_embs.to(device), labels.to(device)

            optimizer.zero_grad()
            output_1 = net.forward(input_stu_ids, input_exer_ids, input_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)

            loss = loss_function(torch.log(output+1e-10), labels)
            loss.backward()
            optimizer.step()
            net.apply_clipper()

            running_loss += loss.item()
            if batch_count % 20 == 19:
                print('[%d, %5d] loss: %.3f' % (epoch + 1 + load_epoch, batch_count + 1, running_loss / 20))
                running_loss = 0.0

        # save_snapshot(net, 'model/model_epoch' + str(epoch + 1 + load_epoch))
        rmse, auc = predict(args, net, epoch + load_epoch)
        pass


def predict(args, net, epoch):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = tgt
    print('predicting model...')
    data_loader.reset()
    net.eval()
    net.get_emb()

    correct_count, exer_count = 0, 0
    batch_count, batch_avg_loss = 0, 0.0
    pred_all, label_all = [], []
    for batch_data in tqdm(data_loader, "Evaluating", file=sys.stdout):
        batch_count += 1
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = batch_data
        input_stu_ids, input_exer_ids, input_knowledge_embs, labels = input_stu_ids.to(device), input_exer_ids.to(
            device), input_knowledge_embs.to(device), labels.to(device)
        output = net.pred(input_stu_ids, input_exer_ids, input_knowledge_embs) #没有forward
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
    with open('result/RCD_val.txt', 'a', encoding='utf8') as f:
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
