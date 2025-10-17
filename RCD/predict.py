import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from data_loader import PsliceDL
from model import Net
from utils import CommonArgParser, construct_local_map



def predict(args, net):
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
    data_loader = PsliceDL()
    print('predicting model...')
    data_loader.reset()
    net.eval()

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
    print('accuracy= %f, rmse= %f, auc= %f' % (accuracy, rmse, auc))
    with open('result/ncd_model_val.txt', 'a', encoding='utf8') as f:
        f.write('accuracy= %f, rmse= %f, auc= %f\n' % (accuracy, rmse, auc))

    return rmse, auc


def load_snapshot(model, filename):
    f = open(filename, 'rb')
    model.load_state_dict(torch.load(f, map_location=lambda s, loc: s))
    f.close()


if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    local_map = construct_local_map(args)
    net = Net(args, local_map)

    load_snapshot(net, 'model/model_epoch1')
    net = net.to('cuda:0')
    net.eval()
    net.get_emb()
    predict(args, net)



