# -*- coding: utf-8 -*-
# @Time    : 2022/6/14 15:07
# @Author  : CcQun
# @Email   : 13698603020@163.com
# @File    : main(semisup).py
# @Software: PyCharm
# @Note    :
import sys
import os.path as osp

dirname = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(dirname, '..'))
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from Main.pargs import pargs
from Main.dataset import WeiboDataset, WeiboFTDataset, PreDataLoader
from Main.word2vec import Embedding, collect_sentences, train_word2vec
from Main.sort import sort_weibo_dataset, sort_weibo_self_dataset, sort_weibo_2class_dataset
from Main.model import ResGCN_graphcl
from Main.utils import create_log_dict_semisup, write_log, write_json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def semisup_train(unsup_train_loader, train_loader, model, optimizer, device, gamma_joao, lamda):
    model.train()
    total_loss = 0

    aug_prob_unsup = unsup_train_loader.dataset.aug_prob
    n_aug_unsup = np.random.choice(25, 1, p=aug_prob_unsup)[0]
    n_aug1, n_aug2 = n_aug_unsup // 5, n_aug_unsup % 5

    aug_prob_train = train_loader.dataset.aug_prob
    n_aug_train = np.random.choice(25, 1, p=aug_prob_train)[0]
    n_aug3, n_aug4 = n_aug_train // 5, n_aug_train % 5

    # iter_train_loader = iter(train_loader)
    # for _, data1, data2 in unsup_train_loader:
    #     try:
    #         data, data3, data4 = next(iter_train_loader)
    #     except StopIteration:
    #         iter_train_loader = iter(train_loader)
    #         data, data3, data4 = next(iter_train_loader)

    for data_tup1, data_tup2 in zip(train_loader, unsup_train_loader):
        data, data1, data2 = data_tup1
        _, data3, data4 = data_tup2

        optimizer.zero_grad()
        data = data.to(device)
        data1 = data1.to(device)
        data2 = data2.to(device)
        data3 = data3.to(device)
        data4 = data4.to(device)
        out = model(data)
        out1 = model.forward_graphcl(data1, n_aug1)
        out2 = model.forward_graphcl(data2, n_aug2)
        out3 = model.forward_graphcl(data3, n_aug3)
        out4 = model.forward_graphcl(data4, n_aug4)
        loss = F.binary_cross_entropy(out, data.y.to(torch.float32)) + \
               model.loss_graphcl(out1, out2) * lamda + \
               model.loss_graphcl(out3, out4) * lamda
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data1.num_graphs

    aug_prob_unsup = joao(unsup_train_loader, model, gamma_joao)
    aug_prob_train = joao(train_loader, model, gamma_joao)
    return total_loss / len(unsup_train_loader.dataset), aug_prob_unsup, aug_prob_train


def test(model, dataloader, device, mode='test_or_val'):
    model.eval()
    error = 0

    y_true = []
    y_pred = []
    for data in dataloader:
        if mode == 'train':
            data = data[0]
        data = data.to(device)
        pred = model(data)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        error += F.binary_cross_entropy(pred, data.y.to(torch.float32)).item() * data.num_graphs
        y_true += data.y.tolist()
        y_pred += pred.tolist()
    acc = accuracy_score(y_true, y_pred)
    prec = [precision_score(y_true, y_pred, pos_label=1, average='binary'),
            precision_score(y_true, y_pred, pos_label=0, average='binary')]
    rec = [recall_score(y_true, y_pred, pos_label=1, average='binary'),
           recall_score(y_true, y_pred, pos_label=0, average='binary')]
    f1 = [f1_score(y_true, y_pred, pos_label=1, average='binary'),
          f1_score(y_true, y_pred, pos_label=0, average='binary')]
    return error / len(dataloader.dataset), acc, prec, rec, f1


def test_and_log(model, val_loader, test_loader, device, epoch, lr, loss, train_acc, log_record):
    val_error, val_acc, val_prec, val_rec, val_f1 = test(model, val_loader, device)
    test_error, test_acc, test_prec, test_rec, test_f1 = test(model, test_loader, device)
    log_info = 'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation BCE: {:.7f}, Test BCE: {:.7f}, Train ACC: {:.3f}, Validation ACC: {:.3f}, Test ACC: {:.3f}, Test PREC(T/F): {:.3f}/{:.3f}, Test REC(T/F): {:.3f}/{:.3f}, Test F1(T/F): {:.3f}/{:.3f}' \
        .format(epoch, lr, loss, val_error, test_error, train_acc, val_acc, test_acc, test_prec[0], test_prec[1],
                test_rec[0],
                test_rec[1], test_f1[0], test_f1[1])

    log_record['val accs'].append(round(val_acc, 3))
    log_record['test accs'].append(round(test_acc, 3))
    log_record['test prec T'].append(round(test_prec[0], 3))
    log_record['test prec F'].append(round(test_prec[1], 3))
    log_record['test rec T'].append(round(test_rec[0], 3))
    log_record['test rec F'].append(round(test_rec[1], 3))
    log_record['test f1 T'].append(round(test_f1[0], 3))
    log_record['test f1 F'].append(round(test_f1[1], 3))
    return val_error, log_info, log_record


def joao(dataloader, model, gamma_joao):
    aug_prob = dataloader.dataset.aug_prob
    # calculate augmentation loss
    loss_aug = np.zeros(25)

    for n in range(25):
        _aug_prob = np.zeros(25)
        _aug_prob[n] = 1
        dataloader.dataset.set_aug_prob(_aug_prob)

        n_aug1, n_aug2 = n // 5, n % 5

        count, count_stop = 0, len(dataloader.dataset) // (
                dataloader.batch_size * 10) + 1  # for efficiency, we only use around 10% of data to estimate the loss
        with torch.no_grad():
            for _, data1, data2 in dataloader:
                data1 = data1.to(device)
                data2 = data2.to(device)
                out1 = model.forward_graphcl(data1, n_aug1)
                out2 = model.forward_graphcl(data2, n_aug2)
                loss = model.loss_graphcl(out1, out2)
                loss_aug[n] += loss.item() * data1.num_graphs
                count += 1
                if count == count_stop:
                    break
        loss_aug[n] /= (count * dataloader.batch_size)

    # view selection, projected gradient descent, reference: https://arxiv.org/abs/1906.03563
    beta = 1
    gamma = gamma_joao

    b = aug_prob + beta * (loss_aug - gamma * (aug_prob - 1 / 25))
    mu_min, mu_max = b.min() - 1 / 25, b.max() - 1 / 25
    mu = (mu_min + mu_max) / 2

    # bisection method
    while abs(np.maximum(b - mu, 0).sum() - 1) > 1e-2:
        if np.maximum(b - mu, 0).sum() > 1:
            mu_min = mu
        else:
            mu_max = mu
        mu = (mu_min + mu_max) / 2

    aug_prob = np.maximum(b - mu, 0)
    aug_prob /= aug_prob.sum()

    return aug_prob


if __name__ == '__main__':
    args = pargs()

    unsup_train_size = args.unsup_train_size
    dataset = args.dataset
    vector_size = args.vector_size
    device = args.gpu if args.cuda else 'cpu'
    runs = args.runs
    k = args.k

    aug_ratio = args.aug_ratio
    batch_size = args.batch_size
    unsup_bs_ratio = args.unsup_bs_ratio
    weight_decay = args.weight_decay
    lamda = args.lamda
    epochs = args.epochs
    gamma_joao = args.gamma_joao

    label_source_path = osp.join(dirname, '..', 'Data', dataset, 'source')
    label_dataset_path = osp.join(dirname, '..', 'Data', dataset, 'dataset')
    train_path = osp.join(label_dataset_path, 'train')
    val_path = osp.join(label_dataset_path, 'val')
    test_path = osp.join(label_dataset_path, 'test')
    unlabel_dataset_path = osp.join(dirname, '..', 'Data', 'Weibo-unsup', 'dataset')
    model_path = osp.join(dirname, '..', 'Model', f'w2v_{dataset}_{unsup_train_size}_{vector_size}.model')

    log_name = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
    log_path = osp.join(dirname, '..', 'Log', f'{log_name}.log')
    log_json_path = osp.join(dirname, '..', 'Log', f'{log_name}.json')

    log = open(log_path, 'w')
    log_dict = create_log_dict_semisup(args)

    if not osp.exists(model_path):
        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path)
        elif dataset == 'Weibo-self':
            sort_weibo_self_dataset(label_source_path, label_dataset_path, unlabel_dataset_path)
        elif dataset == 'Weibo-2class' or dataset == 'Weibo-2class-long':
            sort_weibo_2class_dataset(label_source_path, label_dataset_path)

        sentences = collect_sentences(label_dataset_path, unlabel_dataset_path, unsup_train_size)
        w2v_model = train_word2vec(sentences, vector_size)
        w2v_model.save(model_path)

    for run in range(runs):
        write_log(log, f'run:{run}')
        log_record = {'run': run, 'val accs': [], 'test accs': [], 'test prec T': [], 'test prec F': [],
                      'test rec T': [], 'test rec F': [], 'test f1 T': [], 'test f1 F': []}

        word2vec = Embedding(model_path)
        unlabel_dataset = WeiboDataset(unlabel_dataset_path, word2vec, clean=False)
        unlabel_dataset.set_aug_mode('sample')
        unlabel_dataset.set_aug_ratio(aug_ratio)
        aug_prob = np.ones(25) / 25
        unlabel_dataset.set_aug_prob(aug_prob)

        unsup_train_loader = PreDataLoader(unlabel_dataset, batch_size * unsup_bs_ratio, shuffle=True)

        if dataset == 'Weibo':
            sort_weibo_dataset(label_source_path, label_dataset_path, k)
        elif dataset == 'Weibo-self':
            sort_weibo_self_dataset(label_source_path, label_dataset_path, unlabel_dataset_path, k)
        elif dataset == 'Weibo-2class' or dataset == 'Weibo-2class-long':
            sort_weibo_2class_dataset(label_source_path, label_dataset_path, k)

        train_dataset = WeiboDataset(train_path, word2vec)
        train_dataset.set_aug_mode('sample')
        train_dataset.set_aug_ratio(aug_ratio)
        train_dataset.set_aug_prob(aug_prob)
        val_dataset = WeiboFTDataset(val_path, word2vec)
        test_dataset = WeiboFTDataset(test_path, word2vec)

        train_loader = PreDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = ResGCN_graphcl(dataset=unlabel_dataset, hidden=args.hidden, num_feat_layers=args.n_layers_feat,
                               num_conv_layers=args.n_layers_conv, num_fc_layers=args.n_layers_fc, gfn=False,
                               collapse=False, residual=args.skip_connection, res_branch=args.res_branch,
                               global_pool=args.global_pool, dropout=args.dropout, edge_norm=args.edge_norm).to(device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, min_lr=0.000001)

        val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                       device, 0, args.lr, 0, 0, log_record)
        write_log(log, log_info)

        for epoch in range(1, epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']
            _, aug_prob_unsup, aug_prob_train = semisup_train(unsup_train_loader, train_loader, model, optimizer,
                                                              device, gamma_joao, lamda)

            unsup_train_loader.dataset.set_aug_prob(aug_prob_unsup)
            train_loader.dataset.set_aug_prob(aug_prob_train)

            train_error, train_acc, _, _, _ = test(model, train_loader, device, 'train')
            val_error, log_info, log_record = test_and_log(model, val_loader, test_loader,
                                                           device, epoch, lr, train_error, train_acc,
                                                           log_record)
            write_log(log, log_info)

            aug_prob_unsup = [round(prob, 2) for prob in aug_prob_unsup]
            write_log(log, f'Aug Prob Unsup: {aug_prob_unsup}')
            aug_prob_train = [round(prob, 2) for prob in aug_prob_train]
            write_log(log, f'Aug Prob Train: {aug_prob_train}')

            scheduler.step(val_error)

        log_record['mean acc'] = round(np.mean(log_record['test accs'][-8:]), 3)
        write_log(log, '')

        log_dict['record'].append(log_record)
        write_json(log_dict, log_json_path)
