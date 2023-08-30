# This is a toolkit for existing graph domain adaptation algorithms

import argparse
import numpy as np
import torch
from torch import nn
from models.AdaGCN import AdaGCN
from utils import load_networks, multi_label_acc
from torch import optim


# set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='C', choices=['D', 'A', 'C'], help='Source network')
parser.add_argument('--target', type=str, default='D', choices=['D', 'A', 'C'], help='Target network')
parser.add_argument('--lr', type=float, default=0.05, help='Learning rate')
parser.add_argument('--seed', type=int, default=3407, help='Random seed')
parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
parser.add_argument('--D_train_step', type=int, default=10, help='Number of steps for training discriminator')
parser.add_argument('--lr_gen', type=float, default=1.5e-3, help='Initial learning rate for the generator')
parser.add_argument('--lr_dis', type=float, default=1.5e-3, help='Initial learning rate ')
parser.add_argument('--l2_param', type=float, default=5e-5, help='Weight for L2 regularization')
parser.add_argument('--shrinking', type=float, default=0.8, help='Learning rate decaying factor')
parser.add_argument('--gp_param', type=float, default=10, help='Weight of gradient penalty')
parser.add_argument('--da_param', type=float, default=10, help='Weight of domain adaptation loss')
args = parser.parse_args()


# set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
torch.set_num_threads(10)

# set device
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
args.device = device

# Load dataset
dataset = {'A': 'acmv9.mat', 'C': 'citationv1.mat', 'D': 'dblpv7.mat'}
feature, adj, labels, train_idx, test_idx = load_networks('../data/' + dataset[args.source], '../data/' + dataset[args.target])
feature, adj, labels = feature.to(device), adj.to(device), labels.to(device)
train_idx, test_idx = train_idx.to(device), test_idx.to(device)
print('Data Loaded.')

clf_loss_func = nn.BCEWithLogitsLoss(reduction='none')  # compute loss and update parameter for each sample


# hyper-parameter: conv_dim-1000, hid_dim-100, emb_dim-16, disc_dim-16, pred_dim-1, clf_dim-5
model = AdaGCN(feature.shape[1], 1000, 100, 16, 16, 1, 5, dropout=0.3, gp_param=args.gp_param, smooth_steps=10,
               da_param=args.da_param, device=device).to(device)
lr_gen, lr_dis = args.lr_gen, args.lr_dis

# start training
print('Begin Training')
for epoch in range(args.epochs):
    if (epoch + 1) >= 500 and (epoch + 1) % 100 == 0:
        lr_gen = lr_gen * args.shrinking
        lr_dis = lr_dis * args.shrinking
    optimizer_gen = optim.Adam(list(model.encoder.parameters()) + list(model.classifier.parameters()), lr=lr_gen,
                               weight_decay=args.l2_param)
    optimizer_dis = optim.Adam(model.discriminator.parameters(), lr=lr_dis)

    model.train()
    for _ in range(args.D_train_step):
        pred_logit, disc_loss, clf_loss = model(feature, adj, labels, train_idx, test_idx)
        optimizer_dis.zero_grad()
        disc_loss.backward()
        optimizer_dis.step()

    pred_logit, disc_loss, clf_loss = model(feature, adj, labels, train_idx, test_idx)
    optimizer_gen.zero_grad()
    clf_loss.backward()
    optimizer_gen.step()

    if epoch % 5 == 0:
        model.eval()
        pred_logit, disc_loss, clf_loss = model(feature, adj, labels, train_idx, test_idx)
        micro_f1, macro_f1 = multi_label_acc(pred_logit, labels, test_idx)
        print('Epoch: {}, Micro_F1: {}, Macro_F1: {}'.format(epoch, micro_f1, macro_f1))