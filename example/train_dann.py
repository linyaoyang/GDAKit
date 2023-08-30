import argparse
import numpy as np
import torch
from torch import nn
from models.DANN import DANN, GradientReverse
from utils import load_networks, multi_label_acc
from torch import optim


# set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='C', choices=['D', 'A', 'C'], help='Source network')
parser.add_argument('--target', type=str, default='D', choices=['D', 'A', 'C'], help='Target network')
parser.add_argument('--lr', type=float, default=0.03, help='Learning rate')
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
labels = torch.argmax(labels, dim=1)
feature, labels = feature.to(device), labels.to(device)
train_idx, test_idx = train_idx.to(device), test_idx.to(device)
domain_label = torch.LongTensor([0 for _ in range(len(train_idx))] + [1 for _ in range(len(test_idx))]).to(device)
print('Data Loaded.')


model = DANN(feature.shape[1], 512, 16, 5, 0.5).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-3)
loss_func = nn.CrossEntropyLoss()

print('Begin training')
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    rate = min((epoch + 1) / args.epochs, 0.05)
    GradientReverse.rate = rate
    _, pred_logit, d_logit = model(feature)
    clf_loss = loss_func(pred_logit[train_idx], labels[train_idx])
    domain_loss = loss_func(d_logit, domain_label)
    loss = clf_loss + domain_loss
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        model.eval()
        _, pred_logit, d_logit = model(feature)
        train_acc = torch.sum(torch.argmax(pred_logit[train_idx], dim=1) == labels[train_idx]) / len(train_idx)
        test_acc = torch.sum(torch.argmax(pred_logit[test_idx], dim=1) == labels[test_idx]) / len(test_idx)
        print('epoch: {}, train acc: {}, test acc: {}'.format(epoch, train_acc, test_acc))