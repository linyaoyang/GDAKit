import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from models.UDAGCN import UDAGCN, GradReverse
from utils import load_acm_dblp, load_udagcn_data


# set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='C', choices=['D', 'A', 'C'], help='Source network')
parser.add_argument('--target', type=str, default='D', choices=['D', 'A', 'C'], help='Target network')
parser.add_argument('--lr', type=float, default=3e-3, help='Learning rate')
parser.add_argument('--seed', type=int, default=3407, help='Random seed')
parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
parser.add_argument('--hid_dim', type=int, default=128, help='Hidden dim of encoder')
parser.add_argument('--emb_dim', type=int, default=16, help='Dimension of the encoder')
parser.add_argument('--disc_dim', type=int, default=40, help='Hidden dim of discriminator')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
parser.add_argument('--path_len', type=int, default=10, help='Path length for PPMI random walk')
args = parser.parse_args()

# set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
torch.set_num_threads(10)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

# Load dataset
# dataset = {'A': 'acmv9.mat', 'C': 'citationv1.mat', 'D': 'dblpv7.mat'}
# x_src, adj_src, label_src = load_acm_dblp('acm')
# x_tgt, adj_tgt, label_tgt = load_acm_dblp('dblp')
x_src, adj_src, label_src = load_udagcn_data('../data/citationv1.mat')
x_tgt, adj_tgt, label_tgt = load_udagcn_data('../data/dblpv7.mat')
adj_tgt = torch.LongTensor([[i for i in range(len(x_tgt))], [i for i in range(len(x_tgt))]])
x_src, adj_src, label_src = x_src.to(device), adj_src.to(device), label_src.to(device)
x_tgt, adj_tgt, label_tgt = x_tgt.to(device), adj_tgt.to(device), label_tgt.to(device)
train_idx = torch.LongTensor(np.arange(x_src.shape[0])).to(device)
test_idx = torch.LongTensor(np.arange(x_src.shape[0], x_src.shape[0] + x_tgt.shape[0])).to(device)
domain_label = torch.cat((torch.zeros(x_src.shape[0], dtype=torch.long), torch.ones(x_tgt.shape[0], dtype=torch.long))).to(device)

print('Data Loaded')
loss_func = nn.CrossEntropyLoss()
model = UDAGCN(x_src.shape[1], args.hid_dim, args.emb_dim, args.disc_dim, torch.max(label_src) + 1
               , args.dropout,
               args.path_len, device=device).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def test():
    model.eval()
    pred_logit, d_logit = model(x_src, adj_src, 'source', x_tgt, adj_tgt, 'target')
    tgt_pred = pred_logit[test_idx].argmax(dim=1)
    accuracy = tgt_pred.eq(label_tgt).float().mean()
    return accuracy

for epoch in range(1, args.epochs):
    model.train()
    optimizer.zero_grad()
    rate = min((epoch + 1) / args.epochs, 0.05)
    GradReverse.rate = rate
    pred_logit, d_logit = model(x_src, adj_src, 'source', x_tgt, adj_tgt, 'target')
    clf_loss = loss_func(pred_logit[train_idx], label_src)
    for name, param in model.named_parameters():
        if 'weight' in name:
            clf_loss = clf_loss + param.mean() * 3e-3
    domain_loss = loss_func(d_logit, domain_label)
    tgt_logit = pred_logit[test_idx]
    tgt_pred = F.softmax(tgt_logit, dim=-1)
    tgt_pred = torch.clamp(tgt_pred, min=1e-9, max=1.0)
    loss_entropy = torch.mean(torch.sum(-tgt_pred * torch.log(tgt_pred), dim=-1))
    loss = clf_loss + domain_loss + loss_entropy * (epoch / args.epochs * 0.01)
    loss.backward()
    optimizer.step()

    test_acc = test()
    print('epoch: {}, accuracy: {}'.format(epoch, test_acc))