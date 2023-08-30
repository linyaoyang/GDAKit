import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from models.ASN import ASN, GradReverse
from utils import load_asn_data, load_adj_label


# set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='A', choices=['D', 'A', 'C'], help='Source network')
parser.add_argument('--target', type=str, default='D', choices=['D', 'A', 'C'], help='Target network')
parser.add_argument('--lr', type=float, default=3e-2, help='Learning rate')
parser.add_argument('--lmd_d', type=float, default=0.5, help='Weight for domain loss')
parser.add_argument('--lmd_r', type=float, default=1, help='Weight for reconstruction loss')
parser.add_argument('--lmd_f', type=float, default=0.0001, help='Weight for different loss')
parser.add_argument('--seed', type=int, default=3407, help='Random seed')
parser.add_argument('--epochs', type=int, default=2000, help='Training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--hid_dim', type=int, default=128, help='Hidden dim of encoder')
parser.add_argument('--emb_dim', type=int, default=16, help='Dimension of the encoder')
parser.add_argument('--disc_dim', type=int, default=40, help='Hidden dim of discriminator')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
args = parser.parse_args()

# set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
torch.set_num_threads(10)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.device = device

# Load data
dataset = {'A': 'acmv9.mat', 'C': 'citationv1.mat', 'D': 'dblpv7.mat'}
adj_src, feat_src, label_src, idx_src, X_n_src = load_asn_data('../data/' + dataset[args.source])
adj_tgt, feat_tgt, label_tgt, idx_tgt, X_n_tgt = load_asn_data('../data/' + dataset[args.target], target=True)

# Load adj labels for reconstruction
adj_label_src, pos_weight_src, norm_src = load_adj_label('../data/' + dataset[args.source])
adj_label_tgt, pos_weight_tgt, norm_tgt = load_adj_label('../data/' + dataset[args.target], target=True)

train_idx = torch.arange(feat_src.shape[0])

# Move them to the device
adj_src, feat_src, label_src, idx_src, X_n_src = adj_src.to(device), feat_src.to(device), label_src.to(
    device), idx_src.to(device), X_n_src.to(device)
adj_tgt, feat_tgt, label_tgt, idx_tgt, X_n_tgt = adj_tgt.to(device), feat_tgt.to(device), label_tgt.to(
    device), idx_tgt.to(device), X_n_tgt.to(device)
adj_label_src, pos_weight_src = adj_label_src.to(device), pos_weight_src.to(device)
adj_label_tgt, pos_weight_tgt = adj_label_tgt.to(device), pos_weight_tgt.to(device)
domain_label = torch.cat(
    (torch.zeros(feat_src.shape[0], dtype=torch.long), torch.ones(feat_tgt.shape[0], dtype=torch.long))).to(device)

model = ASN(feat_src.shape[1], args.hid_dim, args.emb_dim, 5, args.dropout, args.lmd_d, args.lmd_r, args.lmd_f,
            args.epochs).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

# Begin training
print('Begin training')
best_acc = 0
for epoch in range(args.epochs):
    model.train()
    optimizer.zero_grad()
    rate = min((epoch + 1) / args.epochs, 0.05)
    GradReverse.rate = rate
    pred_logit, loss = model(feat_src, adj_src, X_n_src, feat_tgt, adj_tgt, X_n_tgt, label_src, domain_label,
                             adj_label_src, adj_label_tgt, norm_src, norm_tgt, pos_weight_src, pos_weight_tgt, train_idx, epoch)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        pred_logit, loss = model(feat_src, adj_src, X_n_src, feat_tgt, adj_tgt, X_n_tgt, label_src, domain_label,
                                 adj_label_src, adj_label_tgt, norm_src, norm_tgt, pos_weight_src, pos_weight_tgt, train_idx,
                                 epoch)
        acc = (torch.argmax(pred_logit[feat_src.shape[0]:, :], dim=1) == label_tgt).sum() / feat_tgt.shape[0]
        print('epoch: {}, accuracy: {}'.format(epoch, acc))
        if acc > best_acc:
            best_acc = acc

print('Best accuracy', best_acc)