import torch
import math
import numpy as np
import scipy.io as sio
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class MLP(nn.Module):
    def __init__(self, input_dim, hid_dim, emb_dim, num_class, dropout):
        super(MLP, self).__init__()
        self.clf = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hid_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, num_class))

    def forward(self, feat):
        return self.clf(feat)

# Load data
def load_data(dataset):
    net = sio.loadmat(dataset)
    X, A, Y = net['attrb'], net['network'], net['group']
    feat = torch.FloatTensor(X)
    adj = torch.FloatTensor(A.toarray())
    label = torch.LongTensor(np.argmax(Y, axis=1))
    return feat, adj, label

src_feat, src_adj, src_label = load_data('../data/citationv1.mat')
tgt_feat, tgt_adj, tgt_label = load_data('../data/dblpv7.mat')

src_feat, src_adj, src_label = src_feat.cuda(), src_adj.cuda(), src_label.cuda()
tgt_feat, tgt_adj, tgt_label = tgt_feat.cuda(), tgt_adj.cuda(), tgt_label.cuda()

model = MLP(src_feat.shape[1], 512, 16, 5, 0.5).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()

print('Begin training!')
for i in range(1000):
    model.train()
    optimizer.zero_grad()
    train_logit = model(src_feat)
    loss = loss_func(train_logit, src_label)
    loss.backward()
    optimizer.step()
    if (i + 1) % 5 == 0:
        model.eval()
        train_logit = model(src_feat)
        train_acc = torch.sum(torch.argmax(train_logit, dim=1) == src_label) / len(src_label)
        test_logit = model(tgt_feat)
        test_acc = torch.sum(torch.argmax(test_logit, dim=1) == tgt_label) / len(tgt_label)
        print('epoch: {}, train acc: {}, test acc: {}'.format(i, train_acc, test_acc))