import torch
import math
import numpy as np
import scipy.io as sio
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
        std = 1. / (math.sqrt(output_dim))
        nn.init.uniform_(self.weight, -std, std)
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_dim))
            nn.init.uniform_(self.bias, -std, std)
        else:
            self.register_parameter('bias', None)

    def forward(self, feat, adj):
        support = torch.mm(feat, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    def __init__(self, input_dim, hid_dim, emb_dim, num_class, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, emb_dim)
        self.clf = nn.Linear(emb_dim, num_class)
        self.dropout = dropout

    def forward(self, feat, adj):
        support = F.relu(self.gc1(feat, adj))
        support = F.dropout(support, self.dropout, training=self.training)
        emb = self.gc2(support, adj)
        pred_logit = self.clf(emb)
        return pred_logit


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
tgt_adj = torch.eye(tgt_feat.shape[0])

src_feat, src_adj, src_label = src_feat.cuda(), src_adj.cuda(), src_label.cuda()
tgt_feat, tgt_adj, tgt_label = tgt_feat.cuda(), tgt_adj.cuda(), tgt_label.cuda()

model = GCN(src_feat.shape[1], 512, 16, 5, 0.5).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
loss_func = nn.CrossEntropyLoss()

print('Begin training!')
for i in range(1000):
    model.train()
    optimizer.zero_grad()
    train_logit = model(src_feat, src_adj)
    loss = loss_func(train_logit, src_label)
    loss.backward()
    optimizer.step()
    if (i + 1) % 5 == 0:
        model.eval()
        train_logit = model(src_feat, src_adj)
        train_acc = torch.sum(torch.argmax(train_logit, dim=1) == src_label) / len(src_label)
        test_logit = model(tgt_feat, tgt_adj)
        test_acc = torch.sum(torch.argmax(test_logit, dim=1) == tgt_label) / len(tgt_label)
        print('epoch: {}, train acc: {}, test acc: {}'.format(i, train_acc, test_acc))