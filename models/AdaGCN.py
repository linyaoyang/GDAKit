import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, smooth_steps, device, sparse=False):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.Tensor(input_dim, output_dim))
        thres = math.sqrt(6.0 / (input_dim + output_dim))
        nn.init.uniform_(self.weight, a=-thres, b=thres)
        self.dropout = dropout
        self.smooth_steps = smooth_steps
        self.device = device
        self.sparse = sparse

    def conv(self, adj, features):
        def rnm(adj, features, k):
            new_feature = features
            for _ in range(k):
                new_feature = torch.matmul(adj, new_feature)
            return new_feature
        res = rnm(adj, features, self.smooth_steps)
        return res

    def forward(self, x, support):
        # Dropout
        if self.sparse:
            # x = F.dropout(x.to_dense(), p=self.dropout, training=self.training)
            mask = torch.rand(x.shape).to(self.device)
            mask = torch.where(mask <= self.dropout, 0, 1)
            x = torch.mul(x, mask)
            x = x * (1.0 / (1 - self.dropout))
        else:
            x = F.dropout(x, self.dropout, training=self.training)

        # Convolution
        pre_sup = torch.matmul(x, self.weight)
        sup = self.conv(support, pre_sup)

        return F.relu(sup)


class Encoder(nn.Module):
    def __init__(self, input_dim, conv_dim, hid_dim, emb_dim, dropout, smooth_steps, device):
        """input_dim: 6775, conv_dim: 1000, hid_dim: 100, emb_dim: 16"""
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.conv = GraphConvolution(input_dim, conv_dim, dropout, smooth_steps, device, sparse=True)
        self.weight1 = Parameter(torch.Tensor(conv_dim, hid_dim))
        thres1 = math.sqrt(6.0 / (conv_dim + hid_dim))
        nn.init.uniform_(self.weight1, a=-thres1, b=thres1)
        self.weight2 = Parameter(torch.Tensor(hid_dim, emb_dim))
        thres2 = math.sqrt(6.0 / (hid_dim + emb_dim))
        nn.init.uniform_(self.weight2, a=-thres2, b=thres2)

    def forward(self, x, support):
        hidden = self.conv(x, support)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        output = torch.matmul(hidden, self.weight1)
        output = F.dropout(output, self.dropout, training=self.training)
        output = torch.matmul(output, self.weight2)
        return output


class DomainDiscriminator(nn.Module):
    def __init__(self, emb_dim, disc_dim, pred_dim):
        super(DomainDiscriminator, self).__init__()
        self.weight1 = Parameter(torch.Tensor(emb_dim, disc_dim))
        thres1 = math.sqrt(6.0 / (emb_dim + disc_dim))
        nn.init.uniform_(self.weight1, a=-thres1, b=thres1)
        self.bias1 = Parameter(torch.zeros(disc_dim))
        self.weight2 = Parameter(torch.Tensor(disc_dim, pred_dim))
        thres2 = math.sqrt(6.0 / (disc_dim + pred_dim))
        nn.init.uniform_(self.weight2, a=-thres2, b=thres2)
        self.bias2 = Parameter(torch.zeros(pred_dim))

    def forward(self, x):
        hidden = torch.tanh(torch.matmul(x, self.weight1) + self.bias1)
        output = torch.matmul(hidden, self.weight2) + self.bias2
        return output


class NodeClassifier(nn.Module):
    def __init__(self, emb_dim, clf_dim):
        super(NodeClassifier, self).__init__()
        self.weight = Parameter(torch.Tensor(emb_dim, clf_dim))
        thres = math.sqrt(6.0 / (emb_dim + clf_dim))
        nn.init.uniform_(self.weight, a=-thres, b=thres)
        self.bias = Parameter(torch.zeros(clf_dim))

    def forward(self, x):
        output = torch.matmul(x, self.weight) + self.bias
        return output


class AdaGCN(nn.Module):
    def __init__(self, input_dim, conv_dim, hid_dim, emb_dim, disc_dim, pred_dim, clf_dim, dropout, gp_param,
                 smooth_steps, da_param, device):
        super(AdaGCN, self).__init__()
        self.encoder = Encoder(input_dim, conv_dim, hid_dim, emb_dim, dropout, smooth_steps, device)
        self.discriminator = DomainDiscriminator(emb_dim, disc_dim, pred_dim)
        self.classifier = NodeClassifier(emb_dim, clf_dim)
        self.gp_param = gp_param
        self.da_param = da_param
        self.device = device
        self.clf_loss_func = nn.BCEWithLogitsLoss()

    def forward(self, x, support, labels, train_idx, test_idx):
        emb = self.encoder(x, support)
        pred_logit = self.classifier(emb)
        N_s, N_t  = len(train_idx), len(test_idx)
        emb_s, emb_t = emb[:N_s, :], emb[N_s:, :]
        if N_s < N_t:
            emb_t_1, emb_t_2 = emb_t[:N_s, :], emb_t[-N_s:, :]
            hidden_s = torch.cat((emb_s, emb_s), dim=0)
            hidden_t = torch.cat((emb_t_1, emb_t_2), dim=0)
            alpha = torch.rand(2 * N_s, 1).to(self.device)
            interpolates = hidden_t + alpha * (hidden_t - hidden_s)
        elif N_s > N_t:
            emb_s_1, emb_s_2 = emb_s[:N_t, :], emb_s[-N_t:, :]
            hidden_s = torch.cat((emb_s_1, emb_s_2), dim=0)
            hidden_t = torch.cat((emb_t, emb_t), dim=0)
            alpha = torch.rand(2 * N_t, 1).to(self.device)
            interpolates = hidden_t + alpha * (hidden_s - hidden_t)
        else:
            hidden_s, hidden_t = emb_s, emb_t
            alpha = torch.rand(N_s, 1).to(self.device)
            interpolates = hidden_t + alpha * (hidden_s - hidden_t)

        hidden_whole = torch.cat((hidden_s, hidden_t, interpolates), dim=0)

        # discriminator loss
        disc_out = self.discriminator(hidden_whole)
        disc_s, disc_t = disc_out[:N_s, :], disc_out[N_s: N_s + N_t, :]
        wd_loss = torch.mean(disc_s) - torch.mean(disc_t)

        # gradient penalty
        gradients = torch.autograd.grad(outputs=disc_out, inputs=hidden_whole,
                                        grad_outputs=torch.ones_like(disc_out).to(self.device), retain_graph=True)[0]
        slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=1))
        gradient_penalty = torch.mean((slopes - 1.0) ** 2)
        disc_total_loss = -wd_loss + self.gp_param * gradient_penalty  # Total loss for training discriminator

        # classifier loss
        clf_loss = self.clf_loss_func(pred_logit[train_idx], labels[train_idx])
        clf_total_loss = clf_loss + self.gp_param * wd_loss

        return pred_logit, disc_total_loss, clf_total_loss