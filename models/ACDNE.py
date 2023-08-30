# ACDNE model "Adversarial Deep Network Embedding for Cross-network Node Classification"

import torch
from torch import nn
import torch.nn.functional as F


class FE1(nn.Module):
    def __init__(self, n_input, n_hidden, dropout):
        super(FE1, self).__init__()
        self.dropout = dropout
        self.h1_self = nn.Linear(n_input, n_hidden[0])
        self.h2_self = nn.Linear(n_hidden[0], n_hidden[1])
        std = 1 / (n_input / 2) ** 0.5
        nn.init.trunc_normal_(self.h1_self.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.h1_self.bias, 0.1)
        std = 1 / (n_hidden[0] / 2) ** 0.5
        nn.init.trunc_normal_(self.h2_self.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.h2_self.bias, 0.1)

    def forward(self, x):
        x = F.dropout(F.relu(self.h1_self(x)), self.dropout)
        return F.relu(self.h2_self(x))


class FE2(nn.Module):
    def __init__(self, n_input, n_hidden, dropout):
        super(FE2, self).__init__()
        self.dropout = dropout
        self.h1_ngh = nn.Linear(n_input, n_hidden[0])
        self.h2_ngh = nn.Linear(n_hidden[0], n_hidden[1])
        std = 1 / (n_input / 2) ** 0.5
        nn.init.trunc_normal_(self.h1_ngh.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.h1_ngh.bias, 0.1)
        std = 1 / (n_hidden[0] / 2) ** 0.5
        nn.init.trunc_normal_(self.h2_ngh.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.h2_ngh.bias, 0.1)

    def forward(self, x_ngh):
        x_ngh = F.dropout(F.relu(self.h1_ngh(x_ngh)), self.dropout)
        return F.relu(self.h2_ngh(x_ngh))


class Encoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_emb, dropout, batch_size):
        super(Encoder, self).__init__()
        self.dropout = dropout
        self.batch_size = batch_size
        self.fe1 = FE1(n_input, n_hidden, dropout)
        self.fe2 = FE2(n_input, n_hidden, dropout)
        self.emb = nn.Linear(n_hidden[-1] * 2, n_emb)
        std = 1 / (n_hidden[-1] * 2) ** 0.5
        nn.init.trunc_normal_(self.emb.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.emb.bias, 0.1)

    def forward(self, x, x_ngh):
        h_self = self.fe1(x)
        h_ngh = self.fe2(x_ngh)
        return F.relu(self.emb(torch.cat((h_self, h_ngh), dim=1)))

    def pairwise_constraint(self, emb):
        emb_s = emb[:int(self.batch_size / 2), :]
        emb_t = emb[int(self.batch_size / 2):, :]
        return emb_s, emb_t

    @staticmethod
    def net_pro_loss(emb, adj):
        r = torch.sum(emb * emb, 1)
        r = torch.reshape(r, (-1, 1))
        dist = r - 2 * torch.matmul(emb, emb.T) + r.T
        return torch.mean(torch.sum(adj.mul(dist), 1))


class NodeClassifier(nn.Module):
    def __init__(self, n_emb, num_class):
        super(NodeClassifier, self).__init__()
        self.layer = nn.Linear(n_emb, num_class)
        std = 1 / (n_emb / 2) ** 0.5
        nn.init.trunc_normal_(self.layer.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.layer.bias, 0.1)

    def forward(self, emb):
        pred_logit = self.layer(emb)
        return pred_logit


class DomainDoscriminator(nn.Module):
    def __init__(self, n_emb):
        super(DomainDoscriminator, self).__init__()
        self.h_dann_1 = nn.Linear(n_emb, 128)
        self.h_dann_2 = nn.Linear(128, 128)
        self.output_layer = nn.Linear(128, 2)
        std = 1 / (n_emb / 2) ** 0.5
        nn.init.trunc_normal_(self.h_dann_1.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.h_dann_1.bias, 0.1)
        nn.init.trunc_normal_(self.h_dann_2.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.constant_(self.h_dann_2.bias, 0.1)
        nn.init.trunc_normal_(self.output_layer.weight, std=0.125, a=-0.25, b=0.25)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, h_grl):
        h_grl = F.relu(self.h_dann_1(h_grl))
        h_grl = F.relu(self.h_dann_2(h_grl))
        d_logit = self.output_layer(h_grl)
        return d_logit


class GradientReverse(torch.autograd.Function):
    rate = 0.0
    @staticmethod
    def forward(ctx, *args, **kwargs):
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0].neg() * GradientReverse.rate
        return grad_output, None


class GRL(nn.Module):
    @staticmethod
    def forward(inp):
        return GradientReverse.apply(inp)


class ACDNE(nn.Module):
    def __init__(self, n_input, n_hidden, n_emb, num_class, batch_size, dropout):
        super(ACDNE, self).__init__()
        self.encoder = Encoder(n_input, n_hidden, n_emb, dropout, batch_size)
        self.node_classifier = NodeClassifier(n_emb, num_class,)
        self.domain_discriminator = DomainDoscriminator(n_emb)
        self.grl = GRL()

    def forward(self, x, x_ngh):
        emb = self.encoder(x, x_ngh)
        pred_logit = self.node_classifier(emb)
        h_grl = self.grl(emb)
        d_logit = self.domain_discriminator(h_grl)
        return emb, pred_logit, d_logit