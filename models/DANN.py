import torch
import math
from torch import nn
import torch.nn.functional as F


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


class DomainDiscriminator(nn.Module):
    def __init__(self, n_emb):
        super(DomainDiscriminator, self).__init__()
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

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, emb_dim, dropout):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(input_dim, hid_dim)
        std = 1 / (hid_dim / 2) ** 0.5
        nn.init.trunc_normal_(self.l1.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.l1.bias, 0.1)
        self.l2 = nn.Linear(hid_dim, emb_dim)
        std = 1 / (emb_dim / 2) ** 0.5
        nn.init.trunc_normal_(self.l2.weight, std=std, a=-2 * std, b=2 * std)
        nn.init.constant_(self.l2.bias, 0.1)
        self.dropout = dropout

    def forward(self, feat):
        hid = F.relu(self.l1(feat))
        hid = F.dropout(hid, self.dropout, training=self.training)
        output = F.relu(self.l2(hid))
        return output

class DANN(nn.Module):
    def __init__(self, input_dim, hid_dim, emb_dim, num_class, dropout):
        super(DANN, self).__init__()
        # self.encoder = nn.Sequential(nn.Linear(input_dim, hid_dim), nn.ReLU(), nn.Dropout(dropout),
        #                              nn.Linear(hid_dim, emb_dim))
        self.encoder = Encoder(input_dim, hid_dim, emb_dim, dropout)
        self.node_classifier = NodeClassifier(emb_dim, num_class)
        self.domain_discriminator = DomainDiscriminator(emb_dim)
        self.grl = GRL()

    def forward(self, feat):
        emb = self.encoder(feat)
        pred_logit = self.node_classifier(emb)
        h_grl = self.grl(emb)
        d_logit = self.domain_discriminator(h_grl)
        return emb, pred_logit, d_logit