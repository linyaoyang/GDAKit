import math
import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


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
    def __init__(self, input_dim, hid_dim, emb_dim, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, emb_dim)
        self.gc3 = GraphConvolution(hid_dim, emb_dim)
        self.dropout = dropout

    def forward(self, feat, adj):
        support = F.relu(self.gc1(feat, adj))
        support = F.dropout(support, self.dropout, training=self.training)
        res1, res2 = self.gc2(support, adj), self.gc3(support, adj)
        if self.training:
            std = torch.exp(res2)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(res1), res1, res2
        else:
            return res1, res1, res2
        # return res1, res1, res2


class GCNVAE(nn.Module):
    def __init__(self, input_dim, hid_dim, emb_dim):
        super(GCNVAE, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hid_dim)
        self.gc2 = GraphConvolution(hid_dim, emb_dim)
        self.gc3 = GraphConvolution(hid_dim, emb_dim)

    def forward(self, feat, adj):
        support = self.gc1(feat, adj)
        res1, res2 = self.gc2(support, adj), self.gc3(support, adj)
        if self.training:
            std = torch.exp(res2)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(res1), res1, res2
        else:
            return res1, res1, res2
        # return res1, res1, res2


class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.transform = nn.Linear(input_dim, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, feat):
        stacked = torch.stack(feat, dim=1)
        weights = F.softmax(self.transform(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout):
        super(Encoder, self).__init__()
        self.private_local_encoder = GCNVAE(input_dim, hid_dim, output_dim)
        self.private_global_encoder = GCNVAE(input_dim, hid_dim, output_dim)
        self.shared_local_encoder = GCN(input_dim, hid_dim, output_dim, dropout)
        self.shared_global_encoder = GCN(input_dim, hid_dim, output_dim, dropout)
        self.att_model = Attention(output_dim)

    def forward(self, feat, adj, ppmi):
        emb_p_l, mu_p_l, logvar_p_l = self.private_local_encoder(feat, adj)
        emb_p_g, mu_p_g, logvar_p_g = self.private_global_encoder(feat, ppmi)
        emb_s_l, mu_s_l, logvar_s_l = self.shared_local_encoder(feat, adj)
        emb_s_g, mu_s_g, logvar_s_g = self.shared_global_encoder(feat, ppmi)
        shared_emb = self.att_model([mu_s_l, mu_s_g])
        return [emb_p_l, mu_p_l, logvar_p_l], [emb_p_g, mu_p_g, logvar_p_g], [emb_s_l, mu_s_l, logvar_s_l], [emb_s_g, mu_s_g, logvar_s_g], shared_emb


class GradReverse(torch.autograd.Function):
    rate = 0.
    @staticmethod
    def forward(ctx, *args, **kwargs):
        return args[0].view_as(args[0])

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output = grad_outputs[0].neg() * GradReverse.rate
        return grad_output, None


class GRL(nn.Module):
    def forward(self, input):
        return GradReverse.apply(input)


class InnerProductDecoder(nn.Module):
    """Decoder which uses inner product for prediction"""
    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


# class DiffLoss(nn.Module):
#     """make input and target to be orthogonal"""
#     def __init__(self):
#         super(DiffLoss, self).__init__()
#
#     def forward(self, input, target):
#         batch_size = input.shape[0]
#         input = input.view(batch_size, -1)
#         target = target.view(batch_size, -1)
#         normalized_input = torch.norm(input, p=2, dim=1, keepdim=True).detach()
#         l2_input = normalized_input.div(normalized_input.expand_as(input) + 1e-6)
#         normalized_target = torch.norm(target, p=2, dim=1, keepdim=True).detach()
#         l2_target = normalized_target.div(normalized_target.expand_as(target) + 1e-6)
#         diff_loss = torch.mean(l2_input.t().mm(l2_target)).pow(2)
#         return diff_loss

class DiffLoss(nn.Module):

    def __init__(self):
        super(DiffLoss, self).__init__()

    def forward(self, input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


class ASN(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, num_class, dropout, lmd_d, lmd_r, lmd_f, epochs):
        super(ASN, self).__init__()
        self.lmd_d = lmd_d
        self.lmd_r = lmd_r
        self.lmd_f = lmd_f
        self.epochs = epochs
        self.encoder = Encoder(input_dim, hid_dim, output_dim, dropout)
        self.node_classifier = nn.Linear(output_dim, num_class)
        self.domain_discriminator = nn.Sequential(GRL(), nn.Linear(output_dim, 10), nn.ReLU(), nn.Dropout(0.1),
                                                  nn.Linear(10, 2))
        self.self_att_src = Attention(output_dim)
        self.self_att_tgt = Attention(output_dim)
        self.decoder_src = InnerProductDecoder(dropout, act=lambda x: x)
        self.decoder_tgt = InnerProductDecoder(dropout, act=lambda x: x)
        self.diff_loss = DiffLoss()
        self.clf_loss = nn.CrossEntropyLoss()

    def reconstruction_loss(self, pred, label, mu, logvar, num_nodes, norm, pos_weight):
        cost = norm * F.binary_cross_entropy_with_logits(pred, label, pos_weight=pos_weight)
        KLD = -0.5 / num_nodes * torch.mean(torch.sum(1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return cost + KLD

    def forward(self, feat_src, adj_src, ppmi_src, feat_tgt, adj_tgt, ppmi_tgt, label_src, domain_label, adj_label_src,
                adj_label_tgt, norm_src, norm_tgt, pos_weight_src, pos_weight_tgt, train_idx, epoch):
        rep_p_l_src, rep_p_g_src, rep_s_l_src, rep_s_g_src, emb_src = self.encoder(feat_src, adj_src, ppmi_src)
        rep_p_l_tgt, rep_p_g_tgt, rep_s_l_tgt, rep_s_g_tgt, emb_tgt = self.encoder(feat_tgt, adj_tgt, ppmi_tgt)
        emb = torch.cat((emb_src, emb_tgt), dim=0)
        pred_logit = self.node_classifier(emb)
        d_logit = self.domain_discriminator(emb)

        # compute encoder difference loss for source and target networks
        diff_loss_src = self.diff_loss(rep_p_l_src[1], rep_s_l_src[1])
        diff_loss_tgt = self.diff_loss(rep_p_l_tgt[1], rep_s_l_tgt[1])
        diff_loss = diff_loss_src + diff_loss_tgt

        # compute node classification loss
        clf_loss = self.clf_loss(pred_logit[train_idx, :], label_src[train_idx])

        # compute domain discrimination loss
        domain_loss = self.clf_loss(d_logit, domain_label)

        # compute reconstruction loss
        z_cat_src = torch.cat(
            (self.self_att_src([rep_p_l_src[0], rep_p_g_src[0]]), self.self_att_src([rep_s_l_src[0], rep_s_g_src[0]])),
            dim=1)
        z_cat_tgt = torch.cat(
            (self.self_att_tgt([rep_p_l_tgt[0], rep_p_g_tgt[0]]), self.self_att_tgt([rep_s_l_tgt[0], rep_s_g_tgt[0]])),
            dim=1)
        recovered_cat_src, recovered_cat_tgt = self.decoder_src(z_cat_src), self.decoder_tgt(z_cat_tgt)
        mu_cat_src = torch.cat((rep_p_l_src[1], rep_p_g_src[1], rep_s_l_src[1], rep_s_g_src[1]), dim=1)
        mu_cat_tgt = torch.cat((rep_p_l_tgt[1], rep_p_g_tgt[1], rep_s_l_tgt[1], rep_s_g_tgt[1]), dim=1)
        logvar_cat_src = torch.cat((rep_p_l_src[2], rep_p_g_src[2], rep_s_l_src[2], rep_s_g_src[2]), dim=1)
        logvar_cat_tgt = torch.cat((rep_p_l_tgt[2], rep_p_g_tgt[2], rep_s_l_tgt[2], rep_s_g_tgt[2]), dim=1)
        rec_loss_src = self.reconstruction_loss(recovered_cat_src, adj_label_src, mu_cat_src, logvar_cat_src,
                                                feat_src.shape[0], norm_src, pos_weight_src)
        rec_loss_tgt = self.reconstruction_loss(recovered_cat_tgt, adj_label_tgt, mu_cat_tgt, logvar_cat_tgt,
                                                feat_tgt.shape[0], norm_tgt, pos_weight_tgt)
        reconstruction_loss = rec_loss_src + rec_loss_tgt

        # compute entropy loss
        target_pred = F.softmax(pred_logit[:feat_src.shape[0], :], dim=-1)
        target_pred = torch.clamp(target_pred, min=1e-9, max=1.0)
        entropy_loss = torch.mean(torch.sum(-target_pred * torch.log(target_pred), dim=-1))

        # overall loss
        loss = clf_loss + self.lmd_d * domain_loss + self.lmd_r * reconstruction_loss + self.lmd_f * diff_loss + entropy_loss * (
                    epoch / self.epochs * 0.01)

        return pred_logit, loss