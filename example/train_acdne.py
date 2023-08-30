import argparse
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from models.ACDNE import ACDNE, GradientReverse
from utils import load_mat_data, aggregate_trans_prob, compute_ppmi, batch_generator, batch_ppmi, f1_scores, f1_score


# set hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='C', choices=['D', 'A', 'C'], help='Source network')
parser.add_argument('--target', type=str, default='A', choices=['D', 'A', 'C'], help='Target network')
parser.add_argument('--lr', type=float, default=0.02, help='Learning rate')
parser.add_argument('--l2_w', type=float, default=1e-3, help='Weight for L2 regularization')
parser.add_argument('--net_pro_w', type=float, default=0.1, help='Weight of pairwise constraint')
parser.add_argument('--seed', type=int, default=3407, help='Random seed')
parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
parser.add_argument('--hid_dim', type=list, default=[512, 128], help='Hidden dim of encoder')
parser.add_argument('--emb_dim', type=int, default=128, help='Dimension of the encoder')
parser.add_argument('--disc_dim', type=int, default=40, help='Hidden dim of discriminator')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
parser.add_argument('--k_step', type=int, default=3, help='Path length for PPMI random walk')
args = parser.parse_args()

# set random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
torch.set_num_threads(10)

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
args.device = device

# load dataset
dataset = {'A': 'acmv9.mat', 'C': 'citationv1.mat', 'D': 'dblpv7.mat'}
X_s, A_s, Y_s = load_mat_data('../data/' + dataset[args.source])
X_t, A_t, Y_t = load_mat_data('../data/' + dataset[args.target])
# Compute PPMI
ngh_A_s = aggregate_trans_prob(A_s, args.k_step)
PPMI_s = compute_ppmi(ngh_A_s)
ngh_X_s = np.matmul(PPMI_s, X_s)
ngh_A_t = aggregate_trans_prob(A_t, args.k_step)
PPMI_t = compute_ppmi(ngh_A_t)
ngh_X_t = np.matmul(PPMI_t, X_t)

# prepare data
x_s_new, x_t_new = np.concatenate((X_s, ngh_X_s), axis=1), np.concatenate((X_t, ngh_X_t), axis=1)
num_nodes_s, num_nodes_t = X_s.shape[0], X_t.shape[0]
feat = torch.FloatTensor(np.vstack([X_s, X_t])).to(device)
feat_ngh = torch.FloatTensor(np.vstack([ngh_X_s, ngh_X_t])).to(device)

clf_loss_func = nn.BCEWithLogitsLoss(reduction='none')
# clf_loss_func = nn.CrossEntropyLoss()
domain_loss_func = nn.CrossEntropyLoss()

model = ACDNE(X_s.shape[1], args.hid_dim, args.emb_dim, 5, args.batch_size, args.dropout).to(device)

n_input = X_s.shape[1]
Y_t_o = np.zeros(np.shape(Y_t))

for epoch in range(args.epochs):
    s_batch = batch_generator([x_s_new, Y_s], int(args.batch_size / 2))
    t_batch = batch_generator([x_t_new, Y_t_o], int(args.batch_size / 2))
    num_batch = round(max(num_nodes_s / (args.batch_size / 2), num_nodes_t / (args.batch_size / 2)))
    p = float(epoch) / args.epochs
    lr = args.lr / (1. + 10 * p) ** 0.75
    grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1
    GradientReverse.rate = grl_lambda
    optimizer = torch.optim.SGD(model.parameters(), lr, 0.9, weight_decay=args.l2_w / 2)
    for cBatch in range(num_batch):
        xs_ys_batch, shuffle_index_s = next(s_batch)
        xs_batch = xs_ys_batch[0]
        ys_batch = xs_ys_batch[1]
        xt_yt_batch, shuffle_index_t = next(t_batch)
        xt_batch = xt_yt_batch[0]
        yt_batch = xt_yt_batch[1]
        x_batch = np.vstack([xs_batch, xt_batch])
        batch_csr = x_batch
        xb = torch.FloatTensor(batch_csr[:, 0:n_input]).to(device)
        xb_nei = torch.FloatTensor(batch_csr[:, -n_input:]).to(device)
        yb = np.vstack([ys_batch, yt_batch])
        mask_l = np.sum(yb, axis=1) > 0
        # 1 if the node is with observed label, 0 if the node is without label
        domain_label = np.vstack([np.tile([1., 0.], [args.batch_size // 2, 1]), np.tile([0., 1.], [
            args.batch_size // 2, 1])])  # [1,0] for source, [0,1] for target
        # #topological proximity matrix between nodes in each mini-batch
        a_s, a_t = batch_ppmi(args.batch_size, shuffle_index_s, shuffle_index_t, PPMI_s, PPMI_t)
        a_s, a_t = torch.from_numpy(a_s).to(device), torch.from_numpy(a_t).to(device)
        model.train()
        optimizer.zero_grad()
        emb, pred_logit, d_logit = model(xb, xb_nei)
        emb_s, emb_t = model.encoder.pairwise_constraint(emb)
        net_pro_loss_s = model.encoder.net_pro_loss(emb_s, a_s)
        net_pro_loss_t = model.encoder.net_pro_loss(emb_t, a_t)
        net_pro_loss = args.net_pro_w * (net_pro_loss_s + net_pro_loss_t)

        clf_loss = clf_loss_func(pred_logit[mask_l], torch.FloatTensor(yb[mask_l]).to(device))
        clf_loss = torch.sum(clf_loss) / np.sum(mask_l)
        domain_loss = domain_loss_func(d_logit, torch.argmax(torch.FloatTensor(domain_label).to(device), 1))
        total_loss = clf_loss + domain_loss + net_pro_loss
        total_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        GradientReverse.rate = 1.0
        _, pred_logit_xs_xt, _ = model(feat, feat_ngh)
        pred_prob_xs_xt = F.sigmoid(pred_logit_xs_xt)
        pred_prob_xs = pred_prob_xs_xt[0:num_nodes_s, :]
        pred_prob_xt = pred_prob_xs_xt[-num_nodes_t:, :]
        print('epoch: ', epoch + 1)
        f1_s = f1_scores(pred_prob_xs.detach().cpu().numpy(), Y_s)
        print('Source micro-F1: %f, macro-F1: %f' % (f1_s[0], f1_s[1]))
        f1_t = f1_scores(pred_prob_xt.detach().cpu().numpy(), Y_t)
        y_pred = torch.argmax(pred_prob_xt, dim=1).detach().cpu().numpy()
        y_true = np.argmax(Y_t, axis=1)
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        print('Testing results: {}, {}'.format(micro_f1, macro_f1))
        print('Target testing micro-F1: %f, macro-F1: %f' % (f1_t[0], f1_t[1]))


# for epoch in range(args.epochs):
#     s_batch = batch_generator([x_s_new, Y_s], int(args.batch_size / 2))
#     t_batch = batch_generator([x_t_new, Y_t_o], int(args.batch_size / 2))
#     num_batch = round(max(num_nodes_s / (args.batch_size / 2), num_nodes_t / (args.batch_size / 2)))
#     p = float(epoch) / args.epochs
#     lr = args.lr / (1. + 10 * p) ** 0.75
#     grl_lambda = 2. / (1. + np.exp(-10. * p)) - 1  # gradually change from 0 to 1
#     GradientReverse.rate = grl_lambda
#     optimizer = torch.optim.SGD(model.parameters(), lr, 0.9, weight_decay=args.l2_w / 2)
#     for cBatch in range(num_batch):
#         xs_ys_batch, shuffle_index_s = next(s_batch)
#         xs_batch = xs_ys_batch[0]
#         ys_batch = xs_ys_batch[1]
#         xt_yt_batch, shuffle_index_t = next(t_batch)
#         xt_batch = xt_yt_batch[0]
#         yt_batch = xt_yt_batch[1]
#         x_batch = np.vstack([xs_batch, xt_batch])
#         batch_csr = x_batch
#         xb = torch.FloatTensor(batch_csr[:, 0:n_input]).to(device)
#         xb_nei = torch.FloatTensor(batch_csr[:, -n_input:]).to(device)
#         yb = np.vstack([ys_batch, yt_batch])
#         mask_l = np.sum(yb, axis=1) > 0
#         # 1 if the node is with observed label, 0 if the node is without label
#         domain_label = np.vstack([np.tile([1., 0.], [args.batch_size // 2, 1]), np.tile([0., 1.], [
#             args.batch_size // 2, 1])])  # [1,0] for source, [0,1] for target
#         # #topological proximity matrix between nodes in each mini-batch
#         a_s, a_t = batch_ppmi(args.batch_size, shuffle_index_s, shuffle_index_t, PPMI_s, PPMI_t)
#         a_s, a_t = torch.FloatTensor(a_s).to(device), torch.FloatTensor(a_t).to(device)
#         model.train()
#         optimizer.zero_grad()
#         emb, pred_logit, d_logit = model(xb, xb_nei)
#         emb_s, emb_t = model.encoder.pairwise_constraint(emb)
#         net_pro_loss_s = model.encoder.net_pro_loss(emb_s, a_s)
#         net_pro_loss_t = model.encoder.net_pro_loss(emb_t, a_t)
#         net_pro_loss = args.net_pro_w * (net_pro_loss_s + net_pro_loss_t)
#
#         clf_loss = clf_loss_func(pred_logit[mask_l], torch.argmax(torch.FloatTensor(yb[mask_l]), dim=1).to(device))
#         clf_loss = torch.sum(clf_loss) / np.sum(mask_l)
#         domain_loss = domain_loss_func(d_logit, torch.argmax(torch.FloatTensor(domain_label).to(device), 1))
#         total_loss = clf_loss + domain_loss + net_pro_loss
#         total_loss.backward()
#         optimizer.step()
#
#     model.eval()
#     with torch.no_grad():
#         GradientReverse.rate = 1.0
#         _, pred_logit_xs_xt, _ = model(feat, feat_ngh)
#         pred_prob_xs_xt = torch.argmax(pred_logit_xs_xt, dim=1)
#
#         src_acc = torch.sum(pred_prob_xs_xt[0:num_nodes_s] == torch.argmax(torch.FloatTensor(Y_s), dim=1).to(device)) / len(Y_s)
#         tgt_acc = torch.sum(pred_prob_xs_xt[-num_nodes_t:] == torch.argmax(torch.FloatTensor(Y_t), dim=1).to(device)) / len(Y_t)
#         print('epoch: {}, source accuracy: {}, target accuracy: {}'.format(epoch + 1, src_acc, tgt_acc))
