import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
import scipy
from scipy.sparse import lil_matrix, csc_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from warnings import filterwarnings
filterwarnings('ignore')


def load_mat_data(file):
    net = sio.loadmat(file)
    X, A, Y = net['attrb'], net['network'], net['group']
    return X, A, Y

def preprocess_adj(src_adj, tgt_adj):
    """Preprocessing adjacency matrix for GCN model (Merge two graphs and Laplacian normalize it)"""
    src_adj, tgt_adj = src_adj + sp.eye(src_adj.shape[0]), tgt_adj + sp.eye(tgt_adj.shape[0])
    edge_src, edge_tgt = src_adj.nonzero(), tgt_adj.nonzero()
    row = np.concatenate((edge_src[0], (edge_tgt[0] + src_adj.shape[0])))
    col = np.concatenate((edge_src[1], (edge_tgt[1] + src_adj.shape[0])))
    data = np.ones(len(row))
    adj_mx = sp.coo_matrix((data, (row, col)), shape=(src_adj.shape[0] + tgt_adj.shape[0], src_adj.shape[0] + tgt_adj.shape[0]))
    rowsum = np.array(adj_mx.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = adj_mx.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    adj_sp_tensor = torch.sparse_coo_tensor([normalized_adj.nonzero()[0], normalized_adj.nonzero()[1]],
                                            normalized_adj.data, (normalized_adj.shape), dtype=torch.float32)
    return adj_sp_tensor

def row_normalize_mat(feat):
    """Row normalize a matrix"""
    rowsum = np.array(feat.sum(1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    feat = r_mat_inv.dot(feat)
    return feat

def load_networks(src_file, tgt_file):
    X_s, A_s, Y_s = load_mat_data(src_file)
    X_t, A_t, Y_t = load_mat_data(tgt_file)
    # X_s, X_t = row_normalize_mat(X_s), row_normalize_mat(X_t)
    # X_s, X_t = normalize(X_s), normalize(X_t)
    # X = normalize(np.vstack((X_s, X_t)))
    adj = preprocess_adj(A_s, A_t)
    feature = torch.FloatTensor(np.vstack((X_s, X_t)))
    labels = torch.FloatTensor(np.vstack((Y_s, Y_t)))
    train_idx, test_idx = torch.LongTensor(np.arange(len(X_s))), torch.LongTensor(np.arange(len(X_t)) + len(X_s))
    return feature, adj, labels, train_idx, test_idx

def multi_label_acc(pred_logit, labels, test_idx, threshold=0.5, epsilon=1e-8):
    y_pred = torch.sigmoid(pred_logit[test_idx])
    y_true = labels[test_idx]
    y_pred = torch.where(y_pred > threshold, 1, 0)
    micro_TP = torch.count_nonzero(y_pred * y_true).float()
    micro_FP = torch.count_nonzero(y_pred * (1 - y_true)).float()
    micro_FN = torch.count_nonzero((y_pred - 1) * y_true).float()
    precision = micro_TP / (micro_TP + micro_FP + epsilon)
    recall = micro_TP / (micro_TP + micro_FN + epsilon)
    micro_f1 = torch.mean(2 * precision * recall / (precision + recall + epsilon))
    macro_TP = torch.count_nonzero(y_pred * y_true, dim=1).float()
    macro_FP = torch.count_nonzero(y_pred * (1 - y_true), dim=1).float()
    macro_FN = torch.count_nonzero((y_pred - 1) * y_true, dim=1).float()
    precision = macro_TP / (macro_TP + macro_FP + epsilon)
    recall = macro_TP / (macro_TP + macro_FN + epsilon)
    macro_f1 = torch.mean(2 * precision * recall / (precision + recall + epsilon))
    return micro_f1, macro_f1

def load_acm_dblp(file_name):
    file_path = '../data/' + file_name + '/raw/'
    feat = np.loadtxt(file_path + file_name + '_docs.txt', dtype=float, delimiter=',')
    edge_index = np.loadtxt(file_path + file_name + '_edgelist.txt', dtype=int, delimiter=',')
    labels = np.loadtxt(file_path + file_name + '_labels.txt', dtype=int)
    feat, labels = torch.FloatTensor(feat), torch.LongTensor(labels)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return feat, edge_index, labels

def load_udagcn_data(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    feat = torch.FloatTensor(x)
    adj = torch.FloatTensor(a.toarray()).nonzero().T
    label = torch.LongTensor(np.argmax(y, axis=1))
    return feat, adj, label

def load_network(file):
    net = sio.loadmat(file)
    x, a, y = net['attrb'], net['network'], net['group']
    if not isinstance(x, scipy.sparse.lil_matrix):
        x = lil_matrix(x)
    return a, x, y

def aggregate_trans_prob(adj, hop):
    """aggregated k-step transition probability"""
    adj = row_normalize_mat(adj).toarray()
    a_k, a = adj, adj
    for k in np.arange(2, hop + 1):
        a_k = np.matmul(a_k, adj)
        a = a + a_k / k
    return a

def compute_ppmi(a):
    """compute PPMI based on aggregated k-step transition probability matrix"""
    np.fill_diagonal(a, 0)
    a = row_normalize_mat(a)
    (p, q) = np.shape(a)
    col = np.sum(a, axis=0)
    col[col == 0] = 1
    ppmi = np.log((float(p) * a) / col[None, :])
    idx_nan = np.isnan(ppmi)
    ppmi[idx_nan] = 0
    ppmi[ppmi < 0] = 0
    ppmi = row_normalize_mat(ppmi)
    return ppmi

def shuffle_aligned_list(data):
    num = data[0].shape[0]
    shuffle_index = np.random.permutation(num)
    return shuffle_index, [d[shuffle_index] for d in data]


def batch_generator(data, batch_size, shuffle=True):
    shuffle_index = None
    if shuffle:
        shuffle_index, data = shuffle_aligned_list(data)
    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= data[0].shape[0]:
            batch_count = 0
            if shuffle:
                shuffle_index, data = shuffle_aligned_list(data)
        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data], shuffle_index[start:end]

def batch_ppmi(batch_size, shuffle_index_s, shuffle_index_t, ppmi_s, ppmi_t):
    """return the PPMI matrix between nodes in each batch"""
    # #proximity matrix between source network nodes in each mini-batch
    # noinspection DuplicatedCode
    a_s = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_s[ii, jj] = ppmi_s[shuffle_index_s[ii], shuffle_index_s[jj]]
    # #proximity matrix between target network nodes in each mini-batch
    # noinspection DuplicatedCode
    a_t = np.zeros((int(batch_size / 2), int(batch_size / 2)))
    for ii in range(int(batch_size / 2)):
        for jj in range(int(batch_size / 2)):
            if ii != jj:
                a_t[ii, jj] = ppmi_t[shuffle_index_t[ii], shuffle_index_t[jj]]
    return row_normalize_mat(a_s), row_normalize_mat(a_t)


def f1_scores(y_pred, y_true):
    def predict(y_tru, y_pre):
        top_k_list = np.array(np.sum(y_tru, 1), np.int32)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = np.zeros(y_tru.shape[1])
            pred_i[np.argsort(y_pre[i, :])[-top_k_list[i]:]] = 1
            prediction.append(np.reshape(pred_i, (1, -1)))
        prediction = np.concatenate(prediction, axis=0)
        return np.array(prediction, np.int32)

    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true, predictions, average=average)
    return results["micro"], results["macro"]


def load_asn_data(file, target=False):
    feat, adj, label = load_mat_data(file)
    """compute ppmi"""
    A_k = aggregate_trans_prob(adj, 3)
    PPMI = compute_ppmi(A_k)
    ngh_PPMI = row_normalize_mat(PPMI)
    X_n = torch.FloatTensor(ngh_PPMI).to_sparse()

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)  # symmetrize
    feat = torch.FloatTensor(row_normalize_mat(feat))
    adj = row_normalize_mat(adj + sp.eye(adj.shape[0]))
    label = torch.LongTensor(np.argmax(label, axis=1))
    adj = torch.FloatTensor(adj.toarray()).to_sparse()
    if target:
        adj = torch.eye(feat.shape[0], dtype=torch.float32).to_sparse()
    idx = torch.LongTensor(np.random.permutation(len(feat)))
    return adj, feat, label, idx, X_n

def load_adj_label(file, target=False):
    _, adj, _ = load_mat_data(file)
    if target:
        adj = csc_matrix(np.eye(adj.shape[0]))
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = torch.FloatTensor(adj_label.toarray())
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight = np.array(pos_weight).reshape(1, 1)
    pos_weight = torch.from_numpy(pos_weight)
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    return adj_label, pos_weight, norm
