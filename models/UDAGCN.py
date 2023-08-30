import random
import numpy as np
from tqdm import tqdm
from collections import Counter
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_geometric.nn.inits import glorot


class GraphConvolution(MessagePassing):
    def __init__(self, input_dim, output_dim, weight=None, bias=None, improved=False, **kwargs):
        super(GraphConvolution, self).__init__(aggr='add', **kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.improved = improved
        self.cache_dict = {}
        if weight is None:
            self.weight = Parameter(torch.FloatTensor(input_dim, output_dim))
            glorot(self.weight)
        else:
            self.weight = weight
            print('Use shared weight')
        if bias is None:
            self.bias = Parameter(torch.zeros(output_dim))
        else:
            self.bias = bias
            print('Use shared bias')

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), 1), dtype=dtype, device=edge_index.device)
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        # Accumulate the degree of nodes according to the weight of each edge
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, cache_name='default_cache', edge_weight=None):
        x = torch.matmul(x, self.weight)
        if not cache_name in self.cache_dict:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            self.cache_dict[cache_name] = edge_index, norm
        else:
            edge_index, norm = self.cache_dict[cache_name]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1)  * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out


class PPMIConvolution(GraphConvolution):
    def __init__(self, input_dim, output_dim, weight=None, bias=None, improved=False, path_len=5, device=None, **kwargs):
        super(PPMIConvolution, self).__init__(input_dim, output_dim, weight, bias, improved, **kwargs)
        self.path_len = path_len
        self.device = device

    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False, dtype=None):
        adj_dict = {}

        def add_edge(a, b):
            """{a: (b1, b2, b3)}"""
            if a in adj_dict:
                neighbors = adj_dict[a]
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        # construct adj_dict according to the edge_index
        for i in range(edge_index.shape[1]):
            add_edge(edge_index[0, i].item(), edge_index[1, i].item())
            add_edge(edge_index[1, i].item(), edge_index[0, i].item())
        # {a: [b1, b2, b3]}
        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}

        # sample a neighbor from its adjacent neighbors
        def sample_neighbor(a):
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]

        walk_counters = {}  # record the number of times that each nodes' distant neighbors being visited by random walk
        for _ in tqdm(range(40)):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, self.path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)  # random walk for given steps
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter
                    walk_counter[b] += 1
                    current_a = b

        def norm(counter):
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter  # normalized probability of each distant node being visited by random walk

        normalized_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}
        prob_sums = Counter()
        for a, normalized_walk_counter in normalized_walk_counters.items():
            for b, prob in normalized_walk_counter.items():
                prob_sums[b] += prob

        ppmis = {}
        for a, normalized_walk_counter in normalized_walk_counters.items():
            for b, prob in normalized_walk_counter.items():
                ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / self.path_len)
                ppmis[(a, b)] = ppmi

        new_edge_index, edge_weight = [], []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)
        edge_index = torch.tensor(new_edge_index).t().to(self.device)
        edge_weight = torch.tensor(edge_weight).to(self.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32)


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


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, output_dim, dropout, path_len=10, device=None):
        super(Encoder, self).__init__()
        self.gcn1 = GraphConvolution(input_dim, hid_dim)
        self.gcn2 = GraphConvolution(hid_dim, output_dim)
        self.dropout = dropout
        self.ppm1 = PPMIConvolution(input_dim, hid_dim, weight=self.gcn1.weight, bias=self.gcn1.bias, path_len=path_len,
                                    device=device)
        self.ppm2 = PPMIConvolution(hid_dim, output_dim, weight=self.gcn2.weight, bias=self.gcn2.bias,
                                    path_len=path_len, device=device)
        self.dense_weight = nn.Linear(output_dim, 1)

    def forward(self, x, edge_index, cache_name):
        gcn_x = self.gcn1(x, edge_index, cache_name)
        gcn_x = F.dropout(F.relu(gcn_x), self.dropout, training=self.training)
        gcn_x = self.gcn2(gcn_x, edge_index, cache_name)
        ppm_x = self.ppm1(x, edge_index, cache_name)
        ppm_x = F.dropout(F.relu(ppm_x), self.dropout, training=self.training)
        ppm_x = self.ppm2(ppm_x, edge_index, cache_name)
        stacked = torch.stack((gcn_x, ppm_x), dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        output = torch.sum(stacked * weights, dim=1)
        return output


class UDAGCN(nn.Module):  # disc_dim: 40
    def __init__(self, input_dim, hid_dim=128, emb_dim=16, disc_dim=40, num_class=6, dropout=0.1, path_len=10,
                 device=None):
        super(UDAGCN, self).__init__()
        self.encoder = Encoder(input_dim, hid_dim, emb_dim, dropout, path_len=path_len, device=device)
        self.node_classifier = nn.Linear(emb_dim, num_class)
        self.domain_discriminator = nn.Sequential(GRL(), nn.Linear(emb_dim, disc_dim), nn.ReLU(), nn.Dropout(dropout),
                                                  nn.Linear(disc_dim, 2))

    def forward(self, x_src, edge_index_src, cache_name_src, x_tgt, edge_index_tgt, cache_name_tgt):
        emb_src = self.encoder(x_src, edge_index_src, cache_name_src)
        emb_tgt = self.encoder(x_tgt, edge_index_tgt, cache_name_tgt)
        emb = torch.cat((emb_src, emb_tgt), dim=0)
        pred_logit = self.node_classifier(emb)
        d_logit = self.domain_discriminator(emb)
        return pred_logit, d_logit