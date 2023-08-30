import numpy as np
from scipy.sparse import csc_matrix
import scipy.io as sio

net = sio.loadmat('data/dblpv7.mat')
X, A, Y = net['attrb'], net['network'], net['group']
adj = np.loadtxt('dblp_adj.txt', dtype=float, delimiter=',')
data = {'attrb': X, 'network': adj, 'group': Y}
sio.savemat('new_dblp.mat', data)
print('finished')