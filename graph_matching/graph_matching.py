import numpy as np
from scipy.spatial import KDTree
import networkx as nx




def nearestNeighborNode(G1, G2, num_nn = 1):
    # get positional information
    g1_pos = np.array([G1.nodes[node]["pos"] for node in G1.nodes])
    g2_pos = np.array([G2.nodes[node]["pos"] for node in G2.nodes])


    kd_sep = KDTree(g1_pos)
    res = kd_sep.query(g2_pos, k = num_nn)

    return res


def nearestNeighborLabeling(G_labeled, G_other, num_nn =1, copy = True):
    res = nearestNeighborNode(G_labeled, G_other, num_nn = num_nn)

    g1_dict = dict(zip(np.arange(G_labeled.order()),G_labeled.nodes()))
    g2_dict = dict(zip(np.arange(G_other.order()),G_other.nodes()))

    # apply labeling to the other graph
    g2_dict_labels = g2_dict.copy()

    for i, nn in enumerate(res[1]):
        g2_dict_labels[i] = str(g2_dict_labels[i]) + g1_dict[nn][-1]

    G_relabeled = nx.relabel_nodes(G = G_other, mapping = g2_dict_labels, copy = True)

    return G_relabeled


