import numpy as np
from scipy.spatial import KDTree
import networkx as nx




def nearestNeighborNode(G1, G2, num_nn = 1):
    ''' Returns for every node in G2 the spatially closest node in G1

    Parameters
    ----------

    G1: A networkx graph
    G2: A networkx graph
    num_nn: The number of closest nodes that are returned
    -------

    res: Closest nodes for every node in G2

     '''
    # get positional information
    g1_pos = np.array([G1.nodes[node]["pos"] for node in G1.nodes])
    g2_pos = np.array([G2.nodes[node]["pos"] for node in G2.nodes])


    kd_sep = KDTree(g1_pos)
    res = kd_sep.query(g2_pos, k = num_nn)

    return res





def nearestNeighborLabeling(G_labeled, G_other, num_nn =1, copy = True):
    ''' Returns a labeled graph for G_other with the label relying on G_labeled.

    Parameters
    ----------

    G_labeled: A labeled networkx graph (label ist last char of name/id)
    G_other: A networkx graph.
    num_nn: The number of closest nodes that are used for the labelig
    copy: indicated if a labeled copy of G_other will be returned
    -------

    res: Closest nodes for every node in G2

     '''
    res = nearestNeighborNode(G_labeled, G_other, num_nn = num_nn)

    g1_dict = dict(zip(np.arange(G_labeled.order()),G_labeled.nodes()))
    g2_dict = dict(zip(np.arange(G_other.order()),G_other.nodes()))

    # apply labeling to the other graph
    g2_dict_labels = g2_dict.copy()

    for i, nn in enumerate(res[1]):
        g2_dict_labels[i] = str(g2_dict_labels[i]) + g1_dict[nn][-1]

    G_relabeled = nx.relabel_nodes(G = G_other, mapping = g2_dict_labels, copy = True)

    return G_relabeled





def assignNodeLabelsByMask(maskList, G, voxel_size = (1,1,1), scaling_vector = (1,1,1), assign_type = "max", kernel_size = 5):
    ''''Returns labels for the nodes in a graph based on the presence of labels ins a provided mask list.

    Parameters
    ----------

    maskList: A list of masks with the same dimension in shape: (x, y ,z)
    G: A graph which nodes will be classified 
    pixel_size: A vector for the size of each voxel.
    scaling_vector: A scaling vector that may have been applied to the node coordinates. 
    assign_type : A string indicating how the assigning based on the mask should be performed. Possible: "max", "mixed"
    kernel_size: The size of the kernel that scans the masks at the node positions. Automatically adjusting for differences in the dimensions.

    Returns
    -------

    class_dict: A dictionary that assigns a class to every node of G. Class assign starts from the first provided mask with 0. If there is no mask at the node position the label -1 is the class value.
     '''


    shape_array = np.array([mask.shape for mask in maskList])
    eq_shape = np.all(shape_array == shape_array[0])

    if not eq_shape:
        raise ValueError("Not all the masks have the same shape")


    sizes = np.array(voxel_size)* np.array(scaling_vector)
    mask_center = np.array(shape_array[0]/2)


    min_size = min(sizes)
    norm_sizes = sizes/min_size
    adj_kernel_size = np.array((kernel_size,kernel_size,kernel_size)) / np.array(norm_sizes)
    adj_kernel_size = np.ceil(adj_kernel_size)
    for i, num in enumerate(adj_kernel_size):
        if num % 2 == 0:
            adj_kernel_size[i] = num-1
    adj_kernel_size = np.ceil(adj_kernel_size/2)
    adj_kernel_size = adj_kernel_size.astype(int)



    label_dict = {}
    for node in G.nodes():
        node_pos = G.nodes[node]["pos"]


        offset = np.rint(np.array(node_pos)/ sizes)
        node_pos_idx = offset + mask_center
        node_pos_idx = node_pos_idx.astype("int")

        max_val = 0
        max_idx = -1

        lowerX, upperX = node_pos_idx[0]-adj_kernel_size[0]+1, node_pos_idx[0]+adj_kernel_size[0]
        lowerY, upperY = node_pos_idx[1]-adj_kernel_size[1]+1, node_pos_idx[1]+adj_kernel_size[1]
        lowerZ, upperZ = node_pos_idx[2]-adj_kernel_size[2]+1, node_pos_idx[2]+adj_kernel_size[2]

        lowerX = lowerX if lowerX >=0 else 0 
        lowerY = lowerY if lowerY >=0 else 0 
        lowerZ = lowerZ if lowerZ >=0 else 0

        for i, mask in enumerate(maskList):
            val = np.sum(mask[lowerX: upperX, lowerY: upperY, lowerZ:upperZ])
            if val > max_val:
                max_val = val 
                max_idx = i 
            elif val <0:
                print(val)

        label_dict[node] = max_idx

    return label_dict



def labelPrimalFromDual(G, L, L_lab_attr, conv_dict, G_lab_attr = None):
    ''''Labels a Graph G based on the labeling that already exists for L. 

    Parameters
    ----------
    G: Primal networkx graph that should be labeled. Used to infer labels.
    L: Dual networkx graph that is labeled.
    L_lab_attr: the attribute that encodes the class information
    conv_dict: A dict that transers the numeric class to characters
    G_lab_attr: the attribute that will be used to store the transfered classifcation. If not specified the name for L_lab_attr is used.

    Returns
    -------

    unique: A boolean if the label is unique or not
    '''

    rev_dict = {}
    rev_dict["n"] = "l"
    rev_dict["l"] = "n"

    res_class = {}
    remain_nodes = []
    for node in L.nodes():
        label_num = L.nodes[node][L_lab_attr]
        label_str = conv_dict[label_num]
        if len(label_str) ==1:
            res_class[node[0]] = label_str
            res_class[node[1]] = label_str
        else: 
            remain_nodes.append(node)


    unlogic_labeling = 0
    change = True
    while len(remain_nodes) > 0 and change:
        change = False
        for node in remain_nodes:
            f1 = node[0]
            f2 = node[1]
            try:
                f1lab = res_class[f1]
            except KeyError:
                f1lab = None
            try:
                f2lab = res_class[f2]
            except KeyError:
                f2lab = None

            sc = int(f1lab is None) + int(f2lab is None)


            if sc ==2:
                continue
            elif sc == 0:
                change = True
                if f1lab == f2lab:
                    unlogic_labeling += 1
                    remain_nodes.remove(node)
                else:
                    remain_nodes.remove(node)
            elif sc == 1:
                change = True
                if f1lab is None:
                    res_class[f1] = rev_dict[f2lab]
                else:
                    res_class[f2] = rev_dict[f1lab]

    if G_lab_attr is None:
        res_store = L_lab_attr
    else:
        res_store = G_lab_attr

    nx.set_node_attributes(G, res_class, res_store)

    if unlogic_labeling > 0:
        print("A total of " + str(unlogic_labeling) + " unlogical labelings occured.")




def labelPrimalFromDualFreq(G, L, L_lab_attr, conv_dict, G_lab_attr = None):
    ''''Labels a Graph G based on the labeling that already exists for L. The labels of the node is assigned based on the majority of classes of the adjacent edges

    Parameters
    ----------
    G: Primal networkx graph that should be labeled. Used to infer labels.
    L: Dual networkx graph that is labeled.
    L_lab_attr: the attribute that encodes the class information
    conv_dict: A dict that transers the numeric class to characters
    G_lab_attr: the attribute that will be used to store the transfered classifcation. If not specified the name for L_lab_attr is used.

    Returns
    -------

    unique: A boolean if the label is unique or not
    '''

    

    # create a dict with all nodes and a dict within that

    majority_dict = {}
    for node in G.nodes():
        majority_dict[node] = {"l": 0, "n" : 0}

    rev_dict = {}
    rev_dict["n"] = "l"
    rev_dict["l"] = "n"

    for node in L.nodes():
        label_num = L.nodes[node][L_lab_attr]
        label_str = conv_dict[label_num]
        if len(label_str) == 1:
            majority_dict[node[0]][label_str] +=1 
            majority_dict[node[1]][label_str] +=1
        else: 
            majority_dict[node[0]][label_str[0]] += 0.5
            majority_dict[node[0]][label_str[1]] += 0.5
            majority_dict[node[1]][label_str[0]] += 0.5
            majority_dict[node[1]][label_str[1]] += 0.5

    res_dict = {}
    for k,v in majority_dict.items():
        majority_dict[k] 
        res = max(majority_dict[k] , key=majority_dict[k].get)
        res_dict[k] = res

    if G_lab_attr is None:
        res_store = L_lab_attr
    else:
        res_store = G_lab_attr
    
    nx.set_node_attributes(G, res_dict, res_store)
