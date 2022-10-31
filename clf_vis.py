import pandas as pd
import numpy as np
from scipy.sparse.csgraph import connected_components
import preprocessing as pp
from visualization import graph_to_mesh, mesh_viewer
from torch_geometric.utils.convert import from_networkx
from models import graphClassifier, nodeClassifier
from tqdm import tqdm
import sampling
import networkx as nx
import random
import torch
import pyvista
import trimesh
import pyvistaqt as pvqt


nodesFileNerve =  "~/Documents/Intestine/nerve-mask/nodes_nerve_bs2.csv"
edgesFileNerve = "~/Documents/Intestine/nerve-mask/edges_nerve_bs2.csv"

nodesFileLymph =  "~/Documents/Intestine/lymph-mask/nodes_lymph_bs2.csv"
edgesFileLymph = "~/Documents/Intestine/lymph-mask/edges_lymph_bs2.csv"

nodes_n = pd.read_csv(nodesFileNerve, sep = ";", index_col= "id")
edges_n = pd.read_csv(edgesFileNerve, sep = ";", index_col= "id")
nodes_l = pd.read_csv(nodesFileLymph, sep = ";", index_col= "id")
edges_l = pd.read_csv(edgesFileLymph, sep = ";", index_col= "id")


# scaling with the factors provided by luciano
nodes_l = pp.scalePosition(nodes_l, (1.65,1.65,6))
nodes_n = pp.scalePosition(nodes_n, (1.65,1.65,6))

# giving nodes from different files unique names
edges_n, nodes_n = pp.relable_edges_nodes(edges_n, nodes_n, "n")
edges_l, nodes_l = pp.relable_edges_nodes(edges_l, nodes_l, "l")


adjMcsr = pp.distance_based_adjacency(nodes_n, nodes_l, th = 0.03)
num, labels = connected_components(csgraph=adjMcsr, directed = False)
con_comp = pp.connected_components_dict(labels)
rel_comp = pp.relevant_connected_components(con_comp, nodes_n.shape[0], ("n","l"))

reverse_dict= {}
for k, v in rel_comp.items():
    for val in v:
        reverse_dict[val] = k



merged_nodes = pd.concat([nodes_l.loc[:,["pos_x", "pos_y", "pos_z"]], nodes_n.loc[:,["pos_x", "pos_y", "pos_z"]]])
new_nodes = pd.DataFrame(columns=merged_nodes.columns )


# replace the contracted node with new nodes
# the position of the new node is an average of all the previous nodes
for k, valList in rel_comp.items():
    new_nodes.loc[k] = merged_nodes.loc[valList].mean()
    merged_nodes.drop(valList, inplace = True)

# concat the all nodes and the new nodes
merged_nodes = pd.concat([merged_nodes, new_nodes])

# createa a combined edge file
merged_edges = pd.concat([edges_l, edges_n], ignore_index = True)

# change the names of the edges to the new names
for idxE, edge in merged_edges.iterrows():
    if edge["node1id"] in reverse_dict.keys():
        merged_edges.loc[idxE,"node1id"] = reverse_dict[edge["node1id"]]
    if edge["node2id"] in reverse_dict.keys():
        merged_edges.loc[idxE,"node2id"] = reverse_dict[edge["node2id"]]


# create a new graph based on the old information

G_contract = pp.createGraph(merged_nodes, merged_edges)
G_contract_einf = pp.convertToEinfach(G_contract, self_loops = False, isolates = False)
pp.enrichNodeAttributes(G_contract_einf)



# create the ground truth for the node class
all_nodes = list(G_contract_einf.nodes)
nerve_class = np.array([elem[-1] == "n" for elem in all_nodes])*0
lymph_class = np.array([elem[-1] == "l" for elem in all_nodes])*1
combined_class = np.array([elem[-1] == "c" for elem in all_nodes])*2
class_assign = nerve_class+lymph_class+combined_class


# create the training and testing masks
train_mask = np.random.choice(np.arange(0, len(class_assign)), size= int(len(class_assign)*0.8), replace = False)
test_mask = np.delete(np.arange(0, len(class_assign)), train_mask)

# convert to torch tensor objects
train_mask= torch.tensor(train_mask)
test_mask= torch.tensor(test_mask)

# convert the graph to a networkx graph
networkXG = from_networkx(G_contract_einf)
networkXG.y = torch.tensor(class_assign)


num_feat = networkXG.x.shape[1]
num_class = len(np.unique(networkXG.y))
netGCN = nodeClassifier.GCN_NC
netSAGE = nodeClassifier.SAGE_NC
netWC = nodeClassifier.WC_NC

optimizer = torch.optim.Adam
criterion = torch.nn.CrossEntropyLoss
modelGCN = nodeClassifier.nodeClassifier(netGCN, hidden_channels=32, features = np.arange(num_feat),classes = num_class, optimizer = optimizer, lossFunc = criterion)
modelSAGE = nodeClassifier.nodeClassifier(netSAGE, hidden_channels=32, features = np.arange(num_feat),classes = num_class, optimizer = optimizer, lossFunc = criterion)
modelWC = nodeClassifier.nodeClassifier(netWC, hidden_channels=32, features = np.arange(num_feat),classes = num_class, optimizer = optimizer, lossFunc = criterion)







def classifiedGraph(G,pred_mask, test_mask):
    corr = np.where(pred_mask>0)[0]
    wrong = np.where(pred_mask==0)[0]

    corr_nodes = np.array(G.nodes)[test_mask][corr]
    wrong_nodes = np.array(G.nodes)[test_mask][wrong]

    wrong_dict = dict(zip(corr_nodes, [elem + "r" for elem in corr_nodes]))
    right_dict = dict(zip(wrong_nodes, [elem + "f" for elem in wrong_nodes]))

    G_class = nx.relabel_nodes(G, wrong_dict)
    G_class = nx.relabel_nodes(G_class, right_dict)

    return G_class


test_mask_np = test_mask.detach().numpy()
train_mask_np = train_mask.detach().numpy()

start_pred = np.zeros_like(test_mask_np)
G_class = classifiedGraph(G_contract_einf, start_pred, test_mask)

nodeMeshesTrain, unique_nodesTrain = graph_to_mesh.nodeMeshesNx(G_class, concat = True, mask = train_mask_np)
nodeMeshesTest, unique_nodesTest = graph_to_mesh.nodeMeshesNx(G_class, concat = False, mask = test_mask_np)
edgeMeshes, unique_edges = graph_to_mesh.edgeMeshesNx(G_class, concat = True)


node_color_dict = dict(zip(['c', 'f', 'n', 'r', 'l'],["blue", "red", "hotpink", "green", "yellow"]))

#plotter = pyvista.Plotter()
plotter = pvqt.BackgroundPlotter()


edge_actors = []
for i in range(len(edgeMeshes)):
    mesh = pyvista.wrap(edgeMeshes[i])
    actor = plotter.add_mesh(mesh, color = "beige")
    edge_actors.append(actor)
print("Edges Done!")



node_train_actors = []
for i in range(len(nodeMeshesTrain)):
    color = node_color_dict[unique_nodesTrain[i]]
    mesh = pyvista.wrap(nodeMeshesTrain[i])
    actor = plotter.add_mesh(mesh, color = color)
    node_train_actors.append(actor)
print("Train Nodes Done!")




node_test_actors =[]
for i in range(len(nodeMeshesTest)):
    color = node_color_dict[unique_nodesTest[i]]
    med_actors=[]
    pbar = tqdm(total = len(nodeMeshesTest[i]))
    for mesh in nodeMeshesTest[i]:
        pbar.update(1)
        mesh = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_normals=mesh.face_normals, vertex_normals=mesh.vertex_normals)
        meshW = pyvista.wrap(mesh)
        actor = plotter.add_mesh(meshW, color = color)
        med_actors.append(actor)
    node_test_actors.append(med_actors)
    pbar.close()
print("Test Nodes Done!")

print(len(node_test_actors))

epoch_text = plotter.add_text("Epoch 0")

plotter.show(interactive_update = True)

epochs = 200
for epoch in range(1, epochs +1):
    loss = modelSAGE.train(networkXG, train_mask)
    #test_acc = modelSAGE.test(networkXG, test_mask)
    #acc_l[epoch-1, i] = test_acc
    pred_mask = modelSAGE.predictions(networkXG, test_mask)
    color = (random.random(),random.random(),random.random())
    for node in node_test_actors[0]:
        node.GetProperty().SetColor(color)
    input("Press Enter for the next epoch...")
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    plotter.remove_actor(epoch_text)
    epoch_text = plotter.add_text("Epoch "+ str(epoch))

    plotter.update()



