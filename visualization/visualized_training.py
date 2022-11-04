import numpy as np
import torch
from  models import nodeClassifier
from visualization import graph_to_mesh, mesh_viewer
from torch_geometric.utils.convert import from_networkx
import pyvista
from tqdm import tqdm
import trimesh
import pyvistaqt as pvqt



def createClassLabels(G):
    all_nodes = list(G.nodes)
    nerve_class = np.array([elem[-1] == "n" for elem in all_nodes])*0
    lymph_class = np.array([elem[-1] == "l" for elem in all_nodes])*1
    combined_class = np.array([elem[-1] == "c" for elem in all_nodes])*2
    class_assign = nerve_class+lymph_class+combined_class

    return class_assign


def changedNodes(node_dict, corr, wrong):
    changed_dict = {}
    for node in corr:
        if node_dict[node][1] == "r":
            continue
        else:
            changed_dict[node] = "r"
    for node in wrong:
        if node_dict[node][1] == "f":
            continue
        else:
            changed_dict[node] = "f"

    return changed_dict


def adjustNodeColor(node_dict, changed_dict):
    for changed_node, val in changed_dict.items():
        if val == "f":
            color = (1,0,0)
        if val == "r":
            color = (0,1,0)
        node_dict[changed_node][0].GetProperty().SetColor(color)


def handleInput(indicator):
    try:
        indicator = int(indicator)
    except ValueError:
        print("Not a valid choice! Going to the next epoch!")
        indicator = 1
    if indicator <0:
        print("Not a valid choice! Going to the next epoch!")
        indicator= 1

    return indicator

import networkx as nx
def classifiedGraph(G,test_mask_np, pred_mask):
    nodes = np.array(list(G.nodes))
    relevant_nodes = nodes[test_mask_np]

    pred_dict = {}
    pred_dict[0] = "f"
    pred_dict[1] = "r"
    new_names = {}
    for i, node_name in enumerate(relevant_nodes):


        lab = node_name[-1]
        if lab not in ("f", "r"):
            new_node_name = node_name + pred_dict[pred_mask[i]]
            new_names[node_name] = new_node_name
        elif lab != pred_dict[pred_mask[i]]:
            new_node_name = node_name[:len(node_name)-1] + pred_dict[pred_mask[i]]#+ pred_reverse[pred_mask[i]]
            new_names[node_name] = new_node_name

    G_class = nx.relabel_nodes(G, new_names)

    return G_class


def visualizedTrainingMesh(G, Classifier, interactive = False):


    class_assign = createClassLabels(G)
    node_num = G.order()

    # create the training and testing masks
    train_mask_np = np.random.choice(np.arange(0, node_num), size= int(node_num*0.8), replace = False)
    test_mask_np = np.delete(np.arange(0, node_num), train_mask_np)

    # convert to torch tensor objects
    train_mask= torch.tensor(train_mask_np)
    test_mask= torch.tensor(test_mask_np)

    # convert the graph to a networkx graph
    networkXG = from_networkx(G)
    networkXG.y = torch.tensor(class_assign)

    nodeMeshesTrain, unique_nodesTrain,test_node_identifiers = graph_to_mesh.nodeMeshesNx(G, concat = True, mask = train_mask_np, node_rad = 0.008)
    nodeMeshesTest, unique_nodesTest, test_node_identifiers = graph_to_mesh.nodeMeshesNx(G, concat = False, mask = test_mask_np, node_rad = 0.012)
    edgeMeshes, unique_edges = graph_to_mesh.edgeMeshesNx(G, concat = True)

    node_color_dict = dict(zip(['c', 'f', 'n', 'r', 'l'],["blue", "red", "hotpink", "green", "yellow"]))


    if not interactive:
        plotter = pyvista.Plotter()
    else:
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



    node_test_actors ={}
    for i in range(len(nodeMeshesTest)):
        color = node_color_dict[unique_nodesTest[i]]
        identifiers = test_node_identifiers[i]
        pbar = tqdm(total = len(nodeMeshesTest[i]))
        for i, mesh in enumerate(nodeMeshesTest[i]):
            pbar.update(1)
            mesh = trimesh.base.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_normals=mesh.face_normals, vertex_normals=mesh.vertex_normals)
            meshW = pyvista.wrap(mesh)
            actor = plotter.add_mesh(meshW, color = color)
            node_test_actors[identifiers[i]] = (actor, "s")
        pbar.close()
    print("Test Nodes Done!")



    epoch_text = plotter.add_text("Epoch 0")

    if not interactive:
        plotter.show(interactive_update = True)


    input("Press enter to start with the first epoch...")
    epochs = 200

    indicator = 1
    for epoch in range(1, epochs +1):
        indicator = indicator -1
        loss = Classifier.train(networkXG, train_mask)
        #test_acc = Classifier.test(networkXG, test_mask)
        #acc_l[epoch-1, i] = test_acc

        if indicator == 0:
            pred_mask = Classifier.predictions(networkXG, test_mask).detach().numpy()

            corr_pred = np.array(G.nodes)[test_mask_np][pred_mask]
            wrong_pred = np.delete(np.array(G.nodes)[test_mask_np],pred_mask)

            changed_dict = changedNodes(node_test_actors,corr_pred,wrong_pred )
            adjustNodeColor(node_test_actors, changed_dict)

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            plotter.remove_actor(epoch_text)
            epoch_text = plotter.add_text("Epoch "+ str(epoch))

            indicator = handleInput(input("Enter number of epochs to skip..."))

        plotter.update()





def visualizedTraining(G, Classifier,epochs = 100, interactive = False):


    class_assign = createClassLabels(G)
    node_num = G.order()

    # create the training and testing masks
    train_mask_np = np.random.choice(np.arange(0, node_num), size= int(node_num*0.8), replace = False)
    test_mask_np = np.delete(np.arange(0, node_num), train_mask_np)

    # convert to torch tensor objects
    train_mask= torch.tensor(train_mask_np)
    test_mask= torch.tensor(test_mask_np)

    # convert the graph to a networkx graph
    networkXG = from_networkx(G)
    networkXG.y = torch.tensor(class_assign)

    input("Press enter to start with the first epoch...")

    for epoch in range(1, epochs +1):
        loss = Classifier.train(networkXG, train_mask)
        #test_acc = Classifier.test(networkXG, test_mask)
        #acc_l[epoch-1, i] = test_acc

        pred_mask = Classifier.predictions(networkXG, test_mask).detach().numpy()
        G_class = classifiedGraph(G, test_mask_np,pred_mask)
        epoch_text = "Epoch "+ str(epoch) + ".png"


        mesh_viewer.renderNXGraph(G_class, pic = epoch_text)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')


