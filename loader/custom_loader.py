from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import subgraph
import numpy as np
import torch
import random
	

class RandomWalkIterator:

    """Iterator that takes a list of graphs and performs random walks on them. Then it returns the resulting subgraphs as torch geom graphs."""

    def __init__(self, graph_list, neighborhood_size = 10, start_node_fraction = 0.1):
        self.iter_pos = 0
        self.graph_list = graph_list
        self.neighborhood_size = neighborhood_size
        self.start_node_fraction = start_node_fraction

    def __iter__(self):
        return self

    def __next__(self):
        iter_pos = self.iter_pos
        if iter_pos >= len(self.graph_list):
            self.iter_pos = 0
            raise StopIteration
            
        else:
            self.iter_pos += 1
            return self.generate_subgraphs(self.graph_list[iter_pos])
    
    def generate_subgraphs(self, graph):

        num_neighbors = np.ones([self.neighborhood_size], dtype= "int")*-1

        loader = NeighborLoader(
            graph,
            # Sample 30 neighbors for each node for 2 iterations
            input_nodes= torch.tensor(np.random.choice(graph.num_nodes, int(graph.num_nodes * self.start_node_fraction)), dtype= torch.long),
            num_neighbors=num_neighbors,
            batch_size=1,
            subgraph_type= "induced",
            shuffle = True
            #disjoint= True
        )

        subgraphs = []
        for data in loader:
            # only use subgraphs with more than 100 nodes
            if data.x.shape[0] >100:
                subgraphs.append(data)
            

        return subgraphs


class NeighborIterator:

    """Iterator that takes a list of graphs and induces subgraphs in them."""

    def __init__(self, graph_list, neighborhood_size = 10, start_node_fraction = 0.1):
        self.iter_pos = 0
        self.graph_list = graph_list
        self.neighborhood_size = neighborhood_size
        self.start_node_fraction = start_node_fraction

        self.induced_graph_y0 = []
        self.induced_graph_y1 = []

        self.stratified_selection = self.generate_stratified_selection()
        random.shuffle(self.stratified_selection)
        self.dataset = CustomGraphDataset(self.stratified_selection)
        #print(len(self.stratified_selection))


    def __iter__(self):
        return self

    def __next__(self):
        iter_pos = self.iter_pos
        if iter_pos >= len(self.stratified_selection):
            self.stratified_selection = self.generate_stratified_selection()
            random.shuffle(self.stratified_selection)
            self.dataset = CustomGraphDataset(self.stratified_selection)
            self.iter_pos = 0
            raise StopIteration
            
        else:
            self.iter_pos += 1
            return self.stratified_selection[iter_pos]
        
        
            
    
    def generate_stratified_selection(self):
        self.generate_subgraphs()

        #print(len(self.induced_graph_y0))
        #print(len(self.induced_graph_y1))

        smaller_frac = min(len(self.induced_graph_y0), len(self.induced_graph_y1))

        y0_selection = []
        y1_selection = []
        for ind in np.random.choice(len(self.induced_graph_y0),smaller_frac):
            y0_selection.append(self.induced_graph_y0[ind])

        for ind in np.random.choice(len(self.induced_graph_y1),smaller_frac):
            y1_selection.append(self.induced_graph_y1[ind])

        res = y0_selection + y1_selection
        #y0_selection.extend(y1_selection)
        #print(len(res))
        return res


    def generate_subgraphs(self):


        self.induced_graph_y0 = []
        self.induced_graph_y1 = []

        for graph in self.graph_list:

            num_neighbors = np.ones([self.neighborhood_size], dtype= "int")*-1


            loader = NeighborLoader(
                graph,
                # Sample 30 neighbors for each node for 2 iterations
                input_nodes= torch.tensor(np.random.choice(graph.num_nodes, int(graph.num_nodes * self.start_node_fraction)), dtype= torch.long),
                num_neighbors=num_neighbors,
                batch_size=1,
                subgraph_type= "induced",
                shuffle = True
                #disjoint= True
            )


            subgraphs = []
            for data in loader:
                # only use subgraphs with more than 100 nodes
                if data.x.shape[0] >100:
                    subgraphs.append(data)

            if graph.y[0] ==0:
                self.induced_graph_y0.extend(subgraphs)
            elif graph.y[0] ==1:
                self.induced_graph_y1.extend(subgraphs)




class RectangularIterator:

    """Iterator that takes a list of graphs and induces subgraphs."""

    def __init__(self, graph_list, selections_per_graph, rect_size, stratified = True, shuffle = True):
        self.iter_pos = 0
        self.graph_list = graph_list
        self.selections_per_graph = selections_per_graph
        self.rect_size = rect_size

        self.induced_graph_y0 = []
        self.induced_graph_y1 = []

        self.stratified = stratified
        self.shuffle = shuffle

        if self.stratified:
            self.selection = self.generate_stratified_selection()
        else:
            self.selection = self.generate_selection()

        if self.shuffle:
            random.shuffle(self.selection)
        self.dataset = CustomGraphDataset(self.selection)
        #print(len(self.stratified_selection))

        self.last_id_list = []
        self.last_region_list = []




    def __iter__(self):
        return self

    def __next__(self):
        iter_pos = self.iter_pos
        if iter_pos == 0:
            self.last_id_list = []
            self.last_region_list = []

        if iter_pos >= len(self.selection):

            # creates a new iterator every time the old one is exhausted

            if self.stratified:
                self.selection = self.generate_stratified_selection()
            else:
                self.selection = self.generate_selection()

            if self.shuffle:
                random.shuffle(self.selection)
            self.dataset = CustomGraphDataset(self.selection)
            self.iter_pos = 0
            raise StopIteration
            
        else:
            self.iter_pos += 1
            self.last_id_list.append(self.selection[iter_pos].graph_id)
            self.last_region_list.append(self.selection[iter_pos].region)
            return self.selection[iter_pos]
        
        
            
    
    def generate_stratified_selection(self):
        self.generate_subgraphs()


        smaller_frac = min(len(self.induced_graph_y0), len(self.induced_graph_y1))

        y0_selection = []
        y1_selection = []
        for ind in np.random.choice(len(self.induced_graph_y0),smaller_frac):
            y0_selection.append(self.induced_graph_y0[ind])

        for ind in np.random.choice(len(self.induced_graph_y1),smaller_frac):
            y1_selection.append(self.induced_graph_y1[ind])

        res = y0_selection + y1_selection

        #print(len(res))
        return res


    def generate_selection(self):
        self.generate_subgraphs()


        res = self.induced_graph_y0 + self.induced_graph_y1

        #print(len(res))
        return res

    def generate_subgraphs(self):


        self.induced_graph_y0 = []
        self.induced_graph_y1 = []

        for graph in self.graph_list:

            edge_positions = graph.edge_pos.detach().numpy()[:,:-1]

            rect_graphs = []

            max_x, min_x = np.max(edge_positions[:,0]), np.min(edge_positions[:,0])
            max_y, min_y = np.max(edge_positions[:,1]), np.min(edge_positions[:,1])

            #print(max_x, min_x, max_y, min_y)
            for _ in range(self.selections_per_graph):

                rdm_x = random.random()
                rdm_y = random.random()

                
                # choose the location of the rectangle with side length rect_size
                # at most half of the rectangle can be outside of the graph

                x_min = min_x + rdm_x*(max_x - min_x - self.rect_size)
                x_max = x_min + self.rect_size

                y_min = min_y + rdm_y*(max_y - min_y - self.rect_size)
                y_max = y_min + self.rect_size

                # induce a subgraph from the graph with all nodes that are inside the rectangle

                induced_nodes = []
                for i, pos in enumerate(edge_positions):
                    if x_min < pos[0] < x_max and y_min < pos[1] < y_max:
                        induced_nodes.append(i)

                #print(induced_nodes)
                
                subg_data = subgraph(induced_nodes, graph.edge_index, graph.edge_attr, relabel_nodes=True)

                subg = Data(x = graph.x[induced_nodes], edge_index = subg_data[0], edge_attr = subg_data[1], y = graph.y, graph_id = graph.graph_id, region = (x_min, x_max, y_min, y_max), edge_pos = graph.edge_pos[induced_nodes])

                rect_graphs.append(subg)
            
            if graph.y[0] ==0:
                self.induced_graph_y0.extend(rect_graphs)
            elif graph.y[0] ==1:
                self.induced_graph_y1.extend(rect_graphs)
        

                








class CustomGraphDataset(Dataset):
    def __init__(self,datalist):
        super().__init__()
        self.data_list = datalist


    def len(self):
        return len(self.data_list)

    def get(self, idx):
        data = self.data_list[idx]
        return data