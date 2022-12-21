import numpy as np
import random
from tqdm import tqdm
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx



class SubgraphSampler():


    def __init__(self, mode):
        # one of rnn, rw or rj 
        self.mode = mode 


    def set_mode(self, mode):
        if mode not in ["rnn", "rw", "rj"]:
            raise ValueError("Not a valid subsampling mode.")
        else:
            self.mode = mode


    def random_node_neighbor(self, graph, sample_size, start_nodes):
        """
        Performs a random choice from all the nodes in the neighborhood that are not discovered yet.
        Nodes in the neighborhood that are connected by multiple discovered node have a proportionally higher probability to be discoered.
        If starting from a node the "sample_size" can not be reached (no. nodes in the connected component < sample_size) then not resulting "node_set" is returned

        graph: a networkX graph
        sample_size: the number of nodes that should finally be contained
        start_nodes: an iterable of all the nodes from which the sampling process starts

        return node_sets: the resulting node sets for the start_nodes. 

        """
        node_sets = []
        pbar = tqdm(start_nodes)
        pbar.set_description("Creating subgraphs using random node neighbor selection.")
        for start_node in pbar:

            # get the size of the connected component of the start node
            comp_size = len(nx.node_connected_component(graph, start_node))

            # skip if the cc is not as large as the sample size
            if comp_size < sample_size:
                continue
            size = 0 
            neigborhood = []
            discovered = {start_node}
            neigborhood += graph.neighbors(start_node)
            while len(discovered) < sample_size:
                if len(neigborhood)<1:
                    #print("entered")
                    break
                new_disc = random.choice(neigborhood)
                neigborhood = list(filter(lambda a: a != new_disc, neigborhood))
                discovered.add(new_disc)

                new_neighbors = list(graph.neighbors(new_disc))
                new_neighbors = list(filter(lambda a: a not in discovered, new_neighbors))
                neigborhood += new_neighbors


            node_sets.append(discovered)
        pbar.close()

        return node_sets




    def random_walk(self, graph, sample_size, start_nodes, c = 0.15):
        """
        Performs a random walk for each provided start_node. The random walk stops when the discovered nodes are as large as the sample size. 

        graph: a networkX graph
        sample_size: the number of nodes that should finally be contained
        start_nodes: an iterable of all the nodes from which the sampling process starts
        c: probability of flying back to the start node

        return node_sets: the resulting node sets for the start_nodes. 

        """
        node_sets = []
        pbar = tqdm(start_nodes)
        pbar.set_description("Creating subgraphs using random walk.")
        for start_node in pbar:

            # get the size of the connected component of the start node
            comp_size = len(nx.node_connected_component(graph, start_node))

            # skip if the cc is not as large as the sample size
            if comp_size < sample_size:
                continue

            size = 0 
            discovered = {start_node}
            mom_node = start_node
            while len(discovered) < sample_size:

                neigborhood = list(graph.neighbors(mom_node))
                if random.random() >c:
                    mom_node = random.choice(neigborhood)
                else :
                    mom_node = start_node

                discovered.add(mom_node)
            node_sets.append(discovered)
        pbar.close()

        return node_sets


    def random_jump(self, graph, sample_size, start_nodes, c = 0.15):
        """
        Performs a random walk for each provided start_node. The random walk stops when the discovered nodes are as large as the sample size. 

        graph: a networkX graph
        sample_size: the number of nodes that should finally be contained
        start_nodes: an iterable of all the nodes from which the sampling process starts
        c: probability of jumping to any random node in the set of nodes 

        return node_sets: the resulting node sets for the start_nodes. 

        """
        node_sets = []
        pbar = tqdm(start_nodes)
        pbar.set_description("Creating subgraphs using random jumping.")
        for start_node in pbar:
            discovered = {start_node}
            mom_node = start_node
            stuck = False
            ct = 0 
            while len(discovered) < sample_size:

                ct+= 1
                neigborhood = list(graph.neighbors(mom_node))
                if random.random() >c:
                    mom_node = random.choice(neigborhood)
                else :
                    mom_node = random.choice(list(graph))
                discovered.add(mom_node)

                if ct> sample_size *20:
                    stuck = True
                    break
            if not stuck:
                node_sets.append(discovered)
        pbar.close()
        return node_sets



    def randomGeomSubgraphs(self, G, label, starts = 1000, node_sample_size = 400):
        """
        Performs a random walk for each provided start_node. The random walk stops when the discovered nodes are as large as the sample size. 

        G: a networkX graph
        label: the number of nodes that should finally be contained
        starts: an iterable of all the nodes from which the sampling process starts
        mode:

        return subGraphs: a list of the randomly sampled subgraphs as networkX graph
        return subGraphsTorch: a list of the randomly sampled subgraphs as torch geom graph

        """
        starts = np.random.choice(np.array(G.nodes),starts)
        mode = self.mode.casefold()

        if mode == "rnn":
            R_walks = self.random_node_neighbor(G, sample_size = node_sample_size, start_nodes = starts)
        elif mode == "rw":
            R_walks = self.random_walk(G, sample_size = node_sample_size, start_nodes = starts, c= 0.0)
        elif mode == "rj":
            R_walks = self.random_jump(G, sample_size = node_sample_size, start_nodes = starts, c= 0.1)
        else:
            raise ValueError("No proper mode selected!")

        subGraphs = []
        subGraphsTorch = []
        for walk in R_walks:
            subG = nx.induced_subgraph(G, walk)

            if subG.size() <1: # sometimes the start n
                continue

            subGraphs.append(subG)
            networkXG = from_networkx(subG)
            networkXG.y = torch.tensor([label])
            subGraphsTorch.append(networkXG)

        return subGraphs, subGraphsTorch