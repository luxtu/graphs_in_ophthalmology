import matplotlib.pyplot as plt


class GraphPlotter2D():

    def __init__(self, torch_geom_graph, line_G = False, pred_vals = None):
        # write a init function that takes pytorch geom graphs and converts them into nodes and edges that can be used to plot the graph in 2D using matplotlib

        self.torch_geom_graph = torch_geom_graph
        #self.edge_positions = self.torch_geom_graph.x.detach().numpy()[:,-3:]
        try:
            self.graph_label = self.torch_geom_graph.y.detach().numpy()
            self.node_positions = self.torch_geom_graph.pos.detach().numpy()
        except AttributeError:
            pass
           #self.node_positions = self.torch_geom_graph.x.detach().numpy()[:,:2]

        self.node_features = self.torch_geom_graph.x.detach().numpy()
        self.edge_indices = self.torch_geom_graph.edge_index.detach().numpy()
        
        self.pred_vals = pred_vals
        self.edge_positions = self.torch_geom_graph.edge_pos.detach().numpy()

        self.line_G = line_G


    def update_graph(self, torch_geom_graph):
        # write a function that updates the graph

        self.torch_geom_graph = torch_geom_graph
        self.node_positions = self.torch_geom_graph.pos.detach().numpy()
        self.node_features = self.torch_geom_graph.x.detach().numpy()
        self.edge_indices = self.torch_geom_graph.edge_index.detach().numpy()
        self.edge_features = self.torch_geom_graph.edge_attr.detach().numpy()
        self.graph_label = self.torch_geom_graph.y


    def plot_graph_2D(self):
        # create a function that returns a plot of the graph in 2D
        # use the node positions and the edge indices to plot the graph

        fig, ax = plt.subplots()
        if not self.line_G:
            ax.scatter(self.node_positions[:,0], self.node_positions[:,1], s = 5) # cmap = "coolwarm"
            for edge in self.edge_indices.T:
                ax.plot(self.node_positions[edge,0], self.node_positions[edge,1], c="black",linewidth=1, alpha=0.5)
        else:
            if self.pred_vals is not None:
                sc = ax.scatter(self.edge_positions[:,0], self.edge_positions[:,1], s = 5, c= self.pred_vals, cmap= "coolwarm", vmin = -0.5, vmax =0.5)
                plt.colorbar(sc)
            else:
                sc = ax.scatter(self.edge_positions[:,0], self.edge_positions[:,1], s = 5)
            for edge in self.edge_indices.T:
                ax.plot(self.edge_positions[edge,0], self.edge_positions[edge,1], c="black",linewidth=1, alpha=0.5)


        plt.ylim(0,1200)
        plt.xlim(0,1200)
        plt.show()


        #fig, ax = plt.subplots()
        #ax.scatter(self.node_positions[:,0], self.node_positions[:,1],  s=5)
        #for edge in self.edge_indices.T:
        #    ax.plot(self.node_positions[edge[0]][0], self.node_positions[edge[1]][1], alpha=1)
        #plt.show()



