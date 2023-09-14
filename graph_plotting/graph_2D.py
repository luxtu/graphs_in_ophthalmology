import matplotlib.pyplot as plt
from line_profiler import LineProfiler

class GraphPlotter2D():

    def __init__(self, torch_geom_graph, line_G = False, pred_vals = None):
        # write a init function that takes pytorch geom graphs and converts them into nodes and edges that can be used to plot the graph in 2D using matplotlib

        self.torch_geom_graph = torch_geom_graph
        #self.edge_positions = self.torch_geom_graph.x.detach().numpy()[:,-3:]
        try:
            self.graph_label = self.torch_geom_graph.y.cpu().detach().numpy()
            self.node_positions = self.torch_geom_graph.pos.cpu().detach().numpy()
        except AttributeError:
            pass
           #self.node_positions = self.torch_geom_graph.x.detach().numpy()[:,:2]

        self.node_features = self.torch_geom_graph.x.cpu().detach().numpy()
        self.edge_indices = self.torch_geom_graph.edge_index.cpu().detach().numpy()
        
        self.pred_vals = pred_vals
        self.edge_positions = self.torch_geom_graph.edge_pos.cpu().detach().numpy()

        self.line_G = line_G


    def update_graph(self, torch_geom_graph):
        # write a function that updates the graph

        self.torch_geom_graph = torch_geom_graph
        self.node_positions = self.torch_geom_graph.pos.cpu().detach().numpy()
        self.node_features = self.torch_geom_graph.x.cpu().detach().numpy()
        self.edge_indices = self.torch_geom_graph.edge_index.cpu().detach().numpy()
        self.edge_features = self.torch_geom_graph.edge_attr.cpu().detach().numpy()
        self.graph_label = self.torch_geom_graph.y.cpu()


    def profile_2d(self):
        lp = LineProfiler()
        lp_wrapper = lp(self.plot_graph_2D)
        lp_wrapper()
        lp.print_stats()#

    def plot_graph_2D(self, ax = None):
        # create a function that returns a plot of the graph in 2D
        # use the node positions and the edge indices to plot the graph

        if ax is None:
            fig, ax = plt.subplots()
            side_l = 5.5
            fig.set_figwidth(side_l)
            fig.set_figheight(side_l)

            plt.ylim(0,1200)
            plt.xlim(0,1200)

        else:
            ax.set_ylim(0,1200)
            ax.set_xlim(0,1200)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        if not self.line_G:
            ax.scatter(self.node_positions[:,1], self.node_positions[:,0], s = 5) # cmap = "coolwarm"
            for edge in self.edge_indices.T:
                ax.plot(self.node_positions[edge,1], self.node_positions[edge,0], c="black",linewidth=1, alpha=0.5)
        else:
            if self.pred_vals is not None:
                sc = ax.scatter(self.edge_positions[:,1], self.edge_positions[:,0], s = 5, c= self.pred_vals, cmap= "coolwarm", vmin = -0.5, vmax =0.5)
                plt.colorbar(sc)
            else:
                sc = ax.scatter(self.edge_positions[:,1], self.edge_positions[:,0], s = 5)


            edges = self.edge_positions[self.edge_indices.T]
            ax.plot(edges[:, :, 1].T, edges[:, :, 0].T, c="black", linewidth=1, alpha=0.5)
                
            #for edge in self.edge_indices.T:
            #    ax.plot(self.edge_positions[edge,0], self.edge_positions[edge,1], c="black",linewidth=1, alpha=0.5)



        #plt.show()


        #fig, ax = plt.subplots()
        #ax.scatter(self.node_positions[:,0], self.node_positions[:,1],  s=5)
        #for edge in self.edge_indices.T:
        #    ax.plot(self.node_positions[edge[0]][0], self.node_positions[edge[1]][1], alpha=1)
        #plt.show()



