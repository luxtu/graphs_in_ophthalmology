import matplotlib.pyplot as plt
from line_profiler import LineProfiler
from torch.nn import functional

class GraphPlotter2D():

    def __init__(self, torch_geom_graph, line_G = False, pred_vals = None):
        # write a init function that takes pytorch geom graphs and converts them into nodes and edges that can be used to plot the graph in 2D using matplotlib

        self.torch_geom_graph = torch_geom_graph
        #self.edge_positions = self.torch_geom_graph.x.detach().numpy()[:,-3:]
        try:
            #self.graph_label = self.torch_geom_graph.y.cpu().detach().numpy()
            self.node_positions = self.torch_geom_graph.pos.cpu().detach().numpy()
        except AttributeError:
            pass
           #self.node_positions = self.torch_geom_graph.x.detach().numpy()[:,:2]

        #self.node_features = self.torch_geom_graph.x.cpu().detach().numpy()
        self.edge_indices = self.torch_geom_graph.edge_index.cpu().detach().numpy()
        
        self.pred_vals = pred_vals

        try:
            self.edge_positions = self.torch_geom_graph.edge_pos.cpu().detach().numpy()
        except AttributeError:
            pass

        self.line_G = line_G


    def profile_2d(self):
        lp = LineProfiler()
        lp_wrapper = lp(self.plot_graph_2D)
        lp_wrapper()
        lp.print_stats()#


    def plot_graph_2D(self, edges= True, ax = None):
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
            if self.pred_vals is not None:
                sc = ax.scatter(self.node_positions[:,1], self.node_positions[:,0], s = 5,c = self.pred_vals, cmap = "coolwarm")
                plt.colorbar(sc)
            else:
                ax.scatter(self.node_positions[:,1], self.node_positions[:,0], s = 8, c = "orange") # cmap = "coolwarm"

            if edges:
                for edge in self.edge_indices.T:
                    ax.plot(self.node_positions[edge,1], self.node_positions[edge,0], c="black",linewidth=1, alpha=0.5)
        else:
            if self.pred_vals is not None:
                sc = ax.scatter(self.edge_positions[:,1], self.edge_positions[:,0], s = 5, c= self.pred_vals, cmap= "coolwarm", vmin = -0.5, vmax =0.5)
                plt.colorbar(sc)
            else:
                sc = ax.scatter(self.edge_positions[:,1], self.edge_positions[:,0], s = 8, c = "orange")

            if edges:
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



class HeteroGraphPlotter2D():

    graph_1_name = "graph_1"
    graph_2_name = "graph_2"

    def set_cls(self, cls):
        self.cls = cls

    def plot_graph_2D(self, het_graph , edges= True, ax = None, pred_val_dict = None):
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

        het_graph1_pos = het_graph[self.graph_1_name].pos.cpu().detach().numpy()
        het_graph2_pos = het_graph[self.graph_2_name].pos.cpu().detach().numpy()

        if edges:
            for edge in het_graph[self.graph_1_name, 'to', self.graph_1_name].edge_index.cpu().detach().numpy().T:
                ax.plot(het_graph1_pos[edge,1], het_graph1_pos[edge,0], c="blue",linewidth=1, alpha=0.5, zorder = 1)
            for edge in het_graph[self.graph_2_name, 'to', self.graph_2_name].edge_index.cpu().detach().numpy().T:
                ax.plot(het_graph2_pos[edge,1], het_graph2_pos[edge,0], c="red",linewidth=1, alpha=0.5, zorder = 1)
            for edge in het_graph[self.graph_1_name, 'to', self.graph_2_name].edge_index.cpu().detach().numpy().T:
                #create a line for each edge 
                x_pos = (het_graph1_pos[edge[0],1], het_graph2_pos[edge[1],1])
                y_pos = (het_graph1_pos[edge[0],0], het_graph2_pos[edge[1],0])
#
                ax.plot(x_pos, y_pos, linewidth=1, alpha=0.5, c= "black", zorder = 1)
                #break
#

        if pred_val_dict is not None:

            #actually not softmaxed
            sft_max1 = pred_val_dict[self.graph_1_name].cpu().detach()#, dim=1)#functional.softmax(
            sft_max2 = pred_val_dict[self.graph_2_name].cpu().detach()#, dim=1)#functional.softmax(


            max_v = max(pred_val_dict[self.graph_1_name].cpu().detach()[:,self.cls].max(), pred_val_dict[self.graph_2_name].cpu().detach()[:,self.cls].max())#1
            min_v = min(pred_val_dict[self.graph_1_name].cpu().detach()[:,self.cls].min(), pred_val_dict[self.graph_2_name].cpu().detach()[:,self.cls].min())#0

            upper_range = max(max_v, abs(min_v))

            sc1 = ax.scatter(het_graph1_pos[:,1], het_graph1_pos[:,0], s = 8, c = sft_max1[:,self.cls], vmin = -upper_range, vmax = upper_range, cmap = "coolwarm", zorder = 2)
            sc2 = ax.scatter(het_graph2_pos[:,1], het_graph2_pos[:,0], s = 12, c = sft_max2[:,self.cls], vmin = -upper_range, vmax = upper_range, cmap = "coolwarm", zorder = 2, marker = "s")

       
            plt.colorbar(sc2)
        else:
            ax.scatter(het_graph1_pos[:,1], het_graph1_pos[:,0], s = 8)
            ax.scatter(het_graph2_pos[:,1], het_graph2_pos[:,0], s = 12, marker = "s")       