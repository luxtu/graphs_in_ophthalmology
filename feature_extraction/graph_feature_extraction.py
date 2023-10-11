import torch

class GraphFeatureExtractor():


    def __init__(self, hetero_graph_dict, mode = torch.mean):
        self.hetero_graph_dict = hetero_graph_dict
        self.mode = mode


        self.feature_dict = self.extract_features()
        self.faz_feature_dict = self.extract_faz_features()


    def extract_faz_features(self):
        faz_feature_dict = {}
        for key, val in self.hetero_graph_dict.items():
            faz_feature_dict[key] = self.extract_faz_features_from_hetero_graph(val)
        return faz_feature_dict
    
    def extract_faz_features_from_hetero_graph(self, hetero_graph):
        faz_feature_dict = {}

        # graph_2 is the region graph
        x_values = hetero_graph.x_dict["graph_2"]
        # get the index where the area is the largest
        idx = torch.argmax(x_values[:, 2]) # area is the third feature
        # get the features for the largest area
        faz_features = x_values[idx]

        return faz_features

    def extract_features(self):
        feature_dict = {}
        for key, val in self.hetero_graph_dict.items():
            feature_dict[key] = self.extract_features_from_hetero_graph(val)
        return feature_dict
    

    def extract_features_from_hetero_graph(self, hetero_graph):
        node_type_dict = {}
        for key, val in hetero_graph.x_dict.items():
            # mean, max, min or add all features
            res = self.mode(val, dim=0)
            #extract the number of nodes
            node_num = val.shape[0]
            edge_num = hetero_graph.edge_index_dict[(key, 'to', key)].shape[1]
            avg_degree = 2*edge_num / node_num
            

            res = torch.cat((res, torch.tensor([node_num, edge_num, avg_degree])), dim=0)

            node_type_dict[key] = res
            
        return node_type_dict
    


    def get_feature_dict(self):
        return self.feature_dict
            

    def get_faz_feature_dict(self):
        return self.faz_feature_dict