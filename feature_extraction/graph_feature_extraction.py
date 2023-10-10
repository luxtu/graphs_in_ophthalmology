import torch

class GraphFeatureExtractor():


    def __init__(self, hetero_graph_dict, mode = torch.mean):
        self.hetero_graph_dict = hetero_graph_dict
        self.mode = mode


        self.feature_dict = self.extract_features()

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
            node_type_dict[key] = res
        return node_type_dict

    def get_feature_dict(self):
        return self.feature_dict
            

        