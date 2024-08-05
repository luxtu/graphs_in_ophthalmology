import copy

import torch

from explainability import utils_homogeneous as utils


class SampleFeatureImportance:
    def __init__(self, explanation, graph, feature_label_dict, graph_type):
        self.explanation = explanation
        self.graph = graph
        self.features_label_dict = feature_label_dict
        self.graph_type = graph_type

        # assert that the graph type is either "vessel" or "region"
        assert (
            self.graph_type == "vessel" or self.graph_type == "region"
        ), "The graph type should be either 'vessel' or 'region'"


        self.node_typeto_proper_name = {
            "vessel": "Vessel",
            "region": "ICP Region"
        }
        self.node_typeto_color_char = {
            "vessel": "#CC0000",
            "region": "#0075E8"
        }

        self.graph_type_to_node_type = {
            "vessel": "graph_1",
            "region": "graph_2"
        }

        #self.feature_label_dict_proper_names = {
        #    key: [
        #        f"{self.node_typeto_proper_name[key]}: {label}"
        #        for label in self.features_label_dict[key]
        #    ]
        #    for key in self.features_label_dict.keys()
        #}
#
        self.feature_label_list_proper_names = [
            f"{self.node_typeto_proper_name[self.graph_type]}: {label}"
            for label in self.features_label_dict[self.graph_type_to_node_type[self.graph_type]]
        ]

    def _check_dirs(self, path):
        import os

        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _set_up_work_explanation(self):
        work_mask = copy.deepcopy(self.explanation.node_mask)
        return work_mask

    def _calculate_score(self, work_mask, graph_rel_pos_idcs):
        score = work_mask[graph_rel_pos_idcs].sum(dim=1).cpu().numpy()
        
        return score
    
    def _calculate_feature_score(self, work_mask, graph_rel_pos_idcs):
        score = work_mask[graph_rel_pos_idcs].sum(dim=0).cpu().numpy()
        
        return score

    def _number_of_relevant_nodes(self, graph_rel_pos_idcs):
        print(f"Number of relevant nodes: {graph_rel_pos_idcs.sum()}" )
        #for key in self.explanation.node_mask_dict.keys():
        #    print(f"{key}: {het_graph_rel_pos_dict[key].sum()}")

    def _relevant_nodes_full_explanation(self, work_mask, feature_score = False):
        # make the score a dictionary with the keys being the graph names
        # score is the sum of the absolute values of the gradients for each feature for each nodetype

        graph_rel_pos_idcs = torch.ones(
            self.explanation.node_mask.sum(dim=-1).shape, dtype=torch.bool
        ).cpu().detach().numpy()

        if feature_score:
            score = self._calculate_feature_score(work_mask, graph_rel_pos_idcs)
        else:
            score = self._calculate_score(work_mask, graph_rel_pos_idcs)

        return score, graph_rel_pos_idcs

    def _relevant_nodes_fraction(
        self, work_mask, explained_gradient, abs, feature_score = False
    ):
        graph_rel_pos_idcs = utils.identifiy_relevant_nodes(
            self.explanation,
            self.graph,
            explained_gradient=explained_gradient,
            abs=abs,
        )
        # print number of relevant nodes
        self._number_of_relevant_nodes(graph_rel_pos_idcs)

        if feature_score:
            score = self._calculate_feature_score(work_mask, graph_rel_pos_idcs)
        else:
            score = self._calculate_score(work_mask, graph_rel_pos_idcs)
        

        return score, graph_rel_pos_idcs

    def _relevant_nodes_top_k(
        self, work_mask, explained_gradient, abs, feature_score = False
    ):
        graph_rel_pos_idcs = utils.top_k_important_nodes(
            self.explanation,
            self.graph,
            top_k=explained_gradient,
            abs=abs,
        )

        # print number of relevant nodes
        self._number_of_relevant_nodes(graph_rel_pos_idcs)

        if feature_score:
            score = self._calculate_feature_score(work_mask, graph_rel_pos_idcs)
        else:
            score = self._calculate_score(work_mask, graph_rel_pos_idcs)

        return score, graph_rel_pos_idcs

    def visualize_feature_importance(
        self,
        path,
        explained_gradient = None,
        abs = False,
        with_boxplot = False,
        num_features = 5,
    ):
        """
        Wrapper for the pytorch geom function, since it does not include tight layout.
        If there is no threshold, all the nodes are considered for the importance score.
        If threshold is the string "adaptive", the threshold is calculated such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
        """

        work_mask = self._set_up_work_explanation()


        # this means the full explanation is used
        if explained_gradient is None:
            score, het_graph_rel_pos_dict = self._relevant_nodes_full_explanation(
                work_mask, feature_score = True
            )

        elif isinstance(explained_gradient, float):
            score, het_graph_rel_pos_dict = self._relevant_nodes_fraction(
                work_mask, explained_gradient, abs, feature_score = True
            )

        elif isinstance(explained_gradient, int):
            score, het_graph_rel_pos_dict = self._relevant_nodes_top_k(
                work_mask, explained_gradient, abs, feature_score = True
            )

        self._feature_importance_plot(
            path,
            score,
            num_features=num_features,
            abs=abs,
        )

        if with_boxplot:
            self._feature_importance_boxplot(
                path,
                het_graph_rel_pos_dict,
                score,
                num_features=num_features,
                abs=abs,
            )



    def _feature_importance_plot(self, path, score, num_features, abs):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # find the max value in the work_mask
        work_mask = self._set_up_work_explanation()
        work_mask = work_mask.abs()
        #max_val = work_mask.max().cpu().detach().numpy()

        # plot the feature importance of all graphs in one plot
        # get the scores for each graph

        df = pd.DataFrame(
            {"score": score}, index= self.feature_label_list_proper_names
        )
        if abs:
            df_sorted = df.reindex(df.sort_values("score", ascending=False).index)
        else:
            df_sorted = df.reindex(df.abs().sort_values("score", ascending=False).index)
        df_sorted = df_sorted.head(num_features)

        fig, ax_out = plt.subplots(figsize=(10, 7))
        ax_out.barh(df_sorted.index, df_sorted["score"], color="red")
        ax_out.invert_yaxis()

        # check if the path exists otherwise create it
        self._check_dirs(path)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()



    def _feature_importance_boxplot(
        self,
        path,
        het_graph_rel_pos_dict,
        score,
        num_features,
        abs,
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # change the path by splitting it and adding _boxplot, split at the last dot
        path = path.rsplit(".", 1)[0] + "_boxplot." + path.rsplit(".", 1)[1]

        # get the most important features across all graphs
        max_score = 0
        for node_type in self.hetero_graph.x_dict.keys():
            max_score = max(max_score, np.abs(score[node_type]).max())

        # plot the feature importance of all graphs in one plot
        # get the scores for each graph
        score_all = np.concatenate(
            [score[node_type] for node_type in self.hetero_graph.x_dict.keys()]
        )
        df = pd.DataFrame(
            {"score": score_all}, index=self.feature_label_list_proper_names
        )

        if abs:
            df_sorted = df.reindex(df.sort_values("score", ascending=False).index)
        else:
            df_sorted = df.reindex(df.abs().sort_values("score", ascending=False).index)

        # for each of the top features, get the values in the graph of the relevant nodes
        top_features = df_sorted.head(num_features).index
        feature_values = []
        for feature in top_features:
            for node_type in self.hetero_graph.x_dict.keys():
                try:
                    feature_values.append(
                        self.hetero_graph[node_type]
                        .x[het_graph_rel_pos_dict[node_type]][
                            :,
                            self.feature_label_dict_proper_names[node_type].index(
                                feature
                            ),
                        ]
                        .cpu()
                        .detach()
                        .numpy()
                    )
                except ValueError:
                    continue

        fig, ax = plt.subplots()
        ax.boxplot(feature_values, vert=False)
        # invert the y axis
        ax.invert_yaxis()
        ax.set_title("Feature importance")
        ax.set_yticklabels(top_features)

        # check if the path exists otherwise create it
        self._check_dirs(path)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()



    def visualize_node_importance_histogram(self, path, abs=True):
        import matplotlib.pyplot as plt

        # creates a copy of the explanation
        work_mask = self._set_up_work_explanation()
        if abs:
            work_mask = work_mask.abs().sum(dim=-1)
        else:
            work_mask = work_mask.sum(dim=-1)

        # find the max value in the work_mask
        max_val = work_mask.max().cpu().detach().numpy()

        # find the min value in the work_dict
        min_val = work_mask.min().cpu().detach().numpy()

        fig, ax = plt.subplots()
        # plot the histogram for all the graphs

        ax.hist(
            work_mask.cpu().detach().numpy(),
            bins=100,
            alpha=0.5,
            range=(min_val, max_val),
            label=self.node_typeto_proper_name[self.graph_type],
        )

        # create a legend for the histogram
        ax.legend(loc="upper right")

        plt.xlabel("Node importance (Abs sum of gradients)")
        plt.ylabel("Number of nodes")

        # plt.xscale('log')
        plt.xlim(min_val, max_val)
        plt.yscale("log")

        # check if the path exists otherwise create it
        self._check_dirs(path)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
