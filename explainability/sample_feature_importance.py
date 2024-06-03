import copy

import torch

from explainability import utils


class SampleFeatureImportance:
    def __init__(self, explanation, hetero_graph, feature_label_dict):
        self.explanation = explanation
        self.hetero_graph = hetero_graph
        self.features_label_dict = feature_label_dict

        self.node_typeto_proper_name = {
            "graph_1": "Vessel",
            "graph_2": "ICP Region",
            "faz": "FAZ",
        }
        self.node_typeto_color_char = {
            "graph_1": "#CC0000",
            "graph_2": "#0075E8",
            "faz": "#007E01",
        }

        self.feature_label_dict_proper_names = {
            key: [
                f"{self.node_typeto_proper_name[key]}: {label}"
                for label in self.features_label_dict[key]
            ]
            for key in self.features_label_dict.keys()
        }

        self.feature_label_list_proper_names = [
            f"{self.node_typeto_proper_name[key]}: {label}"
            for key in self.features_label_dict.keys()
            for label in self.features_label_dict[key]
        ]

    def _check_dirs(self, path):
        import os

        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _set_up_work_explanation(self):
        work_dict = {}
        for key in self.explanation.node_mask_dict.keys():
            work_dict[key] = copy.deepcopy(self.explanation.node_mask_dict[key])
        return work_dict

    def _calculate_score(self, work_dict, het_graph_rel_pos_dict):
        score = {}
        for key in self.explanation.node_mask_dict.keys():
            score[key] = (
                work_dict[key][het_graph_rel_pos_dict[key]].sum(dim=0).cpu().numpy()
            )
        return score

    def _number_of_relevant_nodes(self, het_graph_rel_pos_dict):
        for key in self.explanation.node_mask_dict.keys():
            print(f"{key}: {het_graph_rel_pos_dict[key].sum()}")

    def _relevant_nodes_full_explanation(self, work_dict):
        # make the score a dictionary with the keys being the graph names
        # score is the sum of the absolute values of the gradients for each feature for each nodetype

        het_graph_rel_pos_dict = {}
        for key in self.explanation.node_mask_dict.keys():
            het_graph_rel_pos_dict[key] = (
                torch.ones(
                    self.explanation.node_mask_dict[key].sum(dim=-1).shape,
                    dtype=torch.bool,
                )
                .cpu()
                .detach()
                .numpy()
            )

        score = self._calculate_score(work_dict, het_graph_rel_pos_dict)

        return score, het_graph_rel_pos_dict

    def _relevant_nodes_fraction(
        self, work_dict, explained_gradient, faz_node, only_positive
    ):
        het_graph_rel_pos_dict = utils.identifiy_relevant_nodes(
            self.explanation,
            self.hetero_graph,
            explained_gradient=explained_gradient,
            faz_node=faz_node,
            only_positive=only_positive,
        )
        # print number of relevant nodes
        self._number_of_relevant_nodes(het_graph_rel_pos_dict)

        score = self._calculate_score(work_dict, het_graph_rel_pos_dict)

        return score, het_graph_rel_pos_dict

    def _relevant_nodes_top_k(
        self, work_dict, explained_gradient, faz_node, only_positive
    ):
        het_graph_rel_pos_dict = utils.top_k_important_nodes(
            self.explanation,
            self.hetero_graph,
            faz_node=faz_node,
            top_k=explained_gradient,
            only_positive=only_positive,
        )

        # print number of relevant nodes
        self._number_of_relevant_nodes(het_graph_rel_pos_dict)

        score = self._calculate_score(work_dict, het_graph_rel_pos_dict)

        return score, het_graph_rel_pos_dict

    def visualize_feature_importance(
        self,
        path,
        explained_gradient,
        only_positive,
        with_boxplot,
        num_features,
        each_type,
    ):
        """
        Wrapper for the pytorch geom function, since it does not include tight layout.
        If there is no threshold, all the nodes are considered for the importance score.
        If threshold is the string "adaptive", the threshold is calculated such that 95% of the total gradient is explained (without the gradients from nodes that contribute less than 0.05% to the total gradient)
        """

        work_dict = self._set_up_work_explanation()
        faz_node = False
        if "faz" in self.explanation.node_mask_dict.keys():
            faz_node = True

        # this means the full explanation is used
        if explained_gradient is None:
            score, het_graph_rel_pos_dict = self._relevant_nodes_full_explanation(
                work_dict
            )

        elif isinstance(explained_gradient, float):
            score, het_graph_rel_pos_dict = self._relevant_nodes_fraction(
                work_dict, explained_gradient, faz_node, only_positive
            )

        elif isinstance(explained_gradient, int):
            score, het_graph_rel_pos_dict = self._relevant_nodes_top_k(
                work_dict, explained_gradient, faz_node, only_positive
            )

        if each_type:
            self._feature_importance_plot_each_type(
                path, score, num_features=num_features, only_positive=only_positive
            )
        else:
            self._feature_importance_plot(
                path,
                score,
                num_features=num_features,
                only_positive=only_positive,
            )

        if with_boxplot:
            if each_type:
                self._feature_importance_boxplot_each_type(
                    path,
                    het_graph_rel_pos_dict,
                    score,
                    num_features=num_features,
                    only_positive=only_positive,
                )
            else:
                self._feature_importance_boxplot(
                    path,
                    het_graph_rel_pos_dict,
                    score,
                    num_features=num_features,
                    only_positive=only_positive,
                )

    def _feature_importance_plot(self, path, score, num_features, only_positive):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # get the max score for all graphs
        max_score = 0
        for node_type in self.explanation.node_mask_dict.keys():
            max_score = max(max_score, np.abs(score[node_type]).max())

        # plot the feature importance of all graphs in one plot
        # get the scores for each graph
        score_all = np.concatenate(
            [score[node_type] for node_type in self.explanation.node_mask_dict.keys()]
        )
        df = pd.DataFrame(
            {"score": score_all}, index=self.feature_label_list_proper_names
        )

        if only_positive:
            df_sorted = df.reindex(df.sort_values("score", ascending=False).index)
        else:
            df_sorted = df.reindex(df.abs().sort_values("score", ascending=False).index)
        df_sorted = df_sorted.head(num_features)

        color_list = []
        label_list = []
        # get the top features, assign them a color according to the node type
        for feature in df_sorted.index:
            for node_type in self.explanation.node_mask_dict.keys():
                if feature in self.feature_label_dict_proper_names[node_type]:
                    color_list.append(self.node_typeto_color_char[node_type])
                    label_list.append(self.node_typeto_proper_name[node_type])
                    break

        fig, ax_out = plt.subplots(figsize=(10, 7))
        ax_out.barh(df_sorted.index, df_sorted["score"], color=color_list)
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
        only_positive,
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

        if only_positive:
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

    def _feature_importance_plot_each_type(
        self, path, score, num_features=5, only_positive=False
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd

        # three subplots below each other
        fig, axs = plt.subplots(3, 1, figsize=(10, 7))
        # get the 5 most important features for each graph and plot them below each other

        # get the max score for all graphs
        max_score = 0
        for node_type in self.explanation.node_mask_dict.keys():
            max_score = max(max_score, np.abs(score[node_type]).max())

        #    max_score = np.abs(score).max()

        for i, node_type in enumerate(self.explanation.node_mask_dict.keys()):
            score_indv = score[node_type]
            df = pd.DataFrame(
                {"score": score_indv},
                index=self.feature_label_dict_proper_names[node_type],
            )
            if only_positive:
                df_sorted = df.reindex(df.sort_values("score", ascending=False).index)
            else:
                df_sorted = df.reindex(
                    df.abs().sort_values("score", ascending=False).index
                )
            df_sorted = df_sorted.head(num_features)
            # use the same xlim for all subplots
            # get rid of the legend
            ax_out = df_sorted.plot(
                kind="barh",
                title=self.node_typeto_proper_name[node_type],
                ylabel="Feature label",
                xlim=[-max_score - 0.1 * max_score, max_score + 0.1 * max_score],
                legend=False,
                ax=axs[i],
            )
            ax_out.invert_yaxis()

        # check if the path exists otherwise create it
        self._check_dirs(path)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def _feature_importance_boxplot_each_type(
        self,
        path,
        het_graph_rel_pos_dict,
        score,
        num_features,
        only_positive,
    ):
        import matplotlib.pyplot as plt
        import pandas as pd

        # change the path by splitting it and adding _boxplot, split at the last dot
        path = path.rsplit(".", 1)[0] + "_boxplot." + path.rsplit(".", 1)[1]

        # three subplots below each other
        fig, axs = plt.subplots(3, 1, figsize=(10, 7))
        # get the 5 most important features for each graph and plot them below each other

        top_feature_dict = {}
        for i, node_type in enumerate(self.hetero_graph.x_dict.keys()):
            score_indv = score[node_type]
            df = pd.DataFrame(
                {"score": score_indv},
                index=self.feature_label_dict_proper_names[node_type],
            )
            if only_positive:
                df_sorted = df.reindex(df.sort_values("score", ascending=False).index)
            else:
                df_sorted = df.reindex(
                    df.abs().sort_values("score", ascending=False).index
                )
            df_sorted = df_sorted.head(num_features)
            top_feature_dict[node_type] = df_sorted.index

        # for each of the top features, get the values in the graph of the relevant nodes
        for i, node_type in enumerate(self.hetero_graph.x_dict.keys()):
            # get the top features
            top_features = top_feature_dict[node_type]
            feature_values = {}
            for feature in top_features:
                feature_values[feature] = (
                    self.hetero_graph[node_type]
                    .x[het_graph_rel_pos_dict[node_type]][
                        :,
                        self.feature_label_dict_proper_names[node_type].index(feature),
                    ]
                    .cpu()
                    .detach()
                    .numpy()
                )
            df = pd.DataFrame(feature_values)
            ax_out = df.boxplot(ax=axs[i], vert=False)
            axs[i].set_title(self.node_typeto_proper_name[node_type])

            ax_out.invert_yaxis()
        # check if the path exists otherwise create it
        self._check_dirs(path)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def visualize_node_importance_histogram(self, path, abs=True):
        import matplotlib.pyplot as plt

        # creates a copy of the explanation
        work_dict = self._set_up_work_explanation()
        if abs:
            for key in self.explanation.node_mask_dict.keys():
                work_dict[key] = self.explanation.node_mask_dict[key].abs().sum(dim=-1)

        else:
            for key in self.explanation.node_mask_dict.keys():
                work_dict[key] = self.explanation.node_mask_dict[key].sum(dim=-1)

        # find the max value in the work_dict
        max_val = 0
        for key in work_dict.keys():
            max_val = max(max_val, work_dict[key].max().cpu().detach().numpy())

        # find the min value in the work_dict
        min_val = 0
        for key in work_dict.keys():
            min_val = min(min_val, work_dict[key].min().cpu().detach().numpy())

        fig, ax = plt.subplots()
        # plot the histogram for all the graphs

        for key in work_dict.keys():
            ax.hist(
                work_dict[key].cpu().detach().numpy(),
                bins=100,
                alpha=0.5,
                range=(min_val, max_val),
                label=self.node_typeto_proper_name[key],
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


    def attribution_heatmap(self, path):
        import matplotlib.pyplot as plt
        from graph_plotting import graph_2D

        faz_node = False
        if "faz" in self.explanation.node_mask_dict.keys():
            faz_node = True

        fig, ax = plt.subplots()
        # add titles for each subplot in the figu
        # add legend to the figure that assigns the square marker ("s") to the intercapillary region and the circle marker to the vessel region
        marker_list = ["s", "o", "D"]
        legend_items = []
        for key, marker in zip(self.explanation.node_mask_dict.keys(), marker_list):
            legend_items.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker=marker,
                    color="red",
                    alpha=0.8,
                    label=self.node_typeto_proper_name[key],
                    markerfacecolor=self.node_typeto_color_char[key],
                    markersize=6,
                    linestyle="None",
                )
            )

        plotter2d = graph_2D.HeteroGraphPlotter2D()
        # plotter2d.set_val_range(1, 0)
        work_dict = self._set_up_work_explanation()
        for key in self.explanation.x_dict.keys():
            work_dict[key] = self.explanation.x_dict[key].abs().sum(dim=-1)
        if faz_node:
            plotter2d.plot_graph_2D_faz(
                self.hetero_graph, edges=False, pred_val_dict=work_dict, ax=ax
            )
        else:
            plotter2d.plot_graph_2D(
                self.hetero_graph, edges=False, pred_val_dict=work_dict, ax=ax
            )
        # add the legend to the figure and avoid overlapping
        fig.legend(handles=legend_items, loc='upper left')
        # check if the path exists otherwise create it
        self._check_dirs(path)

        #plt.tight_layout()
        plt.savefig(path, bbox_inches="tight")
        plt.close()
