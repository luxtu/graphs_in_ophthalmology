import matplotlib.pyplot as plt


class GlobalFeatureImportance:
    def __init__(self, importance_dict, feature_label_dict, information_string):
        self.importance_dict = importance_dict
        self.feature_label_dict = feature_label_dict
        self.information_string = information_string
        self.node_typeto_color_char = {
            "graph_1": "#CC0000",
            "graph_2": "#0075E8",
            "faz": "#007E01",
        }

    def get_feature_importance(self):
        return self.model.feature_importances_

    def get_information_string(self):
        return self.information_string

    def get_combined_features(self):
        """
        Get the combined features from all nodes
        """
        combined_features = []
        for key in self.importance_dict.keys():
            # transfer the scores to a list
            try:
                self.importance_dict[key] = self.importance_dict[key].tolist()
            except AttributeError:
                pass
            # order the list descending, also with the feature names
            res = sorted(
                zip(self.importance_dict[key], self.feature_label_dict[key]),
                key=lambda pair: pair[0],
                reverse=True,
            )
            # add the name of the node type to the feature name
            res = [(f"{key}:{feat[1]}", feat[0]) for feat in res]
            # add the features to a combined list
            combined_features.extend(res)

        # order the combined features by importance
        combined_features = sorted(combined_features, key=lambda x: x[1], reverse=True)

        return combined_features

    def combined_feature_plot(self, num_features=10):
        """
        plot the combined features
        """

        combined_features = self.get_combined_features()
        # only keep the features with a positive score
        combined_features = [f for f in combined_features if f[1] > 0][:num_features]
        # divide the scores by the max score to get a relative importance
        max_score = max([f[1] for f in combined_features])
        combined_features = [(f[0], f[1] / max_score) for f in combined_features]

        # create a bar plot of the most important features
        color_list = []
        for f in combined_features:
            node_type = f[0].split(":")[0]
            color_list.append(self.node_typeto_color_char[node_type])

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.bar(
            [f[0] for f in combined_features],
            [f[1] for f in combined_features],
            color=color_list,
        )
        ax.set_ylabel("Relative Importance")
        ax.set_title("Feature Importance")
        plt.xticks(rotation=90)
        plt.show()
