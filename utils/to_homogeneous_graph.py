from torch_geometric.data import Data


def to_vessel_graph(hetero_dataset):
    """
    Converts a hetero graph dataset to a vessel graph dataset.
    """
    vessel_graph_dataset = []
    for i in range(len(hetero_dataset)):
        hetero_data = hetero_dataset[i]

        vessel_graph_data = Data(
            graph_id=hetero_data.graph_id,
            x=hetero_data["graph_1"].x,
            edge_index=hetero_data["graph_1", "to", "graph_1"].edge_index,
            y=hetero_data.y,
            pos=hetero_data["graph_1"].pos,
        )

        vessel_graph_dataset.append(vessel_graph_data)

    return vessel_graph_dataset


def to_void_graph(hetero_dataset):
    """
    Converts a hetero graph dataset to a void graph dataset.
    """
    void_graph_dataset = []
    for i in range(len(hetero_dataset)):
        hetero_data = hetero_dataset[i]

        void_graph_data = Data(
            graph_id=hetero_data.graph_id,
            x=hetero_data["graph_2"].x,
            edge_index=hetero_data["graph_2", "to", "graph_2"].edge_index,
            y=hetero_data.y,
            pos=hetero_data["graph_2"].pos,
        )

        void_graph_dataset.append(void_graph_data)

    return void_graph_dataset
