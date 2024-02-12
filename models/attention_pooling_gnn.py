import torch

class Attention_Pooling_GNN(torch.nn.Module):
    def __init__(self, hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout, 
                 aggregation_mode ,
                 node_types, 
                 meta_data = None, 
                 num_pre_processing_layers = 3, 
                 num_post_processing_layers = 3, 
                 num_final_lin_layers = 3,
                 batch_norm = True, 
                 conv_aggr = "cat", 
                 hetero_conns = True, 
                 faz_node = False,
                 global_node = False,
                 homogeneous_conv = GCNConv,
                 heterogeneous_conv = GATConv,
                 activation = F.relu,
                 start_rep = False,
                 aggr_faz = False,
                 faz_conns = True,
                 skip_connection = True,
                 ):
        super().__init__()
        torch.manual_seed(1234567)


        ######################################## testing attention addon
        # Embedding layer
        self.embedding = Linear(-1, hidden_channels)

        # Attention layer
        self.attention = Linear(hidden_channels, 1)

        # Fully connected layer for classification
        self.fc = Linear(hidden_channels, out_channels)




    def forward(self, x_dict, batch_dict, pos_dict):
        """
        Takes the refined embeedings and pools them into a 6 x 6 grid, the pooling is done with the self.attention_mode function.
        For the spatial pooling the pos_dict information is used, also only nodes from the graph_1 and graph_2 are pooled. The batch dict is used to identify the nodes from the two graphs.
        The pooled embeddings can then attend to each other and the resulting embeddings are used for classification.
        """

        def find_patch_label(x, y, patch_size, num_patches):
            patch_x = (x // patch_size).clamp(0, num_patches-1)
            patch_y = (y // patch_size).clamp(0, num_patches-1)

            # make the patch labels into ints
            patch_x = patch_x.int()
            patch_y = patch_y.int()
            # cannot be larger than 5, and not smaller than 0
            #patch_x = torch.clamp(patch_x, 0, 5)
            #patch_y = torch.clamp(patch_y, 0, 5)
            return patch_x, patch_y

        # Define the size of the larger grid
        grid_size = 1216

        # Define the size of the patch
        patch_size = 202

        # Calculate the number of patches in each dimension
        num_patches = grid_size // patch_size

        labels = torch.zeros((num_patches, num_patches), dtype=torch.long, device=self.device)
        label_counter = 0
        for i in range(num_patches):
            for j in range(num_patches):
                labels[i, j] = label_counter
                label_counter += 1


        pooled_node_type_representations = []
        node_types = ["graph_1", "graph_2"]

        # get the batch size from the unique elements in the batch
        batch_size = torch.unique(batch_dict["graph_1"]).shape[0]
        #print("Batch size: ", batch_size)

        for key in node_types:
            
            node_positions = pos_dict[key].clone().requires_grad_(False)
            node_patch_labels = torch.zeros(node_positions.shape[0], dtype=torch.long, device=self.device)
            # adjust position to -600 to 600 for both x and y axis
            #node_positions -= 600

            x = node_positions[:, 0]
            y = node_positions[:, 1]
            # print the max and min of the x and y coordinates
            #print("Max x: ", torch.max(x))
            #print("Min x: ", torch.min(x))
            #print("Max y: ", torch.max(y))
            #print("Min y: ", torch.min(y))
            patch_x, patch_y = find_patch_label(x, y, patch_size, num_patches)
            node_patch_labels = labels[patch_x, patch_y]

            #for i in range(node_positions.shape[0]):
            #    x = node_positions[i, 0]
            #    y = node_positions[i, 1]
            #    patch_x, patch_y = find_patch_label(x, y, patch_size)
            #    # turn the patche indice into ints
            #    patch_x = patch_x.int()
            #    patch_y = patch_y.int()
            #    node_patch_labels[i] = labels[patch_x, patch_y]
                
            pooled_representations = []
            for patch in range(num_patches**2):
                # print the type of dtype of the index vector
                if isinstance(self.aggregation_mode, list):
                    aggr_list_rep = []
                    for j in range(len(self.aggregation_mode)):
                        # check if the aggregation mode is global_add_pool
                        if self.aggregation_mode[j] == global_add_pool:
                            #pool the nodes from the same patch together
                            aggr_list_rep.append(self.aggregation_mode[j](x_dict[key][node_patch_labels == patch], batch_dict[key][node_patch_labels == patch])) # /1000
                        else:
                            aggr_list_rep.append(self.aggregation_mode[j](x_dict[key][node_patch_labels == patch], batch_dict[key][node_patch_labels == patch]))
                        # concat the pooled representations from the different aggregation modes, axis is the feature dimension
                    pooled_representations.append(torch.cat(aggr_list_rep, dim=1))
                else:
                    pooled_representations.append(self.aggregation_mode(x_dict[key][node_patch_labels == patch], batch_dict[key][node_patch_labels == patch]))
                
                if pooled_representations[-1].shape[0] != batch_size:
                    # identify the sample in the batch that has no node in the patch
                    #print(batch_dict[key][node_patch_labels == patch])
                    #print(batch_dict[key][node_patch_labels == patch].shape)
                    # use unique to find the unique elements in the batch
                    unique_batch = torch.unique(batch_dict[key][node_patch_labels == patch])
                    #print(unique_batch)
                    new_batch = torch.zeros((batch_size, pooled_representations[-1][0].shape[0]), dtype=torch.long, device=self.device)
                    # check which element is missing
                    old_batch_idx = 0
                    for i in range(batch_size):
                        if i not in unique_batch:
                            print("Missing element in batch: ", i)
                            # add a zero tensor at the correct position to the new batch
                            new_batch[i] = torch.zeros_like(pooled_representations[-1][0])
                        else:
                            new_batch[i] = pooled_representations[-1][old_batch_idx]
                            old_batch_idx += 1
                    pooled_representations[-1] = new_batch







                            #if pooled_representations[-1].shape[0] == batch_size - 1:
                            #    # add a zero tensor at the correct position to pooled_representations[-1]
                            #    pooled_representations[-1] = torch.cat([pooled_representations[-1][:i], torch.zeros_like(pooled_representations[-1][0]).unsqueeze(0)], dim=0)
                            #else:
                            #    pooled_representations[-1] = torch.cat([pooled_representations[-1][:i], torch.zeros_like(pooled_representations[-1][0]).unsqueeze(0), pooled_representations[-1][i:]], dim=0)
                            #    print(pooled_representations[-1].shape)
                            # add a zero tensor at the correct position to pooled_representations[-1]
                            #pooled_representations[-1][i] = torch.zeros_like(pooled_representations[-1][i])



                    #raise ValueError("Sth went wrong")


                    #raise ValueError("Sth went wrong")
            # for a pooled representation, the first dimension is the number of patches, the second dimension is the number of patches, the third dimension is the number of features
            # check that for each patch there is an embedding for each sample in the batch
            #print(pooled_representations[0].shape)



            # first dim is the number of patches (must be 36), second dim is the batch size, third dim is the number of features

            pooled_node_type_representations.append(torch.stack(pooled_representations, dim=0))



        
        # concatenate the pooled representations from the two node types
        pooled_node_type_representations = torch.cat(pooled_node_type_representations, dim=2)
        #print(pooled_node_type_representations.shape)

        # flip the second dimension, so that the batch size is the first dimension
        pooled_node_type_representations = pooled_node_type_representations.transpose(0, 1)
        embedded = self.embedding(pooled_node_type_representations)
        #print(embedded.shape)

        # Apply attention mechanism
        attention_weights = F.softmax(self.attention(embedded), dim=1)
        attended = torch.sum(attention_weights * embedded, dim=1)

        # Classification using fully connected layer
        output = self.fc(attended)

        return output
