import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3
num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

from toxfm.models.attention import CoAttentionLayer, TransformerCrossFusion


def _ensure_correct_edge_index(edge_index, num_nodes):
    """Ensure edge_index is properly formatted for PyTorch Geometric"""
    if not isinstance(edge_index, torch.Tensor):
        edge_index = torch.tensor(edge_index, dtype=torch.long)

    edge_index = edge_index.long().contiguous()

    # Move to CPU for safety checks
    device = edge_index.device
    edge_index_cpu = edge_index.cpu()

    # Filter out invalid indices
    mask = (edge_index_cpu[0] < num_nodes) & (edge_index_cpu[1] < num_nodes) & \
           (edge_index_cpu[0] >= 0) & (edge_index_cpu[1] >= 0)
    edge_index_cpu = edge_index_cpu[:, mask]

    # If no edges remain after filtering, add a self-loop to avoid a crash
    if edge_index_cpu.size(1) == 0:
        edge_index_cpu = torch.tensor([[0], [0]], dtype=torch.long, device=edge_index_cpu.device)

    # Return to the original device
    return edge_index_cpu.to(device)


class EnhancedGINConv(MessagePassing):
    """
    Enhanced GIN aggregation: a graph convolution layer that integrates edge information via concatenation.

    Args:
        emb_dim (int): dimensionality of node and edge embeddings
        aggr (str): aggregation method, defaults to "add"
    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, aggr="add"):
        super(EnhancedGINConv, self).__init__()
        # A more complex multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, 4 * emb_dim),
            torch.nn.BatchNorm1d(4 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * emb_dim, 2 * emb_dim),
            torch.nn.BatchNorm1d(2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # Ensure correct format for edge_index
        edge_index = _ensure_correct_edge_index(edge_index, x.size(0))

        # Ensure edge features are of LongTensor type
        if edge_attr is not None:
            edge_attr = edge_attr.long()

        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        if edge_attr is not None:
            # Ensure the self-loop edge features match the dimension of the original edge features
            self_loop_attr = torch.zeros(x.size(0), edge_attr.size(1), dtype=edge_attr.dtype, device=edge_attr.device)
            self_loop_attr[:, 0] = 4  # bond type for self-loop edge
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        else:
            # If there are no edge features, create zero features for all edges (including self-loops)
            num_total_edges = edge_index.size(1)
            edge_attr = torch.zeros((num_total_edges, 2), dtype=x.dtype, device=x.device)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)


    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim=1)

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):

    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # Ensure correct format for edge_index
        edge_index = _ensure_correct_edge_index(edge_index, x.size(0))

        # Ensure edge features are of LongTensor type
        edge_attr = edge_attr.long()

        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings, norm=norm)


    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=4, negative_slope=0.2, aggr="add"):  # increase the number of attention heads
        super(GATConv, self).__init__()

        self.aggr = aggr

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope

        self.weight_linear = torch.nn.Linear(emb_dim, heads * emb_dim)
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = torch.nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # Ensure correct format for edge_index
        edge_index = _ensure_correct_edge_index(edge_index, x.size(0))

        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias

        return aggr_out


class GATConvContinuous(torch.nn.Module):
    """
    GAT convolution layer supporting continuous node features (does not use MessagePassing, to avoid JIT caching issues).
    Suitable for scenarios such as protein-protein interaction networks.
    """
    def __init__(self, emb_dim, heads=4, negative_slope=0.2, drop_ratio=0.1, input_dim=None):
        super(GATConvContinuous, self).__init__()

        self.emb_dim = emb_dim
        self.heads = heads
        self.negative_slope = negative_slope
        self.drop_ratio = drop_ratio
        self.input_dim = input_dim if input_dim is not None else emb_dim

        # Linear transformation of node features
        self.weight_linear = torch.nn.Linear(self.input_dim, heads * emb_dim)
        # Attention parameters
        self.att = torch.nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))
        self.bias = torch.nn.Parameter(torch.Tensor(heads * emb_dim))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [num_nodes, emb_dim] node features
            edge_index: [2, num_edges] edge indices
            edge_attr: unused (kept for interface compatibility)
        Returns:
            [num_nodes, heads * emb_dim] updated node features
        """
        edge_index = _ensure_correct_edge_index(edge_index, x.size(0))

        # Add self-loops
        num_nodes = x.size(0)
        self_loop_index = torch.arange(num_nodes, device=edge_index.device)
        self_loop_index = torch.stack([self_loop_index, self_loop_index], dim=0)
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)

        # Linear transformation: [num_nodes, emb_dim] -> [num_nodes, heads*emb_dim]
        x = self.weight_linear(x)

        # Reshape to multi-head: [num_nodes, heads, emb_dim]
        x = x.view(num_nodes, self.heads, self.emb_dim)

        # Get source-node and target-node features
        row, col = edge_index[0], edge_index[1]  # [num_edges]
        x_j = x[row]  # [num_edges, heads, emb_dim]
        x_i = x[col]  # [num_edges, heads, emb_dim]

        # Compute attention scores
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # [num_edges, heads]
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, col)  # normalize by target nodes
        alpha = F.dropout(alpha, p=self.drop_ratio, training=self.training)

        # Weighted aggregation
        out = x_j * alpha.unsqueeze(-1)  # [num_edges, heads, emb_dim]

        # Use scatter to aggregate to target nodes
        # col: [num_edges] -> [num_edges, 1, 1] to broadcast to [num_edges, heads, emb_dim]
        col_expanded = col.view(-1, 1, 1).expand_as(out)
        out = scatter(out, col_expanded, dim=0, dim_size=num_nodes, reduce='add')
        # out: [num_nodes, heads, emb_dim]

        # Reshape back to 2D and add bias
        out = out.view(num_nodes, -1)  # [num_nodes, heads*emb_dim]
        out = out + self.bias

        return out


class GATContinuous(torch.nn.Module):
    """
    Multi-layer graph neural network based on GATConvContinuous.
    Supports continuous node features and continuous edge features.
    """
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, heads=4):
        super(GATContinuous, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            # The first layer's input is emb_dim; subsequent layers' input is heads * emb_dim
            layer_input_dim = emb_dim if layer == 0 else heads * emb_dim
            self.gnns.append(GATConvContinuous(emb_dim, heads=heads, drop_ratio=drop_ratio, input_dim=layer_input_dim))

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(heads * emb_dim))

        # Residual-connection projection layer: maps emb_dim to heads*emb_dim
        self.residual_proj = nn.Linear(emb_dim, heads * emb_dim)

    def forward(self, x, edge_index, edge_attr=None):
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            # Residual connection: add original node features to graph-convolved features (requires dimension match)
            if layer == 0:
                # First layer: input is emb_dim and output is heads*emb_dim, cannot be added directly
                # Use a projection layer to map the input to the same dimension before adding
                if hasattr(self, 'residual_proj'):
                    h = h + self.residual_proj(h_list[layer])
            else:
                # Later layers: input and output have the same dimension, direct residual connection
                h = h + h_list[layer]

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            node_representation = h_list[-1]

        return node_representation


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.emb_dim = emb_dim
        # increase complexity
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        # Ensure correct format for edge_index
        edge_index = _ensure_correct_edge_index(edge_index, x.size(0))

        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim=1)

    def update(self, aggr_out):
        return self.linear(aggr_out)


class GNN(torch.nn.Module):
    """
    GNN model that extends GIN to integrate edge information via concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(EnhancedGINConv(emb_dim, aggr="add"))  # use the enhanced GIN
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

        # Add a co-attention layer
        self.co_attention = CoAttentionLayer(emb_dim)

    # def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # Ensure x is a LongTensor so it can be used as an embedding index
        x = x.long().to(edge_index.device)

        # Process node features according to the shape of x
        if x.shape[1] >= 2:
            # Standard molecular data with shape [num_nodes, 2] or larger
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        elif x.shape[1] == 1:
            # Protein data with shape [num_nodes, 1]
            # For single-feature data, use only the first feature
            x = self.x_embedding1(x[:, 0])
        else:
            raise ValueError(f"Unexpected node feature shape: {x.shape}")

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation



    def forward_with_coattention(self, *argv, gene_features=None):
        """
        Forward pass with a co-attention mechanism.

        Args:
            *argv: standard GNN input arguments
            gene_features: gene expression features [batch_size, num_genes, emb_dim]

        Returns:
            Enhanced node representations and gene features.
        """
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        # Ensure x is a LongTensor so it can be used as an embedding index
        x = x.long().to(edge_index.device)

        # Process node features according to the shape of x
        if x.shape[1] >= 2:
            # Standard molecular data with shape [num_nodes, 2] or larger
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        elif x.shape[1] == 1:
            # Protein data with shape [num_nodes, 1]
            # For single-feature data, use only the first feature
            x = self.x_embedding1(x[:, 0])
        else:
            raise ValueError(f"Unexpected node feature shape: {x.shape}")

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        # If gene features are provided, apply the co-attention mechanism
        if gene_features is not None:
            # Reshape the node representation into a shape suitable for co-attention
            # Here we assume node_representation is [batch_size*num_nodes, emb_dim]
            # It needs to be converted to [batch_size, num_nodes, emb_dim]
            # Note: this is simplified here; in practice it may need adjustment per case
            enhanced_molecule, enhanced_gene = self.co_attention(
                node_representation.unsqueeze(0),
                gene_features.unsqueeze(0)
            )
            return enhanced_molecule.squeeze(0), enhanced_gene.squeeze(0)

        return node_representation


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, num_tasks, JK="last", drop_ratio=0, graph_pooling="mean", gnn_type="gin"):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (self.num_layer + 1) * emb_dim, self.num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)

        return self.graph_pred_linear(self.pool(node_representation, batch))


class ProteinGINConv(MessagePassing):
    """
    GIN convolution layer designed for protein graphs: no self-loops are added and connected graphs are processed directly.

    Args:
        emb_dim (int): dimensionality of node embeddings
        aggr (str): aggregation method, defaults to "add"
    """

    def __init__(self, emb_dim, aggr="sum"):
        super(ProteinGINConv, self).__init__()
        # More complex multi-layer perceptron with increased non-linearity
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4 * emb_dim),
            torch.nn.BatchNorm1d(4 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),  # add dropout to prevent overfitting
            torch.nn.Linear(4 * emb_dim, emb_dim)
        )
        # Edge embedding for protein graphs; edge features are continuous values
        self.edge_embedding = torch.nn.Linear(1, emb_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # Ensure the edge index is properly formatted
        edge_index = _ensure_correct_edge_index(edge_index, x.size(0))

        # If edge features are provided, convert them to embeddings
        if edge_attr is not None:
            edge_attr = edge_attr.float()
            edge_embeddings = self.edge_embedding(edge_attr)
        else:
            # If there are no edge features, use zero vectors
            edge_embeddings = torch.zeros((edge_index.size(1), x.size(1)), device=x.device, dtype=x.dtype)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        # Add node features and edge features
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class ProteinGNN(torch.nn.Module):
    """
    GNN model designed for protein graphs: no self-loops are added and connected graphs are processed directly.

    Args:
        num_layer (int): number of GNN layers
        emb_dim (int): embedding dimension
        JK (str): jump-connection type ("last", "concat", "max", "sum")
        drop_ratio (float): dropout rate
    """

    def __init__(self, num_layer, emb_dim, JK="sum", drop_ratio=0.1):  # increase the default dropout rate
        super(ProteinGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 1:
            raise ValueError("Number of GNN layers must be at least 1.")

        ### List of GNN layers
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(ProteinGINConv(emb_dim, aggr="add"))

        ### List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.

        Args:
            x: node features [num_nodes, emb_dim]
            edge_index: edge indices [2, num_edges]
            edge_attr: edge features [num_edges, num_edge_features], optional

        Returns:
            node representations [num_nodes, output_dim]
        """
        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # Apply dropout and ReLU during training
            if layer == self.num_layer - 1:
                # The last layer also applies dropout but not ReLU
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        ### Different types of jump connections
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]

        return node_representation


class MolProteinCoAttentionGNN(torch.nn.Module):
    """
    Co-attention GNN model that combines molecular and protein data, using a Transformer-based cross-fusion mechanism.
    """

    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super(MolProteinCoAttentionGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        # Molecular GNN
        self.mol_gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # Protein GNN
        self.protein_gnn = ProteinGNN(num_layer, emb_dim, JK, drop_ratio)

        # Co-attention layer for cross-attention between molecular and protein features
        self.co_attention = CoAttentionLayer(emb_dim)

        # Transformer-based cross-fusion layer
        self.transformer_cross_fusion = TransformerCrossFusion(emb_dim)

        # Molecular and protein feature fusion layer
        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(2 * emb_dim, emb_dim)
        )

    def forward(self, mol_data=None, protein_data=None):
        outputs = {}

        if mol_data is not None:
            # Process molecular data
            mol_repr = self.mol_gnn(mol_data.x, mol_data.edge_index, mol_data.edge_attr)
            outputs['mol_repr'] = mol_repr

        if protein_data is not None:
            # Process protein data
            protein_repr = self.protein_gnn(protein_data.x, protein_data.edge_index, protein_data.edge_attr)
            outputs['protein_repr'] = protein_repr

        if 'mol_repr' in outputs and 'protein_repr' in outputs:
            # Fuse molecular and protein representations using the cross-fusion mechanism
            mol_batch_size = outputs['mol_repr'].size(0)
            protein_batch_size = outputs['protein_repr'].size(0)

            # Ensure both batch sizes are equal; if not, take the smaller one
            min_batch_size = min(mol_batch_size, protein_batch_size)
            mol_repr = outputs['mol_repr'][:min_batch_size]
            protein_repr = outputs['protein_repr'][:min_batch_size]

            # Expand dimensions to fit the co-attention layer (add a node dimension)
            mol_repr_expanded = mol_repr.unsqueeze(1)  # [batch, 1, emb_dim]
            protein_repr_expanded = protein_repr.unsqueeze(1)  # [batch, 1, emb_dim]

            # Use the co-attention mechanism
            mol_enhanced, protein_enhanced = self.co_attention(
                mol_repr_expanded,
                protein_repr_expanded
            )

            # Remove the node dimension
            mol_enhanced = mol_enhanced.squeeze(1)
            protein_enhanced = protein_enhanced.squeeze(1)

            # Use Transformer-based cross-fusion
            cross_fused_repr = self.transformer_cross_fusion(mol_enhanced, protein_enhanced)

            # Fused representation
            fused_repr = self.fusion_layer(
                torch.cat([mol_enhanced, protein_enhanced], dim=-1)
            )
            outputs['fused_repr'] = fused_repr
            outputs['cross_fused_repr'] = cross_fused_repr

        return outputs