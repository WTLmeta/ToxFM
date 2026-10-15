import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros


class CoAttentionLayer(torch.nn.Module):
    """
    Co-attention layer for capturing the interplay between molecular features and gene expression features.
    """

    def __init__(self, emb_dim):
        super(CoAttentionLayer, self).__init__()
        self.emb_dim = emb_dim

        # Mapping from molecular features to attention weights
        self.molecule_attention = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_dim, 1, bias=False)
        )

        # Mapping from gene expression features to attention weights
        self.gene_attention = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_dim, 1, bias=False)
        )

        # Bilinear interaction matrix
        self.W = torch.nn.Parameter(torch.Tensor(emb_dim, emb_dim))

        # Initialize parameters
        glorot(self.W)

    def forward(self, molecule_features, gene_features):
        """
        Forward pass.

        Args:
            molecule_features: molecular features [batch_size, num_atoms, emb_dim]
            gene_features: gene expression features [batch_size, num_genes, emb_dim]

        Returns:
            Enhanced molecular features and gene features.
        """
        batch_size, num_atoms, emb_dim = molecule_features.size()
        _, num_genes, _ = gene_features.size()

        # Compute the interaction matrix
        # molecule_features [batch_size, num_atoms, emb_dim]
        # gene_features [batch_size, num_genes, emb_dim]
        # W [emb_dim, emb_dim]
        interaction = torch.matmul(
            torch.matmul(molecule_features, self.W),
            gene_features.transpose(1, 2)
        )  # [batch_size, num_atoms, num_genes]

        # Compute attention weights
        molecule_att = F.softmax(interaction, dim=2)  # [batch_size, num_atoms, num_genes]
        gene_att = F.softmax(interaction.transpose(1, 2), dim=2)  # [batch_size, num_genes, num_atoms]

        # Apply attention weights
        # Enhance molecular features based on gene features
        enhanced_molecule = torch.matmul(molecule_att, gene_features)  # [batch_size, num_atoms, emb_dim]
        # Enhance gene features based on molecular features
        enhanced_gene = torch.matmul(gene_att, molecule_features)  # [batch_size, num_genes, emb_dim]

        # Residual connection
        enhanced_molecule = enhanced_molecule + molecule_features
        enhanced_gene = enhanced_gene + gene_features

        return enhanced_molecule, enhanced_gene


class TransformerGraphConv(torch.nn.Module):
    """
    Transformer-based graph convolution layer that processes graph-structured data using a self-attention mechanism.

    Args:
        num_layer (int): number of Transformer layers
        emb_dim (int): embedding dimension
        nhead (int): number of attention heads
        drop_ratio (float): dropout rate
        max_seq_len (int): maximum sequence length (number of nodes)
    """

    def __init__(self, num_layer, emb_dim, nhead=8, drop_ratio=0.1, max_seq_len=1000):
        super(TransformerGraphConv, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.nhead = nhead

        # Positional encoding (used to distinguish different nodes) - create a learnable positional encoding
        self.pos_embedding = torch.nn.Parameter(torch.randn(max_seq_len, emb_dim) * 0.02)

        # Transformer encoder layer (uses batch_first=True, input format is [batch, seq_len, emb_dim])
        encoder_layers = torch.nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=emb_dim * 4,
            dropout=drop_ratio,
            activation='relu',
            batch_first=True  # Use [batch, seq_len, emb_dim] format
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layers, num_layers=num_layer)

        # Graph-structure-aware attention bias
        self.graph_bias = SimpleGraphBias()

    def forward(self, x, edge_index, edge_attr=None, batch_size=1):
        """
        Forward pass.

        Args:
            x: node features [num_nodes, emb_dim] (single graph) or [batch_size * num_nodes, emb_dim] (batched giant graph)
            edge_index: edge indices [2, num_edges]
            edge_attr: edge features [num_edges, num_edge_features], optional
            batch_size: batch size, defaults to 1 (single-graph mode). When >1, x is a batched giant graph,
                        the number of nodes is batch_size * lincs_nodes and the positional encoding is reused modulo-wise

        Returns:
            node representations [num_nodes, emb_dim] or [batch_size * num_nodes, emb_dim]
        """
        num_nodes = x.size(0)
        num_nodes_per_graph = num_nodes // batch_size  # number of nodes per graph

        # Add positional encoding: in batched-giant-graph mode, reuse the positional encoding modulo-wise (identical node indices share the positional encoding)
        if batch_size > 1:
            # Generate node indices [0, 1, ..., num_nodes_per_graph-1]
            node_indices = torch.arange(num_nodes_per_graph, device=x.device)
            # Expand to all samples: [batch_size, num_nodes_per_graph] -> [batch_size * num_nodes_per_graph]
            node_indices = node_indices.unsqueeze(0).expand(batch_size, -1).reshape(-1)
            pos_enc = self.pos_embedding[node_indices, :]  # [num_nodes, emb_dim]
        else:
            pos_enc = self.pos_embedding[:num_nodes, :]  # [num_nodes, emb_dim]

        x = x + pos_enc

        # Convert to the format expected by Transformer [batch, seq_len, emb_dim]
        x = x.view(batch_size, num_nodes_per_graph, self.emb_dim)

        # Generate attention bias matrix
        # In batch mode, edge_index contains offsets and must be reduced to the single-graph structure (all samples share the same graph structure)
        if batch_size > 1:
            single_graph_edge_index = edge_index % num_nodes_per_graph
        else:
            single_graph_edge_index = edge_index
        attn_mask = self.graph_bias.generate_mask(single_graph_edge_index, num_nodes_per_graph, x.device)

        # Apply the Transformer encoder
        # TransformerEncoder expects input of shape: [batch, seq_len, emb_dim]
        output = self.transformer_encoder(x, mask=attn_mask)  # [batch_size, num_nodes_per_graph, emb_dim]

        # Restore the original format
        if batch_size > 1:
            output = output.view(-1, self.emb_dim)  # [batch_size * num_nodes_per_graph, emb_dim]
        else:
            output = output.squeeze(0)  # [num_nodes, emb_dim]

        return output


class SimpleGraphBias(torch.nn.Module):
    """
    Simplified graph-bias module that generates an attention mask.
    """

    def __init__(self):
        super(SimpleGraphBias, self).__init__()

    def generate_mask(self, edge_index, num_nodes, device):
        """
        Generate an attention mask matrix (0 where there is a connection, -inf where there is none).

        Args:
            edge_index: edge indices [2, num_edges]
            num_nodes: number of nodes
            device: device

        Returns:
            attention mask matrix [num_nodes, num_nodes]
        """
        # Initialize the mask as -inf (no attention allowed at all)
        mask = torch.full((num_nodes, num_nodes), float('-inf'), device=device)

        # Set the mask to 0 for existing edges (attention allowed)
        if edge_index.size(1) > 0:
            src_nodes = edge_index[0]
            dst_nodes = edge_index[1]
            mask[src_nodes, dst_nodes] = 0.0

        # Add self-loops (each node can attend to itself)
        mask[torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device)] = 0.0

        return mask


class TransformerCrossFusion(torch.nn.Module):
    """
    Transformer-based cross-fusion layer that fuses molecular and protein features.
    """

    def __init__(self, emb_dim, num_heads=8):
        super(TransformerCrossFusion, self).__init__()
        self.emb_dim = emb_dim

        # Automatically adjust the number of heads to ensure emb_dim is divisible by num_heads
        if emb_dim % num_heads != 0:
            # Find the largest factor of emb_dim as the head count, not exceeding the original num_heads
            actual_num_heads = num_heads
            while actual_num_heads > 0 and emb_dim % actual_num_heads != 0:
                actual_num_heads -= 1

            if actual_num_heads == 0:
                # If no suitable factor is found, use a smaller fixed head count
                actual_num_heads = 4 if emb_dim % 4 == 0 else 2 if emb_dim % 2 == 0 else 1

            self.num_heads = actual_num_heads
        else:
            self.num_heads = num_heads

        self.head_dim = emb_dim // self.num_heads

        # Query, Key, Value transformations for molecular features
        self.mol_q = torch.nn.Linear(emb_dim, emb_dim)
        self.mol_k = torch.nn.Linear(emb_dim, emb_dim)
        self.mol_v = torch.nn.Linear(emb_dim, emb_dim)

        # Query, Key, Value transformations for protein features
        self.prot_q = torch.nn.Linear(emb_dim, emb_dim)
        self.prot_k = torch.nn.Linear(emb_dim, emb_dim)
        self.prot_v = torch.nn.Linear(emb_dim, emb_dim)

        # Linear transformation for multi-head attention outputs
        self.mol_out = torch.nn.Linear(emb_dim, emb_dim)
        self.prot_out = torch.nn.Linear(emb_dim, emb_dim)

        # Feed-forward network
        self.mol_ffn = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(4 * emb_dim, emb_dim)
        )
        self.prot_ffn = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(4 * emb_dim, emb_dim)
        )

        # Layer normalization
        self.mol_norm1 = torch.nn.LayerNorm(emb_dim)
        self.mol_norm2 = torch.nn.LayerNorm(emb_dim)
        self.prot_norm1 = torch.nn.LayerNorm(emb_dim)
        self.prot_norm2 = torch.nn.LayerNorm(emb_dim)

    def forward(self, mol_features, prot_features):
        """
        Forward pass.

        Args:
            mol_features: molecular features [batch_size, emb_dim]
            prot_features: protein features [batch_size, emb_dim]

        Returns:
            fused representation
        """
        batch_size = mol_features.size(0)

        # Reshape into multi-head format
        mol_q = self.mol_q(mol_features).view(batch_size, self.num_heads, self.head_dim)
        mol_k = self.mol_k(mol_features).view(batch_size, self.num_heads, self.head_dim)
        mol_v = self.mol_v(mol_features).view(batch_size, self.num_heads, self.head_dim)

        prot_q = self.prot_q(prot_features).view(batch_size, self.num_heads, self.head_dim)
        prot_k = self.prot_k(prot_features).view(batch_size, self.num_heads, self.head_dim)
        prot_v = self.prot_v(prot_features).view(batch_size, self.num_heads, self.head_dim)

        # Compute molecular-protein cross-attention
        # Molecular queries attend to protein keys/values
        mol_prot_attn_scores = torch.matmul(mol_q, prot_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mol_prot_attn_weights = torch.softmax(mol_prot_attn_scores, dim=-1)
        mol_prot_attn_output = torch.matmul(mol_prot_attn_weights, prot_v)
        mol_prot_attn_output = mol_prot_attn_output.view(batch_size, self.emb_dim)

        # Protein queries attend to molecular keys/values
        prot_mol_attn_scores = torch.matmul(prot_q, mol_k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        prot_mol_attn_weights = torch.softmax(prot_mol_attn_scores, dim=-1)
        prot_mol_attn_output = torch.matmul(prot_mol_attn_weights, mol_v)
        prot_mol_attn_output = prot_mol_attn_output.view(batch_size, self.emb_dim)

        # Residual connection and layer normalization
        mol_fused = self.mol_norm1(mol_features + self.mol_out(mol_prot_attn_output))
        prot_fused = self.prot_norm1(prot_features + self.prot_out(prot_mol_attn_output))

        # Feed-forward network
        mol_ffn_output = self.mol_ffn(mol_fused)
        prot_ffn_output = self.prot_ffn(prot_fused)

        # Residual connection and layer normalization
        mol_final = self.mol_norm2(mol_fused + mol_ffn_output)
        prot_final = self.prot_norm2(prot_fused + prot_ffn_output)

        # Final fusion: combine molecular and protein representations
        final_repr = torch.cat([mol_final, prot_final], dim=-1)
        final_repr = torch.nn.functional.relu(final_repr)

        return final_repr