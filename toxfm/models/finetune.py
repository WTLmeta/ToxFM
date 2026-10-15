import torch
import torch.nn.functional as F
import torch_geometric

from toxfm.models.gnn import GNN
from toxfm.models.attention import CoAttentionLayer, TransformerCrossFusion
from toxfm.models.smiles import SMILESContrastivePretrain


class MultimodalFinetuneModel(torch.nn.Module):
    """
    Multimodal model for fine-tuning
    """
    def __init__(self, vocab_size, num_layer, emb_dim, JK="last",
                 drop_ratio=0.1, gnn_type="gin", nhead=8, lincs_nodes=978, num_classes=2):
        super(MultimodalFinetuneModel, self).__init__()

        # Molecular GNN encoder (graph structure)
        self.mol_gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # SMILES text encoder (uses the new MLM + contrastive learning module, consistent with pretraining)
        self.smiles_encoder = SMILESContrastivePretrain(
            vocab_size=vocab_size,
            embed_dim=emb_dim,
            nhead=nhead,
            num_layers=4,
            max_len=128,
            temperature=0.07
        )

        # LINCS graph encoder (Graph2Vec style: WL subtree kernel + attention pooling)
        # Graph2Vec core: multi-hop neighbor aggregation + learnable readout function
        self.lincs_gnn_wl_layers = torch.nn.ModuleList([
            torch.nn.Linear(emb_dim, emb_dim) for _ in range(num_layer)
        ])
        self.lincs_gnn_wl_norm = torch.nn.ModuleList([
            torch.nn.LayerNorm(emb_dim) for _ in range(num_layer)
        ])

        # Readout function: attention-weighted pooling + max pooling
        self.lincs_graph_attention_pool = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim // 2),
            torch.nn.LayerNorm(emb_dim // 2),
            torch.nn.GELU(),
            torch.nn.Linear(emb_dim // 2, 1)
        )
        self.lincs_graph_readout = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim, emb_dim)
        )

        # LINCS embedding encoder (processes node features)
        self.lincs_embed_encoder = torch.nn.Sequential(
            torch.nn.Linear(lincs_nodes, emb_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.ReLU()
        )

        # LINCS graph node feature extractor (enhanced: deeper nonlinear transform, boosting class discrimination)
        self.gene_to_node_feature = torch.nn.Sequential(
            torch.nn.Linear(1, emb_dim * 2),
            torch.nn.LayerNorm(emb_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.LayerNorm(emb_dim),
            torch.nn.GELU()
        )

        # Co-attention layers
        self.mol_smiles_co_attention = CoAttentionLayer(emb_dim)
        self.mol_lincs_embed_co_attention = CoAttentionLayer(emb_dim)
        self.mol_lincs_graph_co_attention = CoAttentionLayer(emb_dim)
        self.smiles_lincs_embed_co_attention = CoAttentionLayer(emb_dim)
        self.smiles_lincs_graph_co_attention = CoAttentionLayer(emb_dim)
        self.lincs_embed_graph_co_attention = CoAttentionLayer(emb_dim)

        # Transformer cross-fusion layer
        self.transformer_cross_fusion = TransformerCrossFusion(emb_dim, nhead)

        # Graph-level pooling (improved: combining multiple pooling strategies)
        self.global_pool_mean = torch_geometric.nn.global_mean_pool
        self.global_pool_max = torch_geometric.nn.global_max_pool

        # Graph-level attention pooling
        self.mol_attention_pool = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_dim // 2, 1)
        )

        # Pooling fusion layer
        self.pool_fusion = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 3, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_ratio)
        )

        # Fine-tuning classification head
        self.classification_head = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(drop_ratio),
            torch.nn.Linear(emb_dim // 2, num_classes)
        )

        # Store parameters
        self.vocab_size = vocab_size
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.JK = JK
        self.drop_ratio = drop_ratio
        self.gnn_type = gnn_type
        self.nhead = nhead
        self.lincs_nodes = lincs_nodes
        self.num_classes = num_classes

        # SMILES to common-dimension projection layer (consistent with pretraining, adapts RNN output dimension)
        self.smiles_to_common_proj = torch.nn.Linear(emb_dim * 2, emb_dim)

        # SMILES attention pooling layer (improved: uses attention to aggregate the whole sequence)
        self.smiles_attention_pool = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, emb_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(emb_dim, 1)
        )

    def forward(self, mol_data=None, smiles_tokens=None,
                lincs_edge_index=None, lincs_edge_attr=None, lincs_embed=None,
                batch=None, modalities=None, labels=None):
        """
        Forward pass supporting different modality combinations
        """
        representations = {}

        # Get batch_size (prefer labels, since labels always exist)
        if labels is not None:
            actual_batch_size = labels.size(0)
        elif batch is not None:
            actual_batch_size = batch.max().item() + 1 if batch.numel() > 0 else 1
        elif lincs_embed is not None:
            actual_batch_size = lincs_embed.size(0)
        else:
            actual_batch_size = 1

        # Decide which modalities to process according to the modalities parameter
        if modalities is None:
            use_mol = mol_data is not None
            use_smiles = smiles_tokens is not None
            use_lincs_embed = lincs_embed is not None
            use_lincs_graph = lincs_edge_index is not None
        else:
            use_mol = 'mol' in modalities and mol_data is not None
            use_smiles = 'smiles' in modalities and smiles_tokens is not None
            use_lincs_embed = 'lincs_embed' in modalities and lincs_embed is not None
            use_lincs_graph = 'lincs_graph' in modalities and lincs_edge_index is not None

        # Process molecular graph data
        if use_mol:
            # Ensure all molecular graph data is on the same device
            mol_x = mol_data.x.to(next(self.parameters()).device)
            mol_edge_index = mol_data.edge_index.to(next(self.parameters()).device)
            mol_edge_attr = mol_data.edge_attr.to(next(self.parameters()).device) if mol_data.edge_attr is not None else None
            mol_node_repr = self.mol_gnn(mol_x, mol_edge_index, mol_edge_attr)

            # Improved: combining multiple pooling strategies (mean + max + attention)
            if batch is not None:
                batch = batch.to(next(self.parameters()).device)

                # 1. Mean pooling
                mol_mean_pool = self.global_pool_mean(mol_node_repr, batch)

                # 2. Max pooling
                mol_max_pool = self.global_pool_max(mol_node_repr, batch)

                # 3. Attention pooling
                attn_weights = self.mol_attention_pool(mol_node_repr)  # [num_nodes, 1]
                attn_weights = torch_geometric.utils.softmax(attn_weights, batch)  # softmax per graph
                mol_attn_pool = torch_geometric.nn.global_add_pool(mol_node_repr * attn_weights, batch)

                # Fuse the three pooling results
                mol_graph_repr = torch.cat([mol_mean_pool, mol_max_pool, mol_attn_pool], dim=-1)
                mol_graph_repr = self.pool_fusion(mol_graph_repr)
            else:
                # If there is only one graph, take the mean
                mol_graph_repr = torch.mean(mol_node_repr, dim=0, keepdim=True)
            representations['mol_graph_repr'] = mol_graph_repr

        # Process SMILES text data
        if use_smiles:
            smiles_tokens = smiles_tokens.to(next(self.parameters()).device)
            # Use SMILESContrastivePretrain's forward method (consistent with pretraining)
            # During fine-tuning, MLM and contrastive learning are not needed; only representations are needed
            _, smiles_embeddings = self.smiles_encoder.encoder(smiles_tokens, return_embeddings=True)
            # RNN output is [batch_size, seq_len, emb_dim*2]

            # Improved: use attention pooling to aggregate the whole sequence (instead of first-position extraction)
            # Compute attention weights
            attention_logits = self.smiles_attention_pool(smiles_embeddings)  # [batch_size, seq_len, 1]
            attention_weights = torch.softmax(attention_logits, dim=1)  # softmax along the sequence dimension

            # Weighted sum to obtain the sequence representation
            smiles_repr_raw = (smiles_embeddings * attention_weights).sum(dim=1)  # [batch_size, emb_dim*2]

            # Project to emb_dim for fusion with other modalities (consistent with pretraining)
            smiles_repr = self.smiles_to_common_proj(smiles_repr_raw)  # [batch_size, emb_dim]
            representations['smiles_repr'] = smiles_repr

        # Process LINCS embedding data
        if use_lincs_embed:
            lincs_embed = lincs_embed.to(next(self.parameters()).device)
            lincs_embed_repr = self.lincs_embed_encoder(lincs_embed)  # [batch_size, emb_dim]
            representations['lincs_embed_repr'] = lincs_embed_repr

        # Process LINCS graph data
        if use_lincs_graph and lincs_edge_index is not None:
            lincs_edge_index = lincs_edge_index.to(next(self.parameters()).device)
            lincs_edge_attr = lincs_edge_attr.to(next(self.parameters()).device) if lincs_edge_attr is not None else None

            # Use the actual batch_size
            batch_size = actual_batch_size

            # Use gene expression values as node features
            # lincs_embed: [batch_size, 978] - expression values of 978 genes per sample
            # Target: [batch_size, 978, emb_dim] - each gene as a node
            if lincs_embed is not None:
                lincs_embed = lincs_embed.to(next(self.parameters()).device)

                # Reshape lincs_embed to [batch_size * 978, 1]
                lincs_flat = lincs_embed.view(-1, 1)  # [batch_size * 978, 1]

                # Project to emb_dim (using gene_to_node_feature)
                node_features = self.gene_to_node_feature(lincs_flat)  # [batch_size * 978, emb_dim]

                # Reshape back to [batch_size, 978, emb_dim]
                node_features = node_features.view(batch_size, self.lincs_nodes, self.emb_dim)
            else:
                # If no lincs_embed, create learnable node features
                node_features = torch.randn(batch_size, self.lincs_nodes, self.emb_dim,
                                                  device=lincs_edge_index.device)

            # Graph2Vec style: WL subtree kernel + attention pooling
            h = node_features  # [batch_size, lincs_nodes, emb_dim]
            h_list = [h]

            # Multi-hop neighbor aggregation (WL subtree kernel iteration) - use scatter_reduce_ to avoid memory explosion
            src_idx = lincs_edge_index[0]  # [num_edges]
            dst_idx = lincs_edge_index[1]  # [num_edges]

            for layer_idx in range(len(self.lincs_gnn_wl_layers)):
                # Aggregate neighbor features sample by sample (avoid large tensors)
                neighbor_msgs_list = []
                for i in range(batch_size):
                    msg = torch.zeros_like(h[i])  # [lincs_nodes, emb_dim]
                    # Aggregate source node features into target nodes
                    msg.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand(-1, self.emb_dim),
                                       h[i][src_idx], reduce='mean', include_self=False)
                    neighbor_msgs_list.append(msg)

                neighbor_msgs = torch.stack(neighbor_msgs_list, dim=0)  # [batch_size, lincs_nodes, emb_dim]

                # Edge weights (if any)
                if lincs_edge_attr is not None:
                    # Weight the neighbor messages
                    for i in range(batch_size):
                        weighted_msg = torch.zeros_like(h[i])
                        edge_w = lincs_edge_attr.squeeze(-1) if lincs_edge_attr.dim() > 1 else lincs_edge_attr
                        weighted_msg.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand(-1, self.emb_dim),
                                                    (h[i][src_idx] * edge_w.unsqueeze(-1)), reduce='mean', include_self=False)
                        neighbor_msgs[i] = weighted_msg

                # Update node features
                h = h + neighbor_msgs  # residual connection
                h = self.lincs_gnn_wl_norm[layer_idx](h)
                h = F.gelu(self.lincs_gnn_wl_layers[layer_idx](h))
                h_list.append(h)

            # 3. Readout function: attention-weighted pooling (replacing simple mean+max, boosting class discrimination)
            # Use the last layer's node representation
            final_h = h_list[-1]  # [batch_size, lincs_nodes, emb_dim]

            # Learnable attention-weighted pooling
            attention_logits = self.lincs_graph_attention_pool(final_h)  # [batch_size, lincs_nodes, 1]
            attention_weights = F.softmax(attention_logits, dim=1)  # softmax along the node dimension
            attn_pool = (final_h * attention_weights).sum(dim=1)  # [batch_size, emb_dim]

            # Keep max pooling as a supplement
            max_pool = final_h.max(dim=1).values  # [batch_size, emb_dim]

            # Concatenate attention pooling and max pooling
            graph_repr = torch.cat([attn_pool, max_pool], dim=-1)  # [batch_size, emb_dim * 2]
            lincs_graph_repr = self.lincs_graph_readout(graph_repr)  # [batch_size, emb_dim]

            # Ensure on the correct device
            lincs_graph_repr = lincs_graph_repr.to(next(self.parameters()).device)
            representations['lincs_graph_repr'] = lincs_graph_repr

        # Fuse multiple representations
        fused_repr = self._fuse_representations(representations, modalities)
        # Ensure the fused representation is on the correct device
        fused_repr = fused_repr.to(next(self.parameters()).device)

        # Classification output
        logits = self.classification_head(fused_repr)
        return logits

    def _fuse_representations(self, representations, modalities=None):
        """
        Fuse multiple representations
        """
        repr_list = []
        device = next(self.parameters()).device

        if 'mol_graph_repr' in representations:
            repr_list.append(representations['mol_graph_repr'].to(device))

        if 'smiles_repr' in representations:
            repr_list.append(representations['smiles_repr'].to(device))

        if 'lincs_embed_repr' in representations:
            repr_list.append(representations['lincs_embed_repr'].to(device))

        if 'lincs_graph_repr' in representations:
            repr_list.append(representations['lincs_graph_repr'].to(device))

        if len(repr_list) == 0:
            raise ValueError("至少需要一种类型的输入表示")

        # If there is only one representation, return it directly
        if len(repr_list) == 1:
            return repr_list[0]

        # Check dimension consistency and adapt if needed
        base_shape = repr_list[0].shape
        for i, repr_tensor in enumerate(repr_list[1:], 1):
            if repr_tensor.shape != base_shape:
                # Adjust the second dimension size
                if len(base_shape) == 2 and len(repr_tensor.shape) == 2:
                    if repr_tensor.shape[1] != base_shape[1]:
                        repr_list[i] = torch.nn.functional.pad(
                            repr_tensor, (0, base_shape[1] - repr_tensor.shape[1]), "constant", 0
                        )

        # If there are two representations, use the co-attention mechanism
        if len(repr_list) == 2:
            repr1, repr2 = repr_list[0], repr_list[1]
            if repr1.shape != repr2.shape:
                max_shape = tuple(max(s1, s2) for s1, s2 in zip(repr1.shape, repr2.shape))

                pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(repr1.shape, max_shape))
                pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                repr1_padded = torch.nn.functional.pad(repr1, pad_tuple, "constant", 0)

                pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(repr2.shape, max_shape))
                pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                repr2_padded = torch.nn.functional.pad(repr2, pad_tuple, "constant", 0)
            else:
                repr1_padded, repr2_padded = repr1, repr2

            repr1_exp = repr1_padded.unsqueeze(1) if repr1_padded.dim() == 2 else repr1_padded
            repr2_exp = repr2_padded.unsqueeze(1) if repr2_padded.dim() == 2 else repr2_padded

            # Use an appropriate co-attention layer (simplified here; can be chosen by modality type)
            attended_repr1, attended_repr2 = self.mol_smiles_co_attention(repr1_exp, repr2_exp)

            attended_repr1 = attended_repr1.squeeze(1) if attended_repr1.dim() == 3 else attended_repr1
            attended_repr2 = attended_repr2.squeeze(1) if attended_repr2.dim() == 3 else attended_repr2

            if torch.isnan(attended_repr1).any() or torch.isinf(attended_repr1).any():
                attended_repr1 = torch.nan_to_num(attended_repr1)
            if torch.isnan(attended_repr2).any() or torch.isinf(attended_repr2).any():
                attended_repr2 = torch.nan_to_num(attended_repr2)

            cross_fused = self.transformer_cross_fusion(attended_repr1, attended_repr2)

            if torch.isnan(cross_fused).any() or torch.isinf(cross_fused).any():
                cross_fused = torch.nan_to_num(cross_fused)

            fusion_layer = torch.nn.Sequential(
                torch.nn.Linear(self.emb_dim * 2, self.emb_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.drop_ratio)
            ).to(attended_repr1.device)

            fused = torch.cat([attended_repr1, attended_repr2], dim=-1)
            return fusion_layer(fused)

        combined_repr = torch.cat(repr_list, dim=-1)

        fusion_layer = torch.nn.Sequential(
            torch.nn.Linear(len(repr_list) * self.emb_dim, self.emb_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(self.drop_ratio),
            torch.nn.Linear(self.emb_dim * 2, self.emb_dim),
            torch.nn.ReLU()
        ).to(combined_repr.device)

        return fusion_layer(combined_repr)