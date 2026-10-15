import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from toxfm.models.gnn import GNN
from toxfm.models.attention import CoAttentionLayer, TransformerCrossFusion
from toxfm.models.smiles import SMILESContrastivePretrain


class DynamicLossBalancer:
    """Dynamic loss balancer for balancing the losses of different modalities."""

    def __init__(self, modalities, initial_weights=None, momentum=0.9):
        """
        Initialize the dynamic loss balancer.

        Args:
            modalities: List of modalities.
            initial_weights: Initial weight dictionary.
            momentum: Momentum parameter used to smooth loss changes.
        """
        self.modalities = modalities
        self.momentum = momentum

        # Initialize weights
        if initial_weights:
            self.weights = initial_weights.copy()
        else:
            # Default initial weight is 1.0
            self.weights = {mod: 1.0 for mod in modalities}

        # Record historical losses for dynamic adjustment
        self.history_losses = {mod: [] for mod in modalities}
        self.avg_losses = {mod: 1.0 for mod in modalities}

    def update_weights(self, losses):
        """
        Update weights based on the current losses.

        Args:
            losses: Dictionary of the current losses for each modality.
        """
        for mod, loss_val in losses.items():
            if mod in self.modalities:
                # Update average loss
                self.avg_losses[mod] = self.momentum * self.avg_losses[mod] + (1 - self.momentum) * loss_val

                # Avoid the weight becoming too large when the average loss is too small
                if self.avg_losses[mod] > 1e-6:
                    # Adjust weight based on the relative loss magnitude.
                    # The larger the loss, the smaller the weight; the smaller the loss, the larger the weight.
                    inv_avg_loss = 1.0 / self.avg_losses[mod]
                    self.weights[mod] = inv_avg_loss / len(self.modalities)

    def get_weights(self):
        """Get the current weights."""
        return self.weights

    def scale_losses(self, losses):
        """Scale losses according to the current weights."""
        scaled_losses = {}
        for mod, loss_val in losses.items():
            if mod in self.weights:
                scaled_losses[mod] = loss_val * self.weights[mod]
            else:
                scaled_losses[mod] = loss_val
        return scaled_losses


class MultimodalPretrainModel(nn.Module):
    """
    Multimodal pretraining model combining a molecular graph neural network, SMILES text,
    LINCS embeddings and LINCS graph structure.
    Uses crossfusion to fuse the representations of the four modalities.
    """

    def __init__(self, vocab_size, num_layer, emb_dim, JK="last",
                 drop_ratio=0.1, gnn_type="gin", nhead=8, lincs_nodes=978):
        super(MultimodalPretrainModel, self).__init__()

        # Molecular GNN encoder (graph structure)
        self.mol_gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        # SMILES text encoder (using the new MLM + contrastive learning module)
        self.smiles_encoder = SMILESContrastivePretrain(
            vocab_size=vocab_size,
            embed_dim=emb_dim,
            nhead=nhead,
            num_layers=4,
            max_len=128,
            temperature=0.07
        )

        self.lincs_gnn_wl_layers = nn.ModuleList([
            nn.Linear(emb_dim, emb_dim) for _ in range(num_layer)
        ])
        self.lincs_gnn_wl_norm = nn.ModuleList([
            nn.BatchNorm1d(emb_dim) for _ in range(num_layer)
        ])
        
        # Readout function: global average pooling + max pooling concatenation
        self.lincs_graph_readout = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),  # Concatenate mean and max
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim, emb_dim)
        )

        # LINCS embedding encoder (processing node features)
        self.lincs_embed_encoder = nn.Sequential(
            nn.Linear(lincs_nodes, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU()
        )
        
        # LINCS graph node feature extractor
        self.gene_to_node_feature = nn.Sequential(
            nn.Linear(1, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.GELU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU()
        )
        
        # LINCS graph learnable attention pooling layer
        self.lincs_graph_attention_pool = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.LayerNorm(emb_dim // 2),
            nn.GELU(),
            nn.Linear(emb_dim // 2, 1)
        )
        
        # LINCS graph-to-embedding projection layer
        self.lincs_graph_to_embed_proj = nn.Linear(emb_dim, emb_dim)
        
        # LINCS graph mask reconstruction head
        self.lincs_graph_node_recon_head = nn.Linear(emb_dim, 1)  
        self.lincs_graph_global_recon_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(emb_dim * 2, lincs_nodes)  
        )

        # Molecular graph-SMILES text co-attention layer
        self.mol_smiles_co_attention = CoAttentionLayer(emb_dim)

        # Molecular graph-LINCS embedding co-attention layer
        self.mol_lincs_embed_co_attention = CoAttentionLayer(emb_dim)

        # Molecular graph-LINCS graph co-attention layer
        self.mol_lincs_graph_co_attention = CoAttentionLayer(emb_dim)

        # SMILES text-LINCS embedding co-attention layer
        self.smiles_lincs_embed_co_attention = CoAttentionLayer(emb_dim)

        # SMILES text-LINCS graph co-attention layer
        self.smiles_lincs_graph_co_attention = CoAttentionLayer(emb_dim)

        # LINCS embedding-LINCS graph co-attention layer
        self.lincs_embed_graph_co_attention = CoAttentionLayer(emb_dim)

        # Transformer cross-fusion layer
        self.transformer_cross_fusion = TransformerCrossFusion(emb_dim, nhead)

        # Graph-level pooling
        self.global_pool = torch_geometric.nn.global_mean_pool

        # Store parameters
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.lincs_nodes = lincs_nodes
        self.vocab_size = vocab_size

        # Molecular graph-SMILES contrastive learning projection head
        self.mol_proj = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        # SMILES projection head (adapted to RNN output dimension: emb_dim * 2)
        self.smiles_proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),  # RNN output is 512 (256*2)
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
        # SMILES to common dimension projection layer (for multimodal fusion)
        self.smiles_to_common_proj = nn.Linear(emb_dim * 2, emb_dim)
        
        # LINCS embedding reconstruction head
        self.lincs_embed_recon_head = nn.Linear(emb_dim, lincs_nodes)
        
        # LINCS graph edge prediction head (new: used for the edge prediction task)
        self.lincs_graph_edge_pred_head = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),  # Concatenate the features of two nodes
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim, 1)  # Output the existence probability or weight of an edge
        )
        
        # LINCS graph - embedding contrastive learning projection head (used to align the two LINCS modalities)
        self.lincs_graph_contrast_proj = nn.Linear(emb_dim, emb_dim)
        self.lincs_embed_contrast_proj = nn.Linear(emb_dim, emb_dim)
        
        # LINCS graph - molecular graph contrastive learning projection head
        self.lincs_graph_mol_contrast_proj = nn.Linear(emb_dim, emb_dim)
        
        self.fusion_layer_2mod = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU()
        )
        self.fusion_layer_3mod = nn.Sequential(
            nn.Linear(emb_dim * 3, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU()
        )

        self.fusion_layer_4mod = nn.Sequential(
            nn.Linear(emb_dim * 4, emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.ReLU()
        )

    def forward(self, mol_data=None, smiles_tokens=None, augmented_smiles_tokens=None, 
                masked_smiles_tokens=None, mlm_labels=None,
                lincs_edge_index=None, lincs_edge_attr=None, lincs_embed=None,
                lincs_masked_embed=None, lincs_mask=None,
                batch=None, modalities=None):
        """
        Forward pass (supports MLM + contrastive learning + LINCS graph mask reconstruction).

        Args:
            mol_data: Molecular graph data (PyG Data object).
            smiles_tokens: Original SMILES token sequence [batch_size, seq_len].
            augmented_smiles_tokens: Augmented SMILES token sequence (for contrastive learning).
            masked_smiles_tokens: Masked SMILES token sequence (for MLM).
            mlm_labels: MLM labels.
            lincs_edge_index: LINCS graph edge index [2, num_edges].
            lincs_edge_attr: LINCS graph edge features [num_edges, num_edge_features].
            lincs_embed: LINCS embeddings [batch_size, lincs_nodes] (node features).
            lincs_masked_embed: Masked LINCS embeddings (for pretraining).
            lincs_mask: LINCS mask indicator [batch_size, lincs_nodes].
            batch: Batch information (for graph pooling).
            modalities: List of modalities to use, e.g. ['mol', 'smiles', 'lincs_embed', 'lincs_graph']; if None, all modalities are used.
        """
        representations = {}
        smiles_losses = {}

        # Decide which modalities to process based on the modalities parameter
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
            mol_graph_repr = self.mol_gnn(mol_data.x, mol_data.edge_index, mol_data.edge_attr)
            # Graph-level representation
            if batch is not None:
                mol_graph_repr = self.global_pool(mol_graph_repr, batch)
            else:
                # If there is only one graph, take the mean
                mol_graph_repr = torch.mean(mol_graph_repr, dim=0, keepdim=True)
            representations['mol_graph_repr'] = mol_graph_repr
            
            # Contrastive learning projection
            representations['mol_proj'] = self.mol_proj(mol_graph_repr)

        # Process SMILES text data (using the new MLM + contrastive learning module)
        if use_smiles:
            # Use the forward method of SMILESContrastivePretrain
            if augmented_smiles_tokens is not None and masked_smiles_tokens is not None and mlm_labels is not None:
                smiles_losses = self.smiles_encoder(
                    input_ids=smiles_tokens,
                    augmented_input_ids=augmented_smiles_tokens,
                    masked_input_ids=masked_smiles_tokens,
                    labels=mlm_labels,
                    return_loss=True
                )
                
                # Get the encoder representation (for later fusion)
                _, smiles_embeddings = self.smiles_encoder.encoder(smiles_tokens, return_embeddings=True)
                # RNN output is [batch, seq_len, emb_dim*2], take the first position
                smiles_repr = smiles_embeddings[:, 0, :]  # [batch, emb_dim*2]
                
                # Project to emb_dim for fusion with other modalities
                smiles_repr_common = self.smiles_to_common_proj(smiles_repr)  # [batch, emb_dim]
                
                representations['smiles_repr'] = smiles_repr_common
                representations['smiles_repr_raw'] = smiles_repr  # Store the raw 512-dimensional representation
                
                # Contrastive learning projection (using the raw 512-dimensional representation)
                representations['smiles_proj'] = self.smiles_proj(smiles_repr)
            else:
                # Inference-only mode
                _, smiles_embeddings = self.smiles_encoder.encoder(smiles_tokens, return_embeddings=True)
                smiles_repr = smiles_embeddings[:, 0, :]  # [batch, emb_dim*2]
                
                # Project to emb_dim
                smiles_repr_common = self.smiles_to_common_proj(smiles_repr)
                
                representations['smiles_repr'] = smiles_repr_common
                representations['smiles_repr_raw'] = smiles_repr
                representations['smiles_proj'] = self.smiles_proj(smiles_repr)

        # Process LINCS embedding data
        if use_lincs_embed:
            # Use the masked embedding as encoder input to avoid label leakage
            lincs_embed_input = lincs_masked_embed if lincs_masked_embed is not None else lincs_embed
            lincs_embed_repr = self.lincs_embed_encoder(lincs_embed_input)  # [batch_size, emb_dim]
            representations['lincs_embed_repr'] = lincs_embed_repr
            
            # Project to the contrastive learning space
            lincs_embed_proj = self.lincs_embed_contrast_proj(lincs_embed_repr)
            representations['lincs_embed_proj'] = lincs_embed_proj

        # Process LINCS graph data
        if use_lincs_graph:
            # If node embeddings are provided, use them; otherwise create dummy node features based on batch size
            if lincs_embed is not None:
                # Convert LINCS embeddings into node features
                batch_size = lincs_embed.size(0)
                lincs_nodes = self.lincs_nodes
                
                # ========== Graph2Vec-style feature encoding ==========
                # Graph2Vec core: WL subtree kernel iterative aggregation + global readout function
                
                # 1. Initialize node features: map gene expression values to emb_dim
                # Use lincs_masked_embed (if available) to avoid label leakage, otherwise use lincs_embed
                lincs_input = lincs_masked_embed if lincs_masked_embed is not None else lincs_embed
                lincs_flat = lincs_input.view(-1, 1)  # [batch_size * lincs_nodes, 1]
                node_features = self.gene_to_node_feature(lincs_flat)  # [batch_size * lincs_nodes, emb_dim]
                h = node_features.view(batch_size, lincs_nodes, self.emb_dim)
                
                # 2. WL subtree kernel iteration: multi-hop neighbor aggregation
                h_list = [h]  # Save the node representation of each layer
                
                for k in range(len(self.lincs_gnn_wl_layers)):
                    # Message passing: aggregate neighbor information
                    # edge_index: [2, num_edges] - single graph structure (0~lincs_nodes-1)
                    src_idx = lincs_edge_index[0]  # [num_edges]
                    dst_idx = lincs_edge_index[1]  # [num_edges]
                    
                    # For each sample, aggregate neighbor features
                    neighbor_msgs_list = []
                    for i in range(batch_size):
                        # h[i]: [lincs_nodes, emb_dim]
                        # Use scatter for message aggregation (edge_index is a single graph, use it directly)
                        msg = torch.zeros_like(h[i])  # [lincs_nodes, emb_dim]
                        msg.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand(-1, self.emb_dim), 
                                           h[i][src_idx], reduce='mean', include_self=False)
                        neighbor_msgs_list.append(msg)
                    
                    neighbor_msgs = torch.stack(neighbor_msgs_list, dim=0)  # [batch_size, lincs_nodes, emb_dim]
                    
                    # WL update: h = f(h + neighbor_msg)
                    h = h + neighbor_msgs  # Residual connection
                    h = h.view(-1, self.emb_dim)  # [batch_size * lincs_nodes, emb_dim]
                    h = self.lincs_gnn_wl_layers[k](h)
                    h = self.lincs_gnn_wl_norm[k](h)
                    h = F.relu(h)
                    h = h.view(batch_size, lincs_nodes, self.emb_dim)
                    
                    h_list.append(h)
                
                # 3. Readout function: attention-weighted pooling (replaces simple mean+max to enhance class discriminability)
                # Use the node representation of the last layer
                final_h = h_list[-1]  # [batch_size, lincs_nodes, emb_dim]
                
                # Learnable attention-weighted pooling
                attention_logits = self.lincs_graph_attention_pool(final_h)  # [batch_size, lincs_nodes, 1]
                attention_weights = F.softmax(attention_logits, dim=1)  # Softmax along the node dimension
                attn_pool = (final_h * attention_weights).sum(dim=1)  # [batch_size, emb_dim]
                
                # Keep max pooling as a supplement
                max_pool = final_h.max(dim=1).values  # [batch_size, emb_dim]
                
                # Concatenate attention pooling and max pooling
                graph_repr = torch.cat([attn_pool, max_pool], dim=-1)  # [batch_size, emb_dim * 2]
                lincs_graph_repr = self.lincs_graph_readout(graph_repr)  # [batch_size, emb_dim]
                # ========================================================
                
                # Project to the contrastive learning space
                lincs_graph_proj = self.lincs_graph_contrast_proj(lincs_graph_repr)
                representations['lincs_graph_proj'] = lincs_graph_proj
                
                # 3. Edge prediction task: predict the existence or weight of protein interaction edges
                # Get all node representations [batch_size, lincs_nodes, emb_dim]
                lincs_node_reprs = h  # Use the node representation of the last WL iteration
                
                # Predict the weight of each edge (batched in parallel)
                src_nodes = lincs_edge_index[0]  # [num_edges]
                dst_nodes = lincs_edge_index[1]  # [num_edges]
                
                # For each sample, extract the node features of the corresponding edges
                # lincs_node_reprs: [batch_size, lincs_nodes, emb_dim]
                src_features = lincs_node_reprs[:, src_nodes, :]  # [batch_size, num_edges, emb_dim]
                dst_features = lincs_node_reprs[:, dst_nodes, :]  # [batch_size, num_edges, emb_dim]
                
                # Concatenate source and destination node features
                edge_features = torch.cat([src_features, dst_features], dim=-1)  # [batch_size, num_edges, emb_dim*2]
                
                # Predict edge weights in batch
                edge_preds = self.lincs_graph_edge_pred_head(edge_features).squeeze(-1)  # [batch_size, num_edges]
                
                # Compute edge prediction loss (using actual edge weights as labels)
                if lincs_edge_attr is not None:
                    # Actual edge weights [num_edges] or [num_edges, 1]
                    actual_edge_weights = lincs_edge_attr.squeeze(-1) if lincs_edge_attr.dim() > 1 else lincs_edge_attr
                    
                    # Batch MSE loss
                    edge_pred_loss = F.mse_loss(edge_preds, actual_edge_weights.unsqueeze(0).expand(batch_size, -1), reduction='mean')
                    representations['lincs_graph_edge_pred_loss'] = edge_pred_loss
                
                # If there is a mask, compute reconstruction loss (enhanced pretraining method)
                if lincs_masked_embed is not None and lincs_mask is not None:
                    # 1. Node-level reconstruction: each node reconstructs its corresponding gene expression value
                    lincs_node_recon = self.lincs_graph_node_recon_head(h.view(-1, self.emb_dim)).squeeze(-1)  # [batch_size * lincs_nodes]
                    lincs_node_recon = lincs_node_recon.view(batch_size, lincs_nodes)  # [batch_size, lincs_nodes]
                    
                    # 2. Graph-level reconstruction: reconstruct the full gene expression vector using the graph representation (similar to lincs_embed)
                    lincs_graph_recon = self.lincs_graph_global_recon_head(lincs_graph_repr)  # [batch_size, lincs_nodes]
                    
                    # Compute the two reconstruction losses
                    if lincs_mask.sum() > 0:
                        # Node-level loss: only compute at masked positions
                        mask_flat = lincs_mask.reshape(-1)
                        lincs_flat = lincs_embed.reshape(-1)
                        masked_indices = mask_flat.bool()
                        
                        node_recon_loss = F.mse_loss(
                            lincs_node_recon.reshape(-1)[masked_indices],
                            lincs_flat[masked_indices],
                            reduction='mean'
                        )
                        
                        # Graph-level loss: compute MSE over all genes (computed even without masking, to provide global supervision)
                        graph_recon_loss = F.mse_loss(
                            lincs_graph_recon,
                            lincs_embed,
                            reduction='mean'
                        )
                        
                        # Combine the two losses (node-level focuses on masked positions, graph-level provides global constraint)
                        total_graph_loss = node_recon_loss + 0.5 * graph_recon_loss
                        representations['lincs_graph_recon_loss'] = total_graph_loss
                        representations['lincs_graph_node_loss'] = node_recon_loss
                        representations['lincs_graph_global_loss'] = graph_recon_loss
            else:
                # If no node embeddings are provided, create dummy node features and use Graph2Vec-style encoding
                batch_size = lincs_embed.size(0) if lincs_embed is not None else 1
                
                # Create dummy node features with shape [batch_size, lincs_nodes, emb_dim]
                h = torch.randn(batch_size, self.lincs_nodes, self.emb_dim,
                              device=lincs_edge_index.device)
                
                # WL subtree kernel iteration: multi-hop neighbor aggregation
                h_list = [h]
                
                for k in range(len(self.lincs_gnn_wl_layers)):
                    src_idx = lincs_edge_index[0]
                    dst_idx = lincs_edge_index[1]
                    
                    neighbor_msgs_list = []
                    for i in range(batch_size):
                        msg = torch.zeros_like(h[i])
                        msg.scatter_reduce_(0, dst_idx.unsqueeze(-1).expand(-1, self.emb_dim),
                                           h[i][src_idx], reduce='mean', include_self=False)
                        neighbor_msgs_list.append(msg)
                    
                    neighbor_msgs = torch.stack(neighbor_msgs_list, dim=0)
                    h = h + neighbor_msgs
                    h = h.view(-1, self.emb_dim)
                    h = self.lincs_gnn_wl_layers[k](h)
                    h = self.lincs_gnn_wl_norm[k](h)
                    h = F.relu(h)
                    h = h.view(batch_size, self.lincs_nodes, self.emb_dim)
                    h_list.append(h)
                
                # Readout function: global pooling
                final_h = h_list[-1]
                mean_pool = final_h.mean(dim=1)
                max_pool = final_h.max(dim=1).values
                graph_repr = torch.cat([mean_pool, max_pool], dim=-1)
                lincs_graph_repr = self.lincs_graph_readout(graph_repr)

            representations['lincs_graph_repr'] = lincs_graph_repr

        # Fuse multiple representations - dynamically create fusion layers based on the number of modalities
        fused_repr = self._fuse_representations(representations, modalities)
        representations['fused_repr'] = fused_repr

        # Output heads
        if 'fused_repr' in representations:
            # If the LINCS embedding modality is used, perform LINCS embedding reconstruction
            if use_lincs_embed:
                lincs_embed_recon = self.lincs_embed_recon_head(
                    representations['fused_repr'])  # [batch_size, lincs_nodes]
                representations['lincs_embed_recon'] = lincs_embed_recon

        # Compute LINCS graph - embedding contrastive learning loss (if both modalities exist)
        if use_lincs_graph and use_lincs_embed and 'lincs_graph_proj' in representations and 'lincs_embed_proj' in representations:
            # Get the projected representations
            graph_proj = representations['lincs_graph_proj']  # [batch_size, emb_dim]
            embed_proj = representations['lincs_embed_proj']  # [batch_size, emb_dim]
            
            # Normalize
            graph_proj = F.normalize(graph_proj, p=2, dim=1)
            embed_proj = F.normalize(embed_proj, p=2, dim=1)
            
            # Compute similarity matrix
            similarity_matrix = torch.matmul(graph_proj, embed_proj.t()) / 0.07  # temperature=0.07
            
            # Contrastive loss (bidirectional)
            labels = torch.arange(similarity_matrix.size(0), device=similarity_matrix.device)
            
            # Cross-entropy loss: graph->embed direction
            graph_to_embed_loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
            
            # Cross-entropy loss: embed->graph direction (symmetric)
            embed_to_graph_loss = nn.CrossEntropyLoss()(similarity_matrix.t(), labels)
            
            # Average contrastive loss
            lincs_embed_graph_contrastive_loss = (graph_to_embed_loss + embed_to_graph_loss) / 2
            
            representations['lincs_embed_graph_contrastive_loss'] = lincs_embed_graph_contrastive_loss
        
        # New: compute LINCS graph - molecular graph contrastive learning loss (cross-modal contrastive learning)
        if use_lincs_graph and use_mol and 'lincs_graph_proj' in representations and 'mol_proj' in representations:
            # Get the projected representations
            lincs_graph_proj = representations['lincs_graph_proj']  # [lincs_batch_size, emb_dim]
            mol_proj = representations['mol_proj']  # [mol_batch_size, emb_dim]
            
            # Ensure batch_size is consistent (take the minimum)
            # Note: the molecular graph batch size may be smaller than the LINCS graph batch size (because some molecules may fail to generate)
            actual_mol_batch_size = mol_proj.size(0)
            actual_lincs_batch_size = lincs_graph_proj.size(0)
            batch_size = min(actual_mol_batch_size, actual_lincs_batch_size)
            
            if batch_size >= 2:  # At least 2 samples are needed to compute the contrastive loss
                # Truncate to the same batch_size (only use the first batch_size samples)
                lincs_graph_proj = lincs_graph_proj[:batch_size]
                mol_proj = mol_proj[:batch_size]
                
                # Use the dedicated projection head for projection
                lincs_graph_mol_proj = self.lincs_graph_mol_contrast_proj(lincs_graph_proj)
                
                # Normalize
                lincs_graph_mol_proj = F.normalize(lincs_graph_mol_proj, p=2, dim=1)
                mol_proj_norm = F.normalize(mol_proj, p=2, dim=1)
                
                # Compute similarity matrix
                similarity_matrix = torch.matmul(lincs_graph_mol_proj, mol_proj_norm.t()) / 0.07
                
                # Contrastive loss (bidirectional)
                labels = torch.arange(batch_size, device=similarity_matrix.device)
                
                # Cross-entropy loss: lincs_graph->mol direction
                graph_to_mol_loss = nn.CrossEntropyLoss()(similarity_matrix, labels)
                
                # Cross-entropy loss: mol->lincs_graph direction (symmetric)
                mol_to_graph_loss = nn.CrossEntropyLoss()(similarity_matrix.t(), labels)
                
                # Average contrastive loss
                lincs_mol_contrastive_loss = (graph_to_mol_loss + mol_to_graph_loss) / 2
                
                representations['lincs_mol_contrastive_loss'] = lincs_mol_contrastive_loss

        # Add SMILES losses to the output
        if len(smiles_losses) > 0:
            representations['smiles_losses'] = smiles_losses

        return representations

    def _fuse_representations(self, representations, modalities=None):
        """
        Fuse multiple representations.
        """
        # Get all available representations
        repr_list = []

        if 'mol_graph_repr' in representations:
            repr_list.append(representations['mol_graph_repr'])

        if 'smiles_repr' in representations:
            repr_list.append(representations['smiles_repr'])

        if 'lincs_embed_repr' in representations:
            repr_list.append(representations['lincs_embed_repr'])

        if 'lincs_graph_repr' in representations:
            repr_list.append(representations['lincs_graph_repr'])

        if len(repr_list) == 0:
            raise ValueError("至少需要一种类型的输入表示")

        # If there is only one representation, return it directly
        if len(repr_list) == 1:
            return repr_list[0]

        # Check whether dimensions are consistent
        base_shape = repr_list[0].shape
        for i, repr_tensor in enumerate(repr_list[1:], 1):
            if repr_tensor.shape != base_shape:
                # If dimensions do not match, adapt them
                if len(base_shape) == 2 and len(repr_tensor.shape) == 2:
                    # Adjust the second dimension size
                    if repr_tensor.shape[1] != base_shape[1]:
                        repr_list[i] = torch.nn.functional.pad(repr_tensor, (0, base_shape[1] - repr_tensor.shape[1]),
                                                               "constant", 0)
                elif len(base_shape) == 3 and len(repr_tensor.shape) == 3:
                    if repr_tensor.shape[2] != base_shape[2]:
                        repr_list[i] = torch.nn.functional.pad(repr_tensor, (0, base_shape[2] - repr_tensor.shape[2]),
                                                               "constant", 0)

        # If there are two representations
        if len(repr_list) == 2:
            repr1, repr2 = repr_list[0], repr_list[1]
            # Ensure the two representations have the same shape
            if repr1.shape != repr2.shape:
                # If shapes do not match, unify to the same shape
                max_shape = tuple(max(s1, s2) for s1, s2 in zip(repr1.shape, repr2.shape))

                # Extend repr1 to the maximum shape
                pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(repr1.shape, max_shape))
                pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                repr1_padded = torch.nn.functional.pad(repr1, pad_tuple, "constant", 0)

                # Extend repr2 to the maximum shape
                pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(repr2.shape, max_shape))
                pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                repr2_padded = torch.nn.functional.pad(repr2, pad_tuple, "constant", 0)
            else:
                repr1_padded = repr1
                repr2_padded = repr2

            # Use co-attention and cross-fusion
            repr1_exp = repr1_padded.unsqueeze(1) if repr1_padded.dim() == 2 else repr1_padded
            repr2_exp = repr2_padded.unsqueeze(1) if repr2_padded.dim() == 2 else repr2_padded

            # Select the appropriate co-attention layer based on the modality type
            attended_repr1, attended_repr2 = self.mol_smiles_co_attention(repr1_exp, repr2_exp)

            # Remove the intermediate dimension
            attended_repr1 = attended_repr1.squeeze(1) if attended_repr1.dim() == 3 else attended_repr1
            attended_repr2 = attended_repr2.squeeze(1) if attended_repr2.dim() == 3 else attended_repr2

            # Check for NaN or Inf
            if torch.isnan(attended_repr1).any() or torch.isinf(attended_repr1).any():
                attended_repr1 = torch.nan_to_num(attended_repr1)
            if torch.isnan(attended_repr2).any() or torch.isinf(attended_repr2).any():
                attended_repr2 = torch.nan_to_num(attended_repr2)

            # Use Transformer cross-fusion
            cross_fused = self.transformer_cross_fusion(attended_repr1, attended_repr2)

            # Check the cross-fusion result
            if torch.isnan(cross_fused).any() or torch.isinf(cross_fused).any():
                cross_fused = torch.nan_to_num(cross_fused)

            # Use the predefined fusion layer
            fused = torch.cat([attended_repr1, attended_repr2], dim=-1)
            return self.fusion_layer_2mod(fused)

        # If there are three or four representations
        mol_graph_repr = representations.get('mol_graph_repr')
        smiles_repr = representations.get('smiles_repr')
        lincs_embed_repr = representations.get('lincs_embed_repr')
        lincs_graph_repr = representations.get('lincs_graph_repr')

        # Process according to the existing modalities
        final_reprs = []

        # Handle various modality combinations
        if mol_graph_repr is not None:
            mol_parts = []
            if smiles_repr is not None:
                # Ensure mol_graph_repr and smiles_repr dimensions are consistent
                if mol_graph_repr.shape != smiles_repr.shape:
                    # Unify to the same shape
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(mol_graph_repr.shape, smiles_repr.shape))

                    # Extend mol_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(mol_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    mol_graph_repr_padded = torch.nn.functional.pad(mol_graph_repr, pad_tuple, "constant", 0)

                    # Extend smiles_repr to the maximum shape
                    pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(smiles_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    smiles_repr_padded = torch.nn.functional.pad(smiles_repr, pad_tuple, "constant", 0)
                else:
                    mol_graph_repr_padded = mol_graph_repr
                    smiles_repr_padded = smiles_repr

                mol_repr_exp = mol_graph_repr_padded.unsqueeze(
                    1) if mol_graph_repr_padded.dim() == 2 else mol_graph_repr_padded
                smiles_repr_exp = smiles_repr_padded.unsqueeze(
                    1) if smiles_repr_padded.dim() == 2 else smiles_repr_padded
                mol_smiles_attended_mol, mol_smiles_attended_smiles = self.mol_smiles_co_attention(mol_repr_exp,
                                                                                                   smiles_repr_exp)
                mol_smiles_attended_mol = mol_smiles_attended_mol.squeeze(
                    1) if mol_smiles_attended_mol.dim() == 3 else mol_smiles_attended_mol
                mol_parts.append(mol_smiles_attended_mol)

            if lincs_embed_repr is not None:
                # Ensure mol_graph_repr and lincs_embed_repr dimensions are consistent
                if mol_graph_repr.shape != lincs_embed_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(mol_graph_repr.shape, lincs_embed_repr.shape))

                    # Extend mol_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(mol_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    mol_graph_repr_padded = torch.nn.functional.pad(mol_graph_repr, pad_tuple, "constant", 0)

                    # Extend lincs_embed_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_embed_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_embed_repr_padded = torch.nn.functional.pad(lincs_embed_repr, pad_tuple, "constant", 0)
                else:
                    mol_graph_repr_padded = mol_graph_repr
                    lincs_embed_repr_padded = lincs_embed_repr

                mol_repr_exp = mol_graph_repr_padded.unsqueeze(
                    1) if mol_graph_repr_padded.dim() == 2 else mol_graph_repr_padded
                lincs_embed_repr_exp = lincs_embed_repr_padded.unsqueeze(
                    1) if lincs_embed_repr_padded.dim() == 2 else lincs_embed_repr_padded
                mol_lincs_embed_attended_mol, mol_lincs_embed_attended_embed = self.mol_lincs_embed_co_attention(
                    mol_repr_exp, lincs_embed_repr_exp)
                mol_lincs_embed_attended_mol = mol_lincs_embed_attended_mol.squeeze(
                    1) if mol_lincs_embed_attended_mol.dim() == 3 else mol_lincs_embed_attended_mol
                mol_parts.append(mol_lincs_embed_attended_mol)

            if lincs_graph_repr is not None:
                # Ensure mol_graph_repr and lincs_graph_repr dimensions are consistent
                if mol_graph_repr.shape != lincs_graph_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(mol_graph_repr.shape, lincs_graph_repr.shape))

                    # Extend mol_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(mol_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    mol_graph_repr_padded = torch.nn.functional.pad(mol_graph_repr, pad_tuple, "constant", 0)

                    # Extend lincs_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_graph_repr_padded = torch.nn.functional.pad(lincs_graph_repr, pad_tuple, "constant", 0)
                else:
                    mol_graph_repr_padded = mol_graph_repr
                    lincs_graph_repr_padded = lincs_graph_repr

                mol_repr_exp = mol_graph_repr_padded.unsqueeze(
                    1) if mol_graph_repr_padded.dim() == 2 else mol_graph_repr_padded
                lincs_graph_repr_exp = lincs_graph_repr_padded.unsqueeze(
                    1) if lincs_graph_repr_padded.dim() == 2 else lincs_graph_repr_padded
                mol_lincs_graph_attended_mol, mol_lincs_graph_attended_graph = self.mol_lincs_graph_co_attention(
                    mol_repr_exp, lincs_graph_repr_exp)
                mol_lincs_graph_attended_mol = mol_lincs_graph_attended_mol.squeeze(
                    1) if mol_lincs_graph_attended_mol.dim() == 3 else mol_lincs_graph_attended_mol
                mol_parts.append(mol_lincs_graph_attended_mol)

            if len(mol_parts) > 0:
                combined_mol = torch.stack(mol_parts).mean(dim=0) if len(mol_parts) > 1 else mol_parts[0]
            else:
                combined_mol = mol_graph_repr
            final_reprs.append(combined_mol)

        if smiles_repr is not None:
            smiles_parts = []
            if mol_graph_repr is not None:
                # Ensure smiles_repr and mol_graph_repr dimensions are consistent
                if smiles_repr.shape != mol_graph_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(smiles_repr.shape, mol_graph_repr.shape))

                    # Extend smiles_repr to the maximum shape
                    pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(smiles_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    smiles_repr_padded = torch.nn.functional.pad(smiles_repr, pad_tuple, "constant", 0)

                    # Extend mol_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(mol_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    mol_graph_repr_padded = torch.nn.functional.pad(mol_graph_repr, pad_tuple, "constant", 0)
                else:
                    smiles_repr_padded = smiles_repr
                    mol_graph_repr_padded = mol_graph_repr

                mol_repr_exp = mol_graph_repr_padded.unsqueeze(
                    1) if mol_graph_repr_padded.dim() == 2 else mol_graph_repr_padded
                smiles_repr_exp = smiles_repr_padded.unsqueeze(
                    1) if smiles_repr_padded.dim() == 2 else smiles_repr_padded
                mol_smiles_attended_mol, mol_smiles_attended_smiles = self.mol_smiles_co_attention(mol_repr_exp,
                                                                                                   smiles_repr_exp)
                mol_smiles_attended_smiles = mol_smiles_attended_smiles.squeeze(
                    1) if mol_smiles_attended_smiles.dim() == 3 else mol_smiles_attended_smiles
                smiles_parts.append(mol_smiles_attended_smiles)

            if lincs_embed_repr is not None:
                # Ensure smiles_repr and lincs_embed_repr dimensions are consistent
                if smiles_repr.shape != lincs_embed_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(smiles_repr.shape, lincs_embed_repr.shape))

                    # Extend smiles_repr to the maximum shape
                    pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(smiles_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    smiles_repr_padded = torch.nn.functional.pad(smiles_repr, pad_tuple, "constant", 0)

                    # Extend lincs_embed_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_embed_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_embed_repr_padded = torch.nn.functional.pad(lincs_embed_repr, pad_tuple, "constant", 0)
                else:
                    smiles_repr_padded = smiles_repr
                    lincs_embed_repr_padded = lincs_embed_repr

                lincs_embed_repr_exp = lincs_embed_repr_padded.unsqueeze(
                    1) if lincs_embed_repr_padded.dim() == 2 else lincs_embed_repr_padded
                smiles_repr_exp = smiles_repr_padded.unsqueeze(
                    1) if smiles_repr_padded.dim() == 2 else smiles_repr_padded
                smiles_lincs_embed_attended_smiles, smiles_lincs_embed_attended_embed = self.smiles_lincs_embed_co_attention(
                    smiles_repr_exp, lincs_embed_repr_exp)
                smiles_lincs_embed_attended_smiles = smiles_lincs_embed_attended_smiles.squeeze(
                    1) if smiles_lincs_embed_attended_smiles.dim() == 3 else smiles_lincs_embed_attended_smiles
                smiles_parts.append(smiles_lincs_embed_attended_smiles)

            if lincs_graph_repr is not None:
                # Ensure smiles_repr and lincs_graph_repr dimensions are consistent
                if smiles_repr.shape != lincs_graph_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(smiles_repr.shape, lincs_graph_repr.shape))

                    # Extend smiles_repr to the maximum shape
                    pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(smiles_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    smiles_repr_padded = torch.nn.functional.pad(smiles_repr, pad_tuple, "constant", 0)

                    # Extend lincs_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_graph_repr_padded = torch.nn.functional.pad(lincs_graph_repr, pad_tuple, "constant", 0)
                else:
                    smiles_repr_padded = smiles_repr
                    lincs_graph_repr_padded = lincs_graph_repr

                lincs_graph_repr_exp = lincs_graph_repr_padded.unsqueeze(
                    1) if lincs_graph_repr_padded.dim() == 2 else lincs_graph_repr_padded
                smiles_repr_exp = smiles_repr_padded.unsqueeze(
                    1) if smiles_repr_padded.dim() == 2 else smiles_repr_padded
                smiles_lincs_graph_attended_smiles, smiles_lincs_graph_attended_graph = self.smiles_lincs_graph_co_attention(
                    smiles_repr_exp, lincs_graph_repr_exp)
                smiles_lincs_graph_attended_smiles = smiles_lincs_graph_attended_smiles.squeeze(
                    1) if smiles_lincs_graph_attended_smiles.dim() == 3 else smiles_lincs_graph_attended_smiles
                smiles_parts.append(smiles_lincs_graph_attended_smiles)

            if len(smiles_parts) > 0:
                combined_smiles = torch.stack(smiles_parts).mean(dim=0) if len(smiles_parts) > 1 else smiles_parts[0]
            else:
                combined_smiles = smiles_repr
            final_reprs.append(combined_smiles)

        if lincs_embed_repr is not None:
            lincs_embed_parts = []
            if mol_graph_repr is not None:
                # Ensure lincs_embed_repr and mol_graph_repr dimensions are consistent
                if lincs_embed_repr.shape != mol_graph_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(lincs_embed_repr.shape, mol_graph_repr.shape))

                    # Extend lincs_embed_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_embed_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_embed_repr_padded = torch.nn.functional.pad(lincs_embed_repr, pad_tuple, "constant", 0)

                    # Extend mol_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(mol_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    mol_graph_repr_padded = torch.nn.functional.pad(mol_graph_repr, pad_tuple, "constant", 0)
                else:
                    lincs_embed_repr_padded = lincs_embed_repr
                    mol_graph_repr_padded = mol_graph_repr

                mol_repr_exp = mol_graph_repr_padded.unsqueeze(
                    1) if mol_graph_repr_padded.dim() == 2 else mol_graph_repr_padded
                lincs_embed_repr_exp = lincs_embed_repr_padded.unsqueeze(
                    1) if lincs_embed_repr_padded.dim() == 2 else lincs_embed_repr_padded
                mol_lincs_embed_attended_mol, mol_lincs_embed_attended_embed = self.mol_lincs_embed_co_attention(
                    mol_repr_exp, lincs_embed_repr_exp)
                mol_lincs_embed_attended_embed = mol_lincs_embed_attended_embed.squeeze(
                    1) if mol_lincs_embed_attended_embed.dim() == 3 else mol_lincs_embed_attended_embed
                lincs_embed_parts.append(mol_lincs_embed_attended_embed)

            if smiles_repr is not None:
                # Ensure lincs_embed_repr and smiles_repr dimensions are consistent
                if lincs_embed_repr.shape != smiles_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(lincs_embed_repr.shape, smiles_repr.shape))

                    # Extend lincs_embed_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_embed_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_embed_repr_padded = torch.nn.functional.pad(lincs_embed_repr, pad_tuple, "constant", 0)

                    # Extend smiles_repr to the maximum shape
                    pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(smiles_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    smiles_repr_padded = torch.nn.functional.pad(smiles_repr, pad_tuple, "constant", 0)
                else:
                    lincs_embed_repr_padded = lincs_embed_repr
                    smiles_repr_padded = smiles_repr

                smiles_repr_exp = smiles_repr_padded.unsqueeze(
                    1) if smiles_repr_padded.dim() == 2 else smiles_repr_padded
                lincs_embed_repr_exp = lincs_embed_repr_padded.unsqueeze(
                    1) if lincs_embed_repr_padded.dim() == 2 else lincs_embed_repr_padded
                smiles_lincs_embed_attended_smiles, smiles_lincs_embed_attended_embed = self.smiles_lincs_embed_co_attention(
                    smiles_repr_exp, lincs_embed_repr_exp)
                smiles_lincs_embed_attended_embed = smiles_lincs_embed_attended_embed.squeeze(
                    1) if smiles_lincs_embed_attended_embed.dim() == 3 else smiles_lincs_embed_attended_embed
                lincs_embed_parts.append(smiles_lincs_embed_attended_embed)

            if lincs_graph_repr is not None:
                # Ensure lincs_embed_repr and lincs_graph_repr dimensions are consistent
                if lincs_embed_repr.shape != lincs_graph_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(lincs_embed_repr.shape, lincs_graph_repr.shape))

                    # Extend lincs_embed_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_embed_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_embed_repr_padded = torch.nn.functional.pad(lincs_embed_repr, pad_tuple, "constant", 0)

                    # Extend lincs_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_graph_repr_padded = torch.nn.functional.pad(lincs_graph_repr, pad_tuple, "constant", 0)
                else:
                    lincs_embed_repr_padded = lincs_embed_repr
                    lincs_graph_repr_padded = lincs_graph_repr

                lincs_graph_repr_exp = lincs_graph_repr_padded.unsqueeze(
                    1) if lincs_graph_repr_padded.dim() == 2 else lincs_graph_repr_padded
                lincs_embed_repr_exp = lincs_embed_repr_padded.unsqueeze(
                    1) if lincs_embed_repr_padded.dim() == 2 else lincs_embed_repr_padded
                lincs_embed_graph_attended_embed, lincs_embed_graph_attended_graph = self.lincs_embed_graph_co_attention(
                    lincs_embed_repr_exp, lincs_graph_repr_exp)
                lincs_embed_graph_attended_embed = lincs_embed_graph_attended_embed.squeeze(
                    1) if lincs_embed_graph_attended_embed.dim() == 3 else lincs_embed_graph_attended_embed
                lincs_embed_parts.append(lincs_embed_graph_attended_embed)

            if len(lincs_embed_parts) > 0:
                combined_lincs_embed = torch.stack(lincs_embed_parts).mean(dim=0) if len(lincs_embed_parts) > 1 else \
                    lincs_embed_parts[0]
            else:
                combined_lincs_embed = lincs_embed_repr
            final_reprs.append(combined_lincs_embed)

        if lincs_graph_repr is not None:
            lincs_graph_parts = []
            if mol_graph_repr is not None:
                # Ensure lincs_graph_repr and mol_graph_repr dimensions are consistent
                if lincs_graph_repr.shape != mol_graph_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(lincs_graph_repr.shape, mol_graph_repr.shape))

                    # Extend lincs_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_graph_repr_padded = torch.nn.functional.pad(lincs_graph_repr, pad_tuple, "constant", 0)

                    # Extend mol_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(mol_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    mol_graph_repr_padded = torch.nn.functional.pad(mol_graph_repr, pad_tuple, "constant", 0)
                else:
                    lincs_graph_repr_padded = lincs_graph_repr
                    mol_graph_repr_padded = mol_graph_repr

                mol_repr_exp = mol_graph_repr_padded.unsqueeze(
                    1) if mol_graph_repr_padded.dim() == 2 else mol_graph_repr_padded
                lincs_graph_repr_exp = lincs_graph_repr_padded.unsqueeze(
                    1) if lincs_graph_repr_padded.dim() == 2 else lincs_graph_repr_padded
                mol_lincs_graph_attended_mol, mol_lincs_graph_attended_graph = self.mol_lincs_graph_co_attention(
                    mol_repr_exp, lincs_graph_repr_exp)
                mol_lincs_graph_attended_graph = mol_lincs_graph_attended_graph.squeeze(
                    1) if mol_lincs_graph_attended_graph.dim() == 3 else mol_lincs_graph_attended_graph
                lincs_graph_parts.append(mol_lincs_graph_attended_graph)

            if smiles_repr is not None:
                # Ensure lincs_graph_repr and smiles_repr dimensions are consistent
                if lincs_graph_repr.shape != smiles_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(lincs_graph_repr.shape, smiles_repr.shape))

                    # Extend lincs_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_graph_repr_padded = torch.nn.functional.pad(lincs_graph_repr, pad_tuple, "constant", 0)

                    # Extend smiles_repr to the maximum shape
                    pad_tuple = tuple(0 if s == max_s else max_s - s for s, max_s in zip(smiles_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    smiles_repr_padded = torch.nn.functional.pad(smiles_repr, pad_tuple, "constant", 0)
                else:
                    lincs_graph_repr_padded = lincs_graph_repr
                    smiles_repr_padded = smiles_repr

                smiles_repr_exp = smiles_repr_padded.unsqueeze(
                    1) if smiles_repr_padded.dim() == 2 else smiles_repr_padded
                lincs_graph_repr_exp = lincs_graph_repr_padded.unsqueeze(
                    1) if lincs_graph_repr_padded.dim() == 2 else lincs_graph_repr_padded
                smiles_lincs_graph_attended_smiles, smiles_lincs_graph_attended_graph = self.smiles_lincs_graph_co_attention(
                    smiles_repr_exp, lincs_graph_repr_exp)
                smiles_lincs_graph_attended_graph = smiles_lincs_graph_attended_graph.squeeze(
                    1) if smiles_lincs_graph_attended_graph.dim() == 3 else smiles_lincs_graph_attended_graph
                lincs_graph_parts.append(smiles_lincs_graph_attended_graph)

            if lincs_embed_repr is not None:
                # Ensure lincs_graph_repr and lincs_embed_repr dimensions are consistent
                if lincs_graph_repr.shape != lincs_embed_repr.shape:
                    max_shape = tuple(max(s1, s2) for s1, s2 in zip(lincs_graph_repr.shape, lincs_embed_repr.shape))

                    # Extend lincs_graph_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_graph_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_graph_repr_padded = torch.nn.functional.pad(lincs_graph_repr, pad_tuple, "constant", 0)

                    # Extend lincs_embed_repr to the maximum shape
                    pad_tuple = tuple(
                        0 if s == max_s else max_s - s for s, max_s in zip(lincs_embed_repr.shape, max_shape))
                    pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                    lincs_embed_repr_padded = torch.nn.functional.pad(lincs_embed_repr, pad_tuple, "constant", 0)
                else:
                    lincs_graph_repr_padded = lincs_graph_repr
                    lincs_embed_repr_padded = lincs_embed_repr

                lincs_graph_repr_exp = lincs_graph_repr_padded.unsqueeze(
                    1) if lincs_graph_repr_padded.dim() == 2 else lincs_graph_repr_padded
                lincs_embed_repr_exp = lincs_embed_repr_padded.unsqueeze(
                    1) if lincs_embed_repr_padded.dim() == 2 else lincs_embed_repr_padded
                lincs_embed_graph_attended_embed, lincs_embed_graph_attended_graph = self.lincs_embed_graph_co_attention(
                    lincs_embed_repr_exp, lincs_graph_repr_exp)
                lincs_embed_graph_attended_graph = lincs_embed_graph_attended_graph.squeeze(
                    1) if lincs_embed_graph_attended_graph.dim() == 3 else lincs_embed_graph_attended_graph
                lincs_graph_parts.append(lincs_embed_graph_attended_graph)

            if len(lincs_graph_parts) > 0:
                combined_lincs_graph = torch.stack(lincs_graph_parts).mean(dim=0) if len(lincs_graph_parts) > 1 else \
                    lincs_graph_parts[0]
            else:
                combined_lincs_graph = lincs_graph_repr
            final_reprs.append(combined_lincs_graph)

        # Check whether the final representations have NaN or Inf
        for i, tensor in enumerate(final_reprs):
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                final_reprs[i] = torch.nan_to_num(tensor)

        # Concatenate all representations and fuse
        if len(final_reprs) == 1:
            return final_reprs[0]
        else:
            # Ensure all tensors in final_reprs have the same shape
            if len(set(t.shape for t in final_reprs)) > 1:  # If shapes differ
                max_shape = tuple(max(s1, s2) for s1, s2 in zip(final_reprs[0].shape, final_reprs[1].shape))
                for i in range(len(final_reprs)):
                    if final_reprs[i].shape != max_shape:
                        pad_tuple = tuple(
                            0 if s == max_s else max_s - s for s, max_s in zip(final_reprs[i].shape, max_shape))
                        pad_tuple = tuple(item for pair in zip((0,) * len(pad_tuple), pad_tuple) for item in pair)[::-1]
                        final_reprs[i] = torch.nn.functional.pad(final_reprs[i], pad_tuple, "constant", 0)

            combined_repr = torch.cat(final_reprs, dim=-1)

            # Use the predefined fusion layer (selected by modality count)
            n_mods = len(final_reprs)
            if n_mods <= 2:
                return self.fusion_layer_2mod(combined_repr)
            elif n_mods == 3:
                return self.fusion_layer_3mod(combined_repr)
            else:
                return self.fusion_layer_4mod(combined_repr)