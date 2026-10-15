"""
Multimodal pretraining entry point.

Combines a molecular graph neural network, SMILES text, LINCS embeddings and a
LINCS protein graph. Uses crossfusion to fuse the four modality representations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from toxfm.models.pretrain import MultimodalPretrainModel, DynamicLossBalancer
from toxfm.models.smiles import (
    SMILESVocabulary, tokenize_smiles, augment_smiles, mask_tokens,
)
from toxfm.data.dataset import MultimodalDataset
from toxfm.data.protein_graph import load_protein_graph_data
from toxfm.data.lincs import mask_lincs_features


def train_multimodal_pretrain(csv_file_path, output_model_path, epochs=20, batch_size=32,
                              num_layer=2, emb_dim=256,
                              JK="last", drop_ratio=0.1, gnn_type="gin", nhead=8,
                              max_len=128, mask_prob=0.15, learning_rate=1e-4, device='cpu',
                              modalities=['smiles', 'mol', 'lincs_embed', 'lincs_graph'],
                              gene_list_file=None):
    """
    Main function for multimodal pretraining, with modality control.

    Args:
        modalities: list of modalities to use, e.g.
                   ['smiles', 'mol'] - only SMILES and molecular graph
                   ['mol', 'lincs_embed', 'lincs_graph'] - molecular graph plus LINCS embedding and graph
                   ['smiles', 'mol', 'lincs_embed', 'lincs_graph'] - all four modalities
        gene_list_file: path to the LINCS gene list file (used to build the protein graph)
    """
    print(f"开始多模态预训练，使用模态: {modalities}")

    # Read CSV data
    print("1. 读取CSV数据...")
    df = pd.read_csv(csv_file_path)

    # Filter out invalid SMILES
    original_count = len(df)
    df = df[df['SMILES'].notna() & (df['SMILES'].str.strip() != '')]
    filtered_count = original_count - len(df)
    if filtered_count > 0:
        print(f"过滤掉 {filtered_count} 个无效SMILES样本")

    # Extract SMILES and gene expression data
    smiles_list = df['SMILES'].tolist()
    gene_cols = [col for col in df.columns if col not in ['pert_id', 'SMILES']]
    gene_expression_data = df[gene_cols].values.astype(np.float32)

    print(f"读取到 {len(smiles_list)} 个分子样本")
    print(f"基因表达数据维度: {gene_expression_data.shape}")

    # Build the vocabulary
    print("2. 创建词汇表...")
    vocab = SMILESVocabulary()
    print(f"词汇表大小: {vocab.vocab_size}")

    # Create the dataset
    print("3. 创建多模态数据集...")
    dataset = MultimodalDataset(csv_file_path, vocab, max_len=max_len, modalities=modalities)

    # Custom collate function to handle mixed data types
    def custom_collate_fn(batch):
        from torch_geometric.data import Batch as PyGBatch
        from torch.utils.data.dataloader import default_collate

        elem = batch[0]
        collated_batch = {}

        for key in elem.keys():
            if key == 'mol_data':
                # Handle PyG Data objects
                mol_data_list = [d[key] for d in batch if d[key] is not None]
                if mol_data_list:
                    collated_batch[key] = PyGBatch.from_data_list(mol_data_list)
                else:
                    collated_batch[key] = None
            else:
                # Handle other tensor types
                collated_batch[key] = default_collate([d[key] for d in batch])

        return collated_batch

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=custom_collate_fn)

    # Prepare LINCS graph data (if needed)
    lincs_edge_index = None
    lincs_edge_attr = None
    lincs_gene_list = []

    if ('lincs_graph' in modalities or 'lincs_embed' in modalities) and gene_list_file:
        print("4a. 加载LINCS蛋白质图数据...")
        lincs_edge_index, lincs_edge_attr, lincs_gene_list = load_protein_graph_data(gene_list_file)
        if lincs_edge_index is not None:
            lincs_edge_index = lincs_edge_index.to(device)
            if lincs_edge_attr is not None:
                lincs_edge_attr = lincs_edge_attr.to(device)
            print(f"Loaded LINCS graph with {len(lincs_gene_list)} nodes and {lincs_edge_index.size(1)} edges")
        else:
            print("警告: 未能加载LINCS图数据，将使用虚拟图结构")

    # Create the multimodal model
    print("4b. 创建多模态模型...")
    model = MultimodalPretrainModel(
        vocab_size=vocab.vocab_size,
        num_layer=num_layer,
        emb_dim=emb_dim,
        JK=JK,
        drop_ratio=drop_ratio,
        gnn_type=gnn_type,
        nhead=nhead,
        lincs_nodes=len(lincs_gene_list) if lincs_gene_list else gene_expression_data.shape[1]  # number of genes as LINCS node count
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Loss functions
    lincs_embed_criterion = nn.MSELoss()  # LINCS embedding reconstruction loss
    lincs_graph_criterion = nn.MSELoss()  # LINCS graph reconstruction loss

    # Contrastive learning temperature
    contrastive_temperature = 0.07

    # Dynamic loss balancer
    loss_balancer = DynamicLossBalancer(modalities, momentum=0.9)

    print("5. 开始训练...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            # Load data conditioned on modality
            smiles_tokens = batch.get('smiles_tokens', None)
            if smiles_tokens is not None:
                smiles_tokens = smiles_tokens.to(device)

            lincs_embed = batch.get('lincs_embed', None)
            if lincs_embed is not None:
                lincs_embed = lincs_embed.to(device)

            # Get molecular graph data (if present)
            mol_data = batch.get('mol_data', None)
            if mol_data is not None:
                mol_data = mol_data.to(device)

            # Generate augmented and masked SMILES versions (for MLM + contrastive learning)
            augmented_smiles_tokens = None
            masked_smiles_tokens = None
            mlm_labels = None

            if 'smiles' in modalities and smiles_tokens is not None:
                # 1. Generate augmented SMILES (for contrastive learning)
                import warnings
                augmented_smiles_list = []
                for i in range(smiles_tokens.size(0)):
                    # Decode back to a SMILES string
                    token_ids = smiles_tokens[i].cpu().tolist()
                    smiles_str = ''.join([
                        vocab.idx_to_char[int(idx)]
                        for idx in token_ids
                        if int(idx) not in [vocab.pad_idx, vocab.cls_idx, vocab.sep_idx, vocab.mask_idx, vocab.unk_idx]
                    ])

                    # Data augmentation (suppress RDKit warnings)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        augmented_smiles = augment_smiles(smiles_str, augment_type='random')

                    # Re-encode
                    augmented_ids = tokenize_smiles(augmented_smiles, vocab, max_len=max_len)
                    augmented_smiles_list.append(augmented_ids)

                augmented_smiles_tokens = torch.tensor(augmented_smiles_list, dtype=torch.long).to(device)

                # 2. Generate masked SMILES (for MLM)
                masked_input, labels = mask_tokens(smiles_tokens.clone(), vocab, mask_prob=mask_prob)
                masked_smiles_tokens = masked_input
                mlm_labels = labels

            # Forward pass according to the specified modalities
            input_kwargs = {}
            if 'smiles' in modalities:
                input_kwargs['smiles_tokens'] = smiles_tokens
                input_kwargs['augmented_smiles_tokens'] = augmented_smiles_tokens
                input_kwargs['masked_smiles_tokens'] = masked_smiles_tokens
                input_kwargs['mlm_labels'] = mlm_labels
            if 'mol' in modalities and mol_data is not None:
                input_kwargs['mol_data'] = mol_data
                # For molecular graph data, pass the batch information
                input_kwargs['batch'] = mol_data.batch if hasattr(mol_data, 'batch') else None
            if 'lincs_embed' in modalities or 'lincs_graph' in modalities:
                input_kwargs['lincs_edge_index'] = lincs_edge_index
                input_kwargs['lincs_edge_attr'] = lincs_edge_attr
                # Use gene expression data as the LINCS embedding
                input_kwargs['lincs_embed'] = lincs_embed  # gene expression data as LINCS embedding

                # Generate masked data for LINCS graph pretraining
                if 'lincs_graph' in modalities:
                    lincs_masked_embed, _, lincs_mask = mask_lincs_features(lincs_embed, mask_rate=0.15)
                    input_kwargs['lincs_masked_embed'] = lincs_masked_embed
                    input_kwargs['lincs_mask'] = lincs_mask

            input_kwargs['modalities'] = modalities

            # Forward pass
            outputs = model(**input_kwargs)

            # Compute per-modality losses
            losses = {}
            metrics = {}

            # Get SMILES losses (if the SMILES modality is used)
            smiles_losses = outputs.get('smiles_losses', {}) if 'smiles' in modalities else {}

            # Molecular graph - SMILES contrastive learning loss (InfoNCE)
            if 'mol' in modalities and 'smiles' in modalities and \
               'mol_proj' in outputs and 'smiles_proj' in outputs:
                # Get projected representations
                mol_proj = outputs['mol_proj']  # [batch_size, emb_dim]
                smiles_proj = outputs['smiles_proj']  # [batch_size, emb_dim]

                # Make sure the batch sizes of the two projections match
                batch_size = min(mol_proj.size(0), smiles_proj.size(0))
                if batch_size < 2:
                    # Batch too small, skip the contrastive loss
                    contrastive_loss = torch.tensor(0.0, device=mol_proj.device)
                else:
                    # Truncate to the same batch size
                    mol_proj = mol_proj[:batch_size]
                    smiles_proj = smiles_proj[:batch_size]

                    # Normalize
                    mol_proj = nn.functional.normalize(mol_proj, p=2, dim=1)
                    smiles_proj = nn.functional.normalize(smiles_proj, p=2, dim=1)

                    # Compute the similarity matrix (with numerical stability)
                    similarity_matrix = torch.matmul(mol_proj, smiles_proj.t()) / contrastive_temperature
                    # Clamp the similarity matrix to prevent numerical overflow
                    similarity_matrix = torch.clamp(similarity_matrix, min=-50, max=50)

                    # InfoNCE loss: diagonal entries are positive pairs
                    labels = torch.arange(batch_size, device=mol_proj.device)

                    # Cross-entropy loss: molecular graph -> SMILES direction
                    mol_to_smiles_loss = nn.CrossEntropyLoss()(similarity_matrix, labels)

                    # Cross-entropy loss: SMILES -> molecular graph direction (symmetric)
                    smiles_to_mol_loss = nn.CrossEntropyLoss()(similarity_matrix.t(), labels)

                    # Average the contrastive loss
                    contrastive_loss = (mol_to_smiles_loss + smiles_to_mol_loss) / 2

                    # Set to 0 if NaN
                    if torch.isnan(contrastive_loss):
                        print("警告: 对比损失出现NaN，设为0")
                        contrastive_loss = torch.tensor(0.0, device=mol_proj.device)

                losses['contrastive_loss'] = contrastive_loss.item()
                metrics['contrastive_loss'] = contrastive_loss.item()

            if 'smiles' in modalities and len(smiles_losses) > 0:
                # SMILES MLM loss (from SMILESContrastivePretrain)
                if 'mlm' in smiles_losses:
                    losses['smiles_mlm_loss'] = smiles_losses['mlm'].item()
                    metrics['smiles_mlm_loss'] = smiles_losses['mlm'].item()

                if 'total' in smiles_losses:
                    losses['smiles_total_loss'] = smiles_losses['total'].item()
                    metrics['smiles_total_loss'] = smiles_losses['total'].item()

            if 'lincs_embed' in modalities and 'lincs_embed_recon' in outputs:
                # LINCS embedding reconstruction loss
                lincs_embed_loss = lincs_embed_criterion(outputs['lincs_embed_recon'], lincs_embed)
                losses['lincs_embed_loss'] = lincs_embed_loss.item()
                metrics['lincs_embed_loss'] = lincs_embed_loss.item()

            if 'lincs_graph' in modalities and 'lincs_graph_recon_loss' in outputs:
                # LINCS graph mask reconstruction loss (enhanced pretraining: node-level + graph-level)
                lincs_graph_recon_loss = outputs['lincs_graph_recon_loss']
                losses['lincs_graph_recon_loss'] = lincs_graph_recon_loss.item()
                metrics['lincs_graph_recon_loss'] = lincs_graph_recon_loss.item()

                # Record sub-losses
                if 'lincs_graph_node_loss' in outputs:
                    losses['lincs_graph_node_loss'] = outputs['lincs_graph_node_loss'].item()
                    metrics['lincs_graph_node_loss'] = outputs['lincs_graph_node_loss'].item()
                if 'lincs_graph_global_loss' in outputs:
                    losses['lincs_graph_global_loss'] = outputs['lincs_graph_global_loss'].item()
                    metrics['lincs_graph_global_loss'] = outputs['lincs_graph_global_loss'].item()

                # Record the edge prediction loss (new)
                if 'lincs_graph_edge_pred_loss' in outputs:
                    losses['lincs_graph_edge_pred_loss'] = outputs['lincs_graph_edge_pred_loss'].item()
                    metrics['lincs_graph_edge_pred_loss'] = outputs['lincs_graph_edge_pred_loss'].item()

            # LINCS graph - embedding contrastive learning loss (new)
            if 'lincs_graph' in modalities and 'lincs_embed' in modalities and 'lincs_embed_graph_contrastive_loss' in outputs:
                lincs_embed_graph_contrastive_loss = outputs['lincs_embed_graph_contrastive_loss']
                losses['lincs_embed_graph_contrastive_loss'] = lincs_embed_graph_contrastive_loss.item()
                metrics['lincs_embed_graph_contrastive_loss'] = lincs_embed_graph_contrastive_loss.item()

            # LINCS graph - molecular graph contrastive learning loss (new: cross-modality contrastive learning)
            if 'lincs_graph' in modalities and 'mol' in modalities and 'lincs_mol_contrastive_loss' in outputs:
                lincs_mol_contrastive_loss = outputs['lincs_mol_contrastive_loss']
                losses['lincs_mol_contrastive_loss'] = lincs_mol_contrastive_loss.item()
                metrics['lincs_mol_contrastive_loss'] = lincs_mol_contrastive_loss.item()

            # Update loss weights
            loss_balancer.update_weights(losses)
            current_weights = loss_balancer.get_weights()

            # Scale losses by weight - use the original loss tensors rather than scalar values
            scaled_losses = {}
            # Use a gradient-requiring tensor as the initial value
            total_batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Map loss tensors to modality names
            loss_tensor_map = {
                'contrastive_loss': contrastive_loss if 'contrastive_loss' in locals() else None,
                'smiles_mlm_loss': smiles_losses.get('mlm', None),
                'lincs_embed_loss': lincs_embed_loss if 'lincs_embed_loss' in locals() else None,
                'lincs_graph_recon_loss': lincs_graph_recon_loss if 'lincs_graph_recon_loss' in locals() else None,
                'lincs_embed_graph_contrastive_loss': lincs_embed_graph_contrastive_loss if 'lincs_embed_graph_contrastive_loss' in locals() else None,
                'lincs_mol_contrastive_loss': lincs_mol_contrastive_loss if 'lincs_mol_contrastive_loss' in locals() else None,
            }

            for mod, loss_val in losses.items():
                # Get the corresponding actual loss tensor
                if mod in loss_tensor_map and loss_tensor_map[mod] is not None:
                    loss_tensor = loss_tensor_map[mod]
                    # Check whether the single loss is NaN
                    if torch.isnan(loss_tensor) or torch.isinf(loss_tensor):
                        print(f"警告: {mod}损失出现NaN/Inf，设为0")
                        loss_tensor = torch.tensor(0.0, device=device)
                    scaled_losses[mod] = loss_tensor * current_weights.get(mod, 1.0)
                    total_batch_loss = total_batch_loss + scaled_losses[mod]
                elif mod == 'lincs_graph_edge_pred_loss':
                    # Edge prediction loss
                    edge_pred_loss = outputs.get('lincs_graph_edge_pred_loss', None)
                    if edge_pred_loss is not None:
                        if torch.isnan(edge_pred_loss) or torch.isinf(edge_pred_loss):
                            print(f"警告: lincs_graph_edge_pred_loss出现NaN/Inf，设为0")
                            edge_pred_loss = torch.tensor(0.0, device=device)
                        scaled_losses[mod] = edge_pred_loss * current_weights.get(mod, 1.0)
                        total_batch_loss = total_batch_loss + scaled_losses[mod]

            # Check whether total_batch_loss is NaN or Inf
            if torch.isnan(total_batch_loss) or torch.isinf(total_batch_loss):
                print(f"警告: total_batch_loss出现NaN/Inf，跳过此batch")
                print(f"  各损失值: {losses}")
                continue

            # Check whether there is actual gradient flow (scaled_losses not empty)
            if len(scaled_losses) == 0:
                print(f"警告: 没有任何损失被累加，跳过此batch")
                continue

            # Backpropagation
            optimizer.zero_grad()
            total_batch_loss.backward()  # total_batch_loss is now a tensor
            # Gradient clipping to prevent gradient explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_batch_loss.item()  # record the numeric value for display
            num_batches += 1

            # Display current weights and all modality losses
            weight_info = ", ".join([f"{k}: {v:.4f}" for k, v in current_weights.items()])
            # Build the detailed loss display
            postfix_dict = {'total_loss': f'{total_batch_loss.item():.4f}'}
            for k, v in metrics.items():
                postfix_dict[k] = f'{v:.4f}'
            progress_bar.set_postfix(postfix_dict)

        avg_total_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_total_loss:.4f}")

    # Save the model
    print(f"6. 保存模型到 {output_model_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': vocab.vocab_size,
            'num_layer': num_layer,
            'emb_dim': emb_dim,
            'JK': JK,
            'drop_ratio': drop_ratio,
            'gnn_type': gnn_type,
            'nhead': nhead,
            'max_len': max_len,
            'lincs_nodes': len(lincs_gene_list) if lincs_gene_list else gene_expression_data.shape[1]
        },
        'modalities_used': modalities
    }, output_model_path)
    print("多模态预训练完成!")


def predict_with_modalities(model, mol_data=None, smiles_tokens=None,
                            lincs_edge_index=None, lincs_edge_attr=None, lincs_embed=None,
                            modalities=None, device='cpu'):
    """
    Perform prediction using the specified modalities.

    Args:
        model: the trained model
        mol_data, smiles_tokens: input data
        lincs_edge_index, lincs_edge_attr, lincs_embed: LINCS graph data and embedding
        modalities: list of modalities to use
        device: device
    """
    model.eval()
    with torch.no_grad():
        input_kwargs = {
            'mol_data': mol_data,
            'smiles_tokens': smiles_tokens,
            'lincs_edge_index': lincs_edge_index,
            'lincs_edge_attr': lincs_edge_attr,
            'lincs_embed': lincs_embed,
            'modalities': modalities
        }
        return model(**input_kwargs)


if __name__ == "__main__":
    # Example usage
    csv_file = "/home/linfeng/toxfm/merged_lincs_combined.csv"  # CSV with molecule and gene expression data
    output_model_path = "./multimodal_four99_graphtrans_model.pth"

    # LINCS gene list file path
    gene_list_file = "/home/linfeng/toxfm/gene_list.txt"

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # Define the modality combination to use
    # Use all four modalities
    modalities = ['mol', 'smiles', 'lincs_embed', 'lincs_graph']

    # Run multimodal pretraining
    train_multimodal_pretrain(
        csv_file_path=csv_file,
        output_model_path=output_model_path,
        epochs=20,
        batch_size=64,
        num_layer=2,
        emb_dim=128,
        JK="last",
        drop_ratio=0.1,
        gnn_type="gin",
        nhead=4,
        max_len=128,
        mask_prob=0.15,
        learning_rate=1e-4,
        device=device,
        modalities=modalities,  # control the modalities used
        gene_list_file=gene_list_file  # LINCS gene list file
    )