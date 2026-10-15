"""
Multimodal fine-tuning entry point.

Supports arbitrary combinations of four modalities: molecular graph (mol),
SMILES text (smiles), LINCS embedding (lincs_embed) and LINCS graph
(lincs_graph).
"""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from tqdm import tqdm

from toxfm.models.finetune import MultimodalFinetuneModel
from toxfm.models.smiles import SMILESVocabulary
from toxfm.data.dataset import MultimodalFinetuneDataset
from toxfm.data.protein_graph import load_protein_graph_data


def compute_metrics(eval_pred):
    """Compute binary classification evaluation metrics."""
    predictions, labels = eval_pred

    # predictions are already softmax probabilities
    probabilities = predictions

    # Probability of the positive class (Label=1)
    positive_probs = probabilities[:, 1]

    # Predicted class from the probabilities
    predicted_labels = np.argmax(probabilities, axis=1)

    # Confusion matrix
    cm = confusion_matrix(
        labels,
        predicted_labels,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    # Sensitivity / Recall
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Precision
    pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1
    f1 = (
        2 * pre * sen / (pre + sen)
        if (pre + sen) > 0
        else 0.0
    )

    # Accuracy
    acc = (
        (tp + tn) / (tp + tn + fp + fn)
        if (tp + tn + fp + fn) > 0
        else 0.0
    )

    # ROC-AUC
    try:
        auc = roc_auc_score(
            labels,
            positive_probs
        )
    except ValueError:
        auc = 0.5

    # AUPR / Average precision
    try:
        aupr = average_precision_score(
            labels,
            positive_probs
        )
    except ValueError:
        aupr = 0.5

    return {
        'auc': auc,
        'aupr': aupr,
        'spe': spe,
        'sen': sen,
        'pre': pre,
        'f1': f1,
        'acc': acc
    }


def finetune_multimodal(csv_file_path, modalities, output_dir,
                       pretrained_model_path, gene_list_file=None, epochs=10, batch_size=16,
                       learning_rate=1e-4, cv_folds=5, device='cpu'):
    """
    Main function for multimodal fine-tuning.

    Args:
        csv_file_path: CSV file path containing SMILES and labels
        modalities: list of modalities to use, e.g. ['mol', 'smiles', 'lincs_embed', 'lincs_graph']
        output_dir: output directory
        pretrained_model_path: pretrained model path
        gene_list_file: LINCS gene list file path (required when using the lincs_graph modality)
        epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
        cv_folds: number of cross-validation folds
        device: device
    """
    print(f"开始多模态微调，使用模态: {modalities}")
    print(f"加载预训练模型: {pretrained_model_path}")

    # Read the CSV file to get gene expression columns
    df = pd.read_csv(csv_file_path)
    gene_expression_cols = [col for col in df.columns if col not in ['SMILES', 'Label', 'label', 'pert_id']]

    # Build the vocabulary
    print("1. 创建词汇表...")
    vocab = SMILESVocabulary()
    print(f"词汇表大小: {vocab.vocab_size}")

    # Prepare LINCS graph data (if needed)
    lincs_edge_index = None
    lincs_edge_attr = None
    lincs_gene_list = []

    if 'lincs_graph' in modalities and gene_list_file:
        print("2. 加载LINCS蛋白质图数据...")
        lincs_edge_index, lincs_edge_attr, lincs_gene_list = load_protein_graph_data(gene_list_file)
        if lincs_edge_index is not None:
            lincs_edge_index = lincs_edge_index.to(device)
            if lincs_edge_attr is not None:
                lincs_edge_attr = lincs_edge_attr.to(device)
            print(f"Loaded LINCS graph with {len(lincs_gene_list)} nodes and {lincs_edge_index.size(1)} edges")
        else:
            print("警告: 未能加载LINCS图数据，将使用虚拟图结构")

    # Get the number of classes
    if 'Label' in df.columns:
        num_classes = len(df['Label'].unique())
    elif 'label' in df.columns:
        num_classes = len(df['label'].unique())
    else:
        raise ValueError("CSV文件必须包含'Label'或'label'列")

    # Load the pretrained model
    print("3. 加载预训练模型...")
    checkpoint = torch.load(pretrained_model_path, map_location=device)

    # Create the model instance
    model = MultimodalFinetuneModel(
        vocab_size=vocab.vocab_size,
        num_layer=checkpoint['model_config']['num_layer'],
        emb_dim=checkpoint['model_config']['emb_dim'],
        JK=checkpoint['model_config']['JK'],
        drop_ratio=checkpoint['model_config']['drop_ratio'],
        gnn_type=checkpoint['model_config']['gnn_type'],
        nhead=checkpoint['model_config']['nhead'],
        lincs_nodes=checkpoint['model_config']['lincs_nodes'],
        num_classes=num_classes
    ).to(device)

    # Load pretrained weights (excluding the classification head)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and 'classification_head' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    print("预训练模型加载完成")

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Create a dataset object for the entire dataset
    full_dataset = MultimodalFinetuneDataset(
        csv_file_path=csv_file_path,
        vocab=vocab,
        modalities=modalities,
        gene_expression_cols=gene_expression_cols
    )

    # Get labels for cross-validation splitting
    all_labels = [full_dataset[i]['label'].item() for i in range(len(full_dataset))]

    # Store per-fold results
    cv_results = []

    # DataLoader collate function
    def custom_collate_fn(batch):
        from torch_geometric.data import Batch as PyGBatch
        from torch.utils.data.dataloader import default_collate

        elem = batch[0]
        collated_batch = {'label': []}

        # Initialize all possible keys
        for key in ['smiles_tokens', 'mol_data', 'lincs_embed']:
            if key in elem:
                collated_batch[key] = []

        for item in batch:
            collated_batch['label'].append(item['label'])
            for key in ['smiles_tokens', 'lincs_embed']:
                if key in item:
                    collated_batch[key].append(item[key])

            if 'mol_data' in item:
                collated_batch['mol_data'].append(item['mol_data'])

        # Handle labels
        collated_batch['label'] = torch.stack(collated_batch['label'])

        # Handle SMILES and LINCS embeddings
        for key in ['smiles_tokens', 'lincs_embed']:
            if key in collated_batch and collated_batch[key]:
                collated_batch[key] = torch.stack(collated_batch[key])

        # Handle molecular graph data
        if 'mol_data' in collated_batch and collated_batch['mol_data']:
            collated_batch['mol_data'] = PyGBatch.from_data_list(collated_batch['mol_data'])

        return collated_batch

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(full_dataset)), all_labels)):
        print(f"\nFold {fold + 1}/{cv_folds}")

        # Recreate the model and load pretrained weights for each fold
        model = MultimodalFinetuneModel(
            vocab_size=vocab.vocab_size,
            num_layer=checkpoint['model_config']['num_layer'],
            emb_dim=checkpoint['model_config']['emb_dim'],
            JK=checkpoint['model_config']['JK'],
            drop_ratio=checkpoint['model_config']['drop_ratio'],
            gnn_type=checkpoint['model_config']['gnn_type'],
            nhead=checkpoint['model_config']['nhead'],
            lincs_nodes=checkpoint['model_config']['lincs_nodes'],
            num_classes=num_classes
        ).to(device)

        # Load pretrained weights (excluding the classification head)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in model_dict and 'classification_head' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        # Create training and validation subsets
        train_subset = torch.utils.data.Subset(full_dataset, train_idx)
        val_subset = torch.utils.data.Subset(full_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

        # Reinitialize the classification head since we are fine-tuning
        model.classification_head = torch.nn.Sequential(
            torch.nn.Linear(model.emb_dim, model.emb_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(model.drop_ratio),
            torch.nn.Linear(model.emb_dim // 2, num_classes)
        ).to(device)

        # Optimizer - fine-tune the entire network with a small learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        print(f"  开始训练 Fold {fold + 1}...")
        model.train()

        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in progress_bar:
                # Get input data
                smiles_tokens = batch.get('smiles_tokens', None)
                mol_data = batch.get('mol_data', None)
                lincs_embed = batch.get('lincs_embed', None)
                labels = batch['label'].to(device)

                if smiles_tokens is not None:
                    smiles_tokens = smiles_tokens.to(device)
                if mol_data is not None:
                    mol_data = mol_data.to(device)
                if lincs_embed is not None:
                    lincs_embed = lincs_embed.to(device)

                # Forward pass
                logits = model(
                    mol_data=mol_data,
                    smiles_tokens=smiles_tokens,
                    lincs_edge_index=lincs_edge_index,
                    lincs_edge_attr=lincs_edge_attr,
                    lincs_embed=lincs_embed,
                    batch=mol_data.batch if mol_data is not None else None,
                    modalities=modalities
                )

                # Compute loss
                loss = criterion(logits, labels)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_loss = total_loss / num_batches
            print(f"  Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Evaluate on the validation set
        print(f"  开始评估 Fold {fold + 1}...")
        model.eval()
        all_predictions = []
        all_labels_eval = []

        with torch.no_grad():
            for batch in val_loader:
                smiles_tokens = batch.get('smiles_tokens', None)
                mol_data = batch.get('mol_data', None)
                lincs_embed = batch.get('lincs_embed', None)
                labels = batch['label'].to(device)

                if smiles_tokens is not None:
                    smiles_tokens = smiles_tokens.to(device)
                if mol_data is not None:
                    mol_data = mol_data.to(device)
                if lincs_embed is not None:
                    lincs_embed = lincs_embed.to(device)

                logits = model(
                    mol_data=mol_data,
                    smiles_tokens=smiles_tokens,
                    lincs_edge_index=lincs_edge_index,
                    lincs_edge_attr=lincs_edge_attr,
                    lincs_embed=lincs_embed,
                    batch=mol_data.batch if mol_data is not None else None,
                    modalities=modalities
                )

                # Get predicted probabilities
                probs = torch.softmax(logits, dim=1)
                all_predictions.extend(probs.cpu().numpy())
                all_labels_eval.extend(labels.cpu().numpy())

        # Compute evaluation metrics
        eval_pred = (np.array(all_predictions), np.array(all_labels_eval))
        fold_metrics = compute_metrics(eval_pred)
        cv_results.append(fold_metrics)

        print(f"  Fold {fold + 1} 结果: {fold_metrics}")

    # Compute mean and standard deviation of the cross-validation results
    metrics_keys = ['auc', 'aupr', 'spe', 'sen', 'pre', 'f1', 'acc']
    mean_results = {}
    std_results = {}

    for key in metrics_keys:
        values = [result[key] for result in cv_results]
        if values:
            mean_results[key] = np.mean(values)
            std_results[key] = np.std(values)

    # Print cross-validation results
    print(f"\n{cv_folds}折交叉验证结果:")
    for key in metrics_keys:
        if key in mean_results:
            print(f"{key.upper()}: {mean_results[key]:.4f} ± {std_results[key]:.4f}")

    # Retrain the model on the full dataset and save it
    print("\n使用完整数据集重新训练模型...")

    # Create the full data loader
    full_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

    # Reinitialize the classification head for the full training data
    model.classification_head = torch.nn.Sequential(
        torch.nn.Linear(model.emb_dim, model.emb_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Dropout(model.drop_ratio),
        torch.nn.Linear(model.emb_dim // 2, num_classes)
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the full model
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(full_loader, desc=f"Final Epoch {epoch + 1}/{epochs}")
        for batch in progress_bar:
            smiles_tokens = batch.get('smiles_tokens', None)
            mol_data = batch.get('mol_data', None)
            lincs_embed = batch.get('lincs_embed', None)
            labels = batch['label'].to(device)

            if smiles_tokens is not None:
                smiles_tokens = smiles_tokens.to(device)
            if mol_data is not None:
                mol_data = mol_data.to(device)
            if lincs_embed is not None:
                lincs_embed = lincs_embed.to(device)

            logits = model(
                mol_data=mol_data,
                smiles_tokens=smiles_tokens,
                lincs_edge_index=lincs_edge_index,
                lincs_edge_attr=lincs_edge_attr,
                lincs_embed=lincs_embed,
                batch=mol_data.batch if mol_data is not None else None,
                modalities=modalities
            )

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        print(f"Final Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

    # Save the final model
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, f"multimodal_finetuned_model_{'_'.join(modalities)}.pth")

    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': vocab.vocab_size,
            'num_layer': checkpoint['model_config']['num_layer'],
            'emb_dim': checkpoint['model_config']['emb_dim'],
            'JK': checkpoint['model_config']['JK'],
            'drop_ratio': checkpoint['model_config']['drop_ratio'],
            'gnn_type': checkpoint['model_config']['gnn_type'],
            'nhead': checkpoint['model_config']['nhead'],
            'lincs_nodes': checkpoint['model_config']['lincs_nodes'],
            'num_classes': num_classes
        },
        'modalities_used': modalities
    }, model_save_path)

    print(f"微调模型已保存到 {model_save_path}")

    return mean_results, std_results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='多模态模型微调 - 支持四种模态的不同组合')
    parser.add_argument('--csv_file', type=str, default='/home/linfeng/pretrain-gnns-2/merge_linc.csv',
                        help='包含SMILES和标签的CSV文件路径')
    parser.add_argument('--pretrained_model_path', type=str, default='/home/linfeng/toxfm/multimodal_pretrain_model.pth',
                        help='预训练模型路径')
    parser.add_argument('--modalities', type=str, nargs='+', default=['lincs_embed'],
                        choices=['mol', 'smiles', 'lincs_embed', 'lincs_graph'],
                        help='要使用的模态组合，可选: mol, smiles, lincs_embed, lincs_graph')
    parser.add_argument('--output_dir', type=str, default='./multimodal_finetuned',
                        help='微调后模型输出目录')
    parser.add_argument('--gene_list_file', type=str, default='/home/linfeng/pretrain-gnns-2/chem/gene_list.txt',
                        help='LINCS基因列表文件路径（当使用lincs_graph模态时必需）')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='批次大小')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='交叉验证折数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率')

    args = parser.parse_args()

    # Check file existence
    if not os.path.exists(args.csv_file):
        raise FileNotFoundError(f"CSV文件不存在: {args.csv_file}")

    if not os.path.exists(args.pretrained_model_path):
        raise FileNotFoundError(f"预训练模型不存在: {args.pretrained_model_path}")

    if 'lincs_graph' in args.modalities and args.gene_list_file is None:
        raise ValueError("当使用lincs_graph模态时，必须提供gene_list_file参数")

    if args.gene_list_file is not None and not os.path.exists(args.gene_list_file):
        raise FileNotFoundError(f"基因列表文件不存在: {args.gene_list_file}")

    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run fine-tuning
    mean_results, std_results = finetune_multimodal(
        csv_file_path=args.csv_file,
        modalities=args.modalities,
        output_dir=args.output_dir,
        pretrained_model_path=args.pretrained_model_path,
        gene_list_file=args.gene_list_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cv_folds=args.cv_folds,
        device=args.device,
        learning_rate=args.learning_rate
    )

    print("\n最终结果:")
    metrics_keys = ['auc', 'aupr', 'spe', 'sen', 'pre', 'f1', 'acc']
    for key in metrics_keys:
        if key in mean_results:
            print(f"{key.upper()}: {mean_results[key]:.4f} ± {std_results[key]:.4f}")


if __name__ == "__main__":
    main()