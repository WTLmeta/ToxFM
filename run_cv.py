"""
Cross-validation entry point using pre-split 5-fold datasets.

Based on the fine-tuning code; automatically discovers the fold data files.
"""
import os
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from toxfm.models.finetune import MultimodalFinetuneModel
from toxfm.models.smiles import SMILESVocabulary
from toxfm.data.dataset import MultimodalFinetuneDataset
from toxfm.data.protein_graph import load_protein_graph_data
from toxfm.data.folds import find_fold_files
from toxfm.utils.metrics import compute_metrics, FocalLoss
from torch_geometric.data import Batch as PyGBatch


def custom_collate_fn(batch, modalities):
    """Custom collate function."""
    from torch.utils.data.dataloader import default_collate

    elem = batch[0]
    collated_batch = {'label': []}

    # Only initialize the keys of the required modalities
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


def train_fold(fold_num, train_csv, val_csv, modalities, model, device,
               lincs_edge_index, lincs_edge_attr, epochs, batch_size, learning_rate, output_dir,
               loss_type="weighted_ce", loss_alpha=0.25, loss_gamma=2.0, toxicity_name="unknown"):
    """Train a single fold, using test.csv as the validation set.

    Args:
        loss_type: loss function type ("weighted_ce", "focal")
        loss_alpha: alpha parameter of Focal Loss
        loss_gamma: gamma parameter of Focal Loss
    """
    import time

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"Fold {fold_num + 1} 训练开始")
    print(f"训练集：{train_csv}")
    print(f"验证集 (test.csv): {val_csv}")
    print(f"使用模态：{modalities}")
    print(f"损失函数：{loss_type} (alpha={loss_alpha}, gamma={loss_gamma})")
    print(f"{'='*60}\n")

    # Create the vocabulary (use the predefined vocabulary; no need to build it from SMILES)
    vocab = SMILESVocabulary()

    # Create the datasets
    train_dataset = MultimodalFinetuneDataset(
        csv_file_path=train_csv,
        vocab=vocab,
        modalities=modalities,
        max_len=128
    )

    val_dataset = MultimodalFinetuneDataset(
        csv_file_path=val_csv,
        vocab=vocab,
        modalities=modalities,
        max_len=128
    )

    # Create the DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: custom_collate_fn(batch, modalities)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: custom_collate_fn(batch, modalities)
    )

    print(f"训练集样本数：{len(train_dataset)}")
    print(f"验证集样本数：{len(val_dataset)}")

    # Reinitialize the classification head
    model.classification_head = torch.nn.Sequential(
        torch.nn.Linear(model.emb_dim, model.emb_dim // 2),
        torch.nn.ReLU(),
        torch.nn.Dropout(model.drop_ratio),
        torch.nn.Linear(model.emb_dim // 2, model.num_classes)
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    # Use a weighted loss to handle class imbalance
    # Compute the positive/negative sample ratio in the training set
    train_labels = pd.read_csv(train_csv)['Label'].values
    pos_ratio = np.sum(train_labels == 1) / len(train_labels)
    neg_ratio = 1 - pos_ratio
    print(f"  训练集类别分布：正类={pos_ratio:.3f}, 负类={neg_ratio:.3f}")

    # Select the loss function according to loss_type
    if loss_type == "focal":
        criterion = FocalLoss(alpha=loss_alpha, gamma=loss_gamma)
        print(f"  使用 Focal Loss: alpha={loss_alpha}, gamma={loss_gamma}")
    else:  # weighted_ce
        # Create class weights for CrossEntropyLoss
        class_weights = torch.tensor([1.0 / neg_ratio, 1.0 / pos_ratio], dtype=torch.float32, device=device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print(f"  使用 Weighted CrossEntropy Loss")

    # Learning-rate warmup + cosine annealing scheduler
    # Warm up for 5 epochs, then use cosine annealing
    warmup_epochs = 5
    total_steps = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Warmup phase: linear increase
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing phase
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Best-model saving + early stopping
    best_auc = 0.0
    patience = 40  # stop if AUC does not improve for `patience` consecutive epochs
    patience_counter = 0
    best_model_path = os.path.join(output_dir, f'best_model_{toxicity_name}_fold_{fold_num}.pth')
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Fold {fold_num+1} | Epoch {epoch+1}/{epochs}")
        for batch in progress_bar:
            smiles_tokens = batch.get('smiles_tokens', None)
            mol_data = batch.get('mol_data', None)
            lincs_embed = batch.get('lincs_embed', None)
            labels = batch['label'].to(device)

            if smiles_tokens is not None:
                smiles_tokens = smiles_tokens.to(device)
            if mol_data is not None:
                mol_data = mol_data.to(device)
                if hasattr(mol_data, 'x') and mol_data.x is not None:
                    mol_data.x = mol_data.x.to(device)
                if hasattr(mol_data, 'edge_index') and mol_data.edge_index is not None:
                    mol_data.edge_index = mol_data.edge_index.to(device)
                if hasattr(mol_data, 'edge_attr') and mol_data.edge_attr is not None:
                    mol_data.edge_attr = mol_data.edge_attr.to(device)
                if hasattr(mol_data, 'batch') and mol_data.batch is not None:
                    mol_data.batch = mol_data.batch.to(device)
            if lincs_embed is not None:
                lincs_embed = lincs_embed.to(device)

            logits = model(
                mol_data=mol_data,
                smiles_tokens=smiles_tokens,
                lincs_edge_index=lincs_edge_index,
                lincs_edge_attr=lincs_edge_attr,
                lincs_embed=lincs_embed,
                batch=mol_data.batch if mol_data is not None else None,
                modalities=modalities,
                labels=labels
            )

            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Validate once per epoch
        model.eval()
        val_preds = []
        val_labels = []
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
                    if hasattr(mol_data, 'x') and mol_data.x is not None:
                        mol_data.x = mol_data.x.to(device)
                    if hasattr(mol_data, 'edge_index') and mol_data.edge_index is not None:
                        mol_data.edge_index = mol_data.edge_index.to(device)
                    if hasattr(mol_data, 'edge_attr') and mol_data.edge_attr is not None:
                        mol_data.edge_attr = mol_data.edge_attr.to(device)
                    if hasattr(mol_data, 'batch') and mol_data.batch is not None:
                        mol_data.batch = mol_data.batch.to(device)
                if lincs_embed is not None:
                    lincs_embed = lincs_embed.to(device)

                logits = model(
                    mol_data=mol_data,
                    smiles_tokens=smiles_tokens,
                    lincs_edge_index=lincs_edge_index,
                    lincs_edge_attr=lincs_edge_attr,
                    lincs_embed=lincs_embed,
                    batch=mol_data.batch if mol_data is not None else None,
                    modalities=modalities,
                    labels=labels
                )
                probs = torch.softmax(logits, dim=1)[:, 1]
                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_metrics = compute_metrics((np.array(val_preds), np.array(val_labels)))
        val_auc = val_metrics.get('auc', 0.0)
        print(f"  Epoch {epoch+1}/{epochs} Val AUC: {val_auc:.4f}")

        # Save the best model
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'vocab_size': model.vocab_size,
                    'num_layer': model.num_layer,
                    'emb_dim': model.emb_dim,
                    'JK': model.JK,
                    'drop_ratio': model.drop_ratio,
                    'gnn_type': model.gnn_type,
                    'nhead': model.nhead,
                    'lincs_nodes': model.lincs_nodes,
                },
                'modalities': modalities,
                'epoch': epoch + 1,
                'val_auc': val_auc,
                'fold': fold_num + 1,
            }, best_model_path)
            print(f"  ★ 保存最佳模型 (AUC={val_auc:.4f}) → {best_model_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"   Early Stopping: AUC 连续 {patience} 个 epoch 未提升，停止训练")
                break

        model.train()

    # Final validation
    print(f"\n  Fold {fold_num + 1} 最终验证（加载最佳模型）...")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Fold {fold_num+1} | Validation"):
            smiles_tokens = batch.get('smiles_tokens', None)
            mol_data = batch.get('mol_data', None)
            lincs_embed = batch.get('lincs_embed', None)
            labels = batch['label'].to(device)

            if smiles_tokens is not None:
                smiles_tokens = smiles_tokens.to(device)
            if mol_data is not None:
                mol_data = mol_data.to(device)
                # Ensure all attributes are on the correct device
                if hasattr(mol_data, 'x') and mol_data.x is not None:
                    mol_data.x = mol_data.x.to(device)
                if hasattr(mol_data, 'edge_index') and mol_data.edge_index is not None:
                    mol_data.edge_index = mol_data.edge_index.to(device)
                if hasattr(mol_data, 'edge_attr') and mol_data.edge_attr is not None:
                    mol_data.edge_attr = mol_data.edge_attr.to(device)
                if hasattr(mol_data, 'batch') and mol_data.batch is not None:
                    mol_data.batch = mol_data.batch.to(device)
            if lincs_embed is not None:
                lincs_embed = lincs_embed.to(device)

            logits = model(
                mol_data=mol_data,
                smiles_tokens=smiles_tokens,
                lincs_edge_index=lincs_edge_index,
                lincs_edge_attr=lincs_edge_attr,
                lincs_embed=lincs_embed,
                batch=mol_data.batch if mol_data is not None else None,
                modalities=modalities,
                labels=labels
            )

            # Save the positive-class probability
            probs = torch.softmax(logits, dim=1)[:, 1]  # positive-class probability only
            all_predictions.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    fold_metrics = compute_metrics((np.array(all_predictions), np.array(all_labels)))

    # Compute training time
    end_time = time.time()
    training_time = end_time - start_time
    fold_metrics['training_time'] = training_time
    fold_metrics['modalities'] = modalities

    print(f"\n  Fold {fold_num + 1} 结果:")
    for key, value in fold_metrics.items():
        if key != 'modalities':  # modality info displayed separately
            print(f"    {key.upper()}: {value:.4f}" if isinstance(value, float) else f"    {key.upper()}: {value}")

    return fold_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='使用已划分的 5 折数据集进行交叉验证')
    parser.add_argument('--base_dir', type=str, default='/home/linfeng/toxfm/dataset_splits/acute_no_overlap_42',
                        help='5 折数据集目录路径')
    parser.add_argument('--pretrained_model_path', type=str, default='/home/linfeng/toxfm/multimodal_pretrain_model.pth',
                        help='预训练模型路径')
    parser.add_argument('--gene_list_file', type=str, default='/home/linfeng/toxfm/gene_list.txt',
                        help='LINCS 基因列表文件路径')
    parser.add_argument('--output_dir', type=str, default='./multimodal_finetune_5fold_results',
                        help='输出目录')
    parser.add_argument('--modalities', type=str, nargs='+', default=['mol', 'smiles', 'lincs_embed', 'lincs_graph'],
                        choices=['mol', 'smiles', 'lincs_embed', 'lincs_graph'],
                        help='要使用的模态组合')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='学习率')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='设备类型')
    parser.add_argument('--loss_type', type=str, default='weighted_ce', choices=['weighted_ce', 'focal'],
                        help='损失函数类型: weighted_ce (加权交叉熵) 或 focal (Focal Loss)')
    parser.add_argument('--loss_alpha', type=float, default=0.25,
                        help='Focal Loss的alpha参数 (类别权重平衡)')
    parser.add_argument('--loss_gamma', type=float, default=2.0,
                        help='Focal Loss的gamma参数 (难易样本权重)')

    args = parser.parse_args()

    print("=" * 60)
    print("使用已划分的 5 折数据集进行交叉验证")
    print("=" * 60)

    # Configure parameters
    base_dir = args.base_dir
    pretrained_model_path = args.pretrained_model_path
    gene_list_file = args.gene_list_file
    output_dir = args.output_dir

    # Extract the toxicity name from the dataset directory name (e.g. heart_no_overlap_42 -> heart)
    base_dir_name = os.path.basename(base_dir)
    if 'acute' in base_dir_name.lower():
        toxicity_name = 'acute'
    elif 'heart' in base_dir_name.lower():
        toxicity_name = 'heart'
    else:
        # Default to the first part of the directory name
        toxicity_name = base_dir_name.split('_')[0]

    # Modality combination
    modalities = args.modalities

    # Training parameters
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    device = torch.device(args.device)

    print(f"\n数据集目录：{base_dir}")
    print(f"毒性名称：{toxicity_name}")
    print(f"预训练模型：{pretrained_model_path}")
    print(f"基因列表：{gene_list_file}")
    print(f"输出目录：{output_dir}")
    print(f"\n使用的模态：{', '.join(modalities)}")
    print(f"训练轮数：{epochs}")
    print(f"批次大小：{batch_size}")
    print(f"学习率：{learning_rate}")
    print(f"设备：{device}")
    print(f"损失函数：{args.loss_type} (alpha={args.loss_alpha}, gamma={args.loss_gamma})")

    # Find fold files
    fold_files = find_fold_files(base_dir)

    if len(fold_files) == 0:
        print("\n错误：未找到任何 fold 的数据文件")
        print("\n请确认数据集目录结构，支持以下命名模式:")
        print("  - fold_1_train.csv + fold_1_val.csv")
        print("  - fold1_train.csv + fold1_val.csv")
        print("  - train_fold_1.csv + val_fold_1.csv")
        print("  - train_1.csv + val_1.csv")
        print("  - fold_1.csv (单个文件)")
        return

    print(f"\n找到 {len(fold_files)} 个 fold 的数据集")

    # Check the pretrained model
    if not os.path.exists(pretrained_model_path):
        print(f"错误：预训练模型不存在：{pretrained_model_path}")
        return

    # Load protein graph data
    print("\n加载蛋白质图数据...")
    lincs_edge_index, lincs_edge_attr, gene_list = load_protein_graph_data(gene_list_file)
    if lincs_edge_index is not None:
        print(f"蛋白质图：{len(gene_list)} 个节点，{lincs_edge_index.size(1)} 条边")
        lincs_edge_index = lincs_edge_index.to(device)
        if lincs_edge_attr is not None:
            lincs_edge_attr = lincs_edge_attr.to(device)
    else:
        print("警告：蛋白质图加载失败，将使用默认图结构")
        lincs_edge_index = None
        lincs_edge_attr = None

    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)

    # Preload the checkpoint (read only once)
    print(f"\n加载预训练模型：{pretrained_model_path}")
    checkpoint = torch.load(pretrained_model_path, map_location=device, weights_only=False)

    # Train each fold
    cv_results = []
    for fold_num, (train_csv, val_csv) in enumerate(fold_files):
        if val_csv is None:
            print(f"\n跳过 Fold {fold_num + 1}：验证集文件缺失")
            continue

        # ===== Recreate the model and load pretrained weights for each fold =====
        print(f"\n{'='*50}")
        print(f"Fold {fold_num + 1}: 重新创建模型并加载预训练权重")
        print(f"{'='*50}")

        model = MultimodalFinetuneModel(
            vocab_size=checkpoint['model_config']['vocab_size'],
            num_layer=checkpoint['model_config']['num_layer'],
            emb_dim=checkpoint['model_config']['emb_dim'],
            JK=checkpoint['model_config']['JK'],
            drop_ratio=checkpoint['model_config']['drop_ratio'],
            gnn_type=checkpoint['model_config']['gnn_type'],
            nhead=checkpoint['model_config']['nhead'],
            lincs_nodes=checkpoint['model_config']['lincs_nodes'],
            num_classes=2
        )

        # Filter pretrained weights, loading only matching layers
        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()

        # 1. Filter out unnecessary keys (reconstruction heads, contrastive projection heads, pretrain classification head)
        filtered_pretrained_dict = {}
        skipped_keys = {
            'recon_head': 0,
            'mol_proj': 0,
            'smiles_proj': 0,
            'smiles_classification_head': 0,
            'lincs_embed_recon_head': 0,
            'lincs_graph_recon_head': 0,
        }

        for k, v in pretrained_dict.items():
            skip = False
            for skip_key in skipped_keys.keys():
                if skip_key in k:
                    skipped_keys[skip_key] += 1
                    skip = True
                    break

            if not skip and k in model_dict:
                if v.shape != model_dict[k].shape:
                    print(f"  ⚠ 跳过维度不匹配的参数: {k} (checkpoint: {v.shape}, model: {model_dict[k].shape})")
                    skip = True

            if not skip and k in model_dict:
                filtered_pretrained_dict[k] = v

        # 2. Update the model dictionary
        model_dict.update(filtered_pretrained_dict)

        # 3. Load the weights
        model.load_state_dict(model_dict, strict=False)

        # 4. Move the model to the device
        model = model.to(device)

        print(f"预训练模型加载完成！")
        print(f"  - 加载的参数：{len(filtered_pretrained_dict)}")
        print(f"  - 跳过的预训练头参数：{sum(skipped_keys.values())}")
        for key, count in skipped_keys.items():
            if count > 0:
                print(f"    - {key}: {count}")

        fold_metrics = train_fold(
            fold_num=fold_num,
            train_csv=train_csv,
            val_csv=val_csv,
            modalities=modalities,
            model=model,
            device=device,
            lincs_edge_index=lincs_edge_index,
            lincs_edge_attr=lincs_edge_attr,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            output_dir=output_dir,
            loss_type=args.loss_type,
            loss_alpha=args.loss_alpha,
            loss_gamma=args.loss_gamma,
            toxicity_name=toxicity_name
        )
        cv_results.append(fold_metrics)

    # Compute average results
    print("\n" + "=" * 60)
    print("5 折交叉验证完成！")
    print("=" * 60)

    # Metric names following cv_5fold_final.py
    metrics_keys = ['auc', 'aupr', 'accuracy', 'precision', 'recall', 'specificity', 'balance_acc', 'f1']
    metric_names = {
        'auc': 'AUC-ROC',
        'aupr': 'AUPR',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Sensitivity',
        'specificity': 'Specificity',
        'balance_acc': 'Balance Acc',
        'f1': 'F1'
    }
    mean_results = {}
    std_results = {}

    for key in metrics_keys:
        values = [result[key] for result in cv_results]
        mean_results[key] = np.mean(values)
        std_results[key] = np.std(values)

    print("\n最终结果 (均值 ± 标准差):")
    for key in metrics_keys:
        print(f"{metric_names[key]:<15} {mean_results[key]:.4f} ± {std_results[key]:.4f}")

    # Save results
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    modalities_str = '_'.join(modalities) if isinstance(modalities, list) else str(modalities)
    loss_str = f"{args.loss_type}_a{args.loss_alpha}_g{args.loss_gamma}"
    results_file = os.path.join(output_dir, f'cv_results_{timestamp}_{modalities_str}_{loss_str}.txt')

    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("5 折交叉验证结果\n")
        f.write("=" * 60 + "\n\n")

        # Record basic info
        f.write(f"实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用模态：{modalities}\n")
        f.write(f"预训练模型：{pretrained_model_path}\n")
        f.write(f"数据集：{base_dir}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Loss Type: {args.loss_type}\n")
        f.write(f"Loss Alpha: {args.loss_alpha}\n")
        f.write(f"Loss Gamma: {args.loss_gamma}\n\n")

        f.write("每折结果:\n")
        for i, result in enumerate(cv_results):
            f.write(f"\nFold {i+1}:\n")
            f.write(f"  使用模态：{result.get('modalities', 'N/A')}\n")
            f.write(f"  训练时间：{result.get('training_time', 0):.2f} 秒\n")
            for key in metrics_keys:
                f.write(f"  {metric_names[key]:<15} {result[key]:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("汇总 (均值 ± 标准差):\n")
        f.write("=" * 60 + "\n")

        # Compute total and average training time
        total_time = sum([result.get('training_time', 0) for result in cv_results])
        avg_time = total_time / len(cv_results) if len(cv_results) > 0 else 0

        # Compute the std of the training time
        time_values = [result.get('training_time', 0) for result in cv_results]
        time_std = np.std(time_values) if len(time_values) > 0 else 0

        f.write(f"总训练时间：{total_time:.2f} 秒 ({total_time/60:.2f} 分钟)\n")
        f.write(f"平均每折时间：{avg_time:.2f} ± {time_std:.2f} 秒\n\n")

        for key in metrics_keys:
            f.write(f"{metric_names[key]:<15} {mean_results[key]:.4f} ± {std_results[key]:.4f}\n")

    print(f"\n结果已保存到：{results_file}")

    # Save results
    results_df = pd.DataFrame({
        'Metric': metrics_keys,
        'Mean': [mean_results[k] for k in metrics_keys],
        'Std': [std_results[k] for k in metrics_keys]
    })
    results_path = os.path.join(output_dir, "cv_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\n结果已保存到：{results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()