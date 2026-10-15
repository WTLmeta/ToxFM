# TOXFM: A Multimodal Foundation Model for Molecular Toxicity Prediction

We propose **TOXFM**, a multimodal foundation model for small-molecule toxicity and drug-response prediction. It encodes each compound through four complementary views — a molecular graph (message-passing GNN), a SMILES sequence (a self-supervised RNN encoder), a LINCS transcriptomic embedding, and a LINCS protein-protein-interaction graph — and fuses them with a cross-attention / co-attention module. The model is first pretrained with a set of modality-aware self-supervised objectives (masked-language modeling, graph/embedding reconstruction, and inter-modality contrastive learning) and then fine-tuned end-to-end for downstream classification, producing robust generalization to unseen chemical scaffolds compared with conventional single-modality methods.

## Repository Structure

```
toxfm/
├── toxfm/                     # Core Python package
│   ├── models/
│   │   ├── gnn.py             # Graph encoders (GIN / GCN / GAT / GraphSAGE, ProteinGNN)
│   │   ├── attention.py       # Co-attention + Transformer cross-fusion
│   │   ├── smiles.py          # SMILES vocabulary, augmentation, self-supervised encoder
│   │   ├── pretrain.py        # MultimodalPretrainModel + dynamic loss balancer
│   │   └── finetune.py        # MultimodalFinetuneModel (classification head)
│   ├── data/
│   │   ├── protein_graph.py   # STRING PPI graph construction / loading
│   │   ├── lincs.py           # LINCS loading + masking
│   │   ├── dataset.py         # Pretraining / fine-tuning datasets
│   │   └── folds.py           # Fold-split discovery utilities
│   └── utils/
│       └── metrics.py         # Evaluation metrics + FocalLoss
├── train_pretrain.py          # Pretraining entry point
├── train_finetune.py          # Fine-tuning entry point (single CV, StratifiedKFold)
├── run_cv.py                  # 5-fold cross-validation entry point
├── gene_list.txt              # LINCS probe/gene list used to build the PPI graph
└── legacy/                    # Original monolithic scripts (kept for reference)
```

## Environment Setup

### Core Dependencies

- Python 3.9
- PyTorch 2.5
- PyTorch Geometric 2.5 (with `torch-scatter`)

### Key Python Packages

- `numpy`
- `pandas`
- `scikit-learn`
- `tqdm`
- `rdkit` (SMILES canonicalization and augmentation)
- `networkx` (protein-protein-interaction graph)
- `requests` (STRING API access)

### Installation

```bash
# 1. Create and activate a conda environment
conda create -n toxfm python=3.9 -y
conda activate toxfm

# 2. Install PyTorch (adjust the CUDA build to match your hardware)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric
pip install torch_geometric==2.5.1 torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# 4. Install the remaining dependencies
pip install numpy pandas scikit-learn tqdm rdkit networkx requests
```

## Quick Start

### Notice

The pipeline assumes the following inputs are available:

- A **CSV file** that contains a `SMILES` column, one `Label` column (for supervised tasks), and additional columns that are interpreted as LINCS gene-expression features.
- A **gene list file** (`gene_list.txt`) listing the LINCS probes/genes, used to construct the protein-protein-interaction graph via the STRING API.
- A **pretrained model checkpoint** produced by `train_pretrain.py`.

### 1. Pretraining

`train_pretrain.py` performs self-supervised multimodal pretraining. The configuration (paths, modalities, hyperparameters) is set at the bottom of the script; edit it and run:

```bash
python train_pretrain.py
```

By default it uses all four modalities `['mol', 'smiles', 'lincs_embed', 'lincs_graph']` and saves the checkpoint defined by `output_model_path`.

### 2. Fine-tuning

`train_finetune.py` fine-tunes a pretrained checkpoint on a labeled CSV using stratified K-fold cross-validation and reports the mean and standard deviation of the metrics:

```bash
python train_finetune.py \
  --csv_file data/merge_linc.csv \
  --pretrained_model_path model/multimodal_pretrain_model.pth \
  --gene_list_file gene_list.txt \
  --modalities mol smiles lincs_embed lincs_graph \
  --epochs 50 --batch_size 64 --cv_folds 5 --learning_rate 3e-4 \
  --output_dir output/finetuned
```

> `--modalities` accepts any combination of `mol`, `smiles`, `lincs_embed`, `lincs_graph`. The `gene_list_file` is required when `lincs_graph` is used.

### 3. Cross-validation on pre-split folds

`run_cv.py` runs 5-fold cross-validation on a dataset that is already partitioned into fold files, and exports per-fold and aggregated metrics:

```bash
python run_cv.py \
  --base_dir data/dataset_splits \
  --pretrained_model_path model/multimodal_pretrain_model.pth \
  --gene_list_file gene_list.txt \
  --modalities mol smiles lincs_embed lincs_graph \
  --epochs 50 --batch_size 32 --learning_rate 3e-4 \
  --loss_type weighted_ce \
  --output_dir output/cv_results
```

Set `--loss_type focal --loss_alpha 0.25 --loss_gamma 2.0` to switch to the Focal Loss.