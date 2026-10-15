import logging
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from toxfm.models.smiles import tokenize_smiles

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    Multimodal dataset that reads data from LINCS CSV files
    """

    def __init__(self, csv_file_path, vocab, max_len=128, modalities=None):
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Use all modalities by default
        if modalities is None:
            modalities = ['smiles', 'mol', 'lincs_embed', 'lincs_graph']

        self.modalities = modalities

        # Extract the SMILES column
        self.smiles_list = df['SMILES'].tolist()

        # Extract gene expression data (all columns except pert_id and SMILES)
        if 'lincs_embed' in modalities or 'lincs_graph' in modalities:
            gene_cols = [col for col in df.columns if col not in ['pert_id', 'SMILES']]
            self.gene_expression = df[gene_cols].values.astype(np.float32)

            # Check and fill NaN values
            nan_count = np.isnan(self.gene_expression).sum()
            if nan_count > 0:
                print(f"  Found {nan_count} NaN values in gene expression data, filling with 0...")
                self.gene_expression = np.nan_to_num(self.gene_expression, nan=0.0)

            # Use the raw gene expression data, no log transform
            print(f"Gene expression data range: [{self.gene_expression.min():.4f}, {self.gene_expression.max():.4f}]")
        else:
            self.gene_expression = None

        self.vocab = vocab
        self.max_len = max_len
        self.size = len(self.smiles_list)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        result = {}

        # SMILES data
        if 'smiles' in self.modalities:
            token_ids = tokenize_smiles(smiles, self.vocab, self.max_len)
            result['smiles_tokens'] = torch.tensor(token_ids, dtype=torch.long)

        # LINCS embedding data
        if 'lincs_embed' in self.modalities or 'lincs_graph' in self.modalities:
            gene_expr = self.gene_expression[idx]
            result['lincs_embed'] = torch.tensor(gene_expr, dtype=torch.float)

        # Molecular graph data
        if 'mol' in self.modalities:
            try:
                from rdkit import Chem
                from torch_geometric.data import Data

                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    atom_features = []
                    for atom in mol.GetAtoms():
                        atom_features.append([
                            atom.GetAtomicNum(),
                            atom.GetChiralTag()
                        ])

                    edge_indices = []
                    edge_features = []
                    for bond in mol.GetBonds():
                        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                        edge_indices.append([start, end])
                        edge_indices.append([end, start])
                        edge_features.append([bond.GetBondTypeAsDouble(), 0])
                        edge_features.append([bond.GetBondTypeAsDouble(), 1])

                    if len(edge_indices) > 0:
                        x = torch.tensor(atom_features, dtype=torch.float)
                        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                        edge_attr = torch.tensor(edge_features, dtype=torch.float)
                        result['mol_data'] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                    else:
                        x = torch.tensor(atom_features, dtype=torch.float)
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                        edge_attr = torch.empty((0, 2), dtype=torch.float)
                        result['mol_data'] = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                else:
                    result['mol_data'] = None
            except Exception:
                result['mol_data'] = None

        return result


class MultimodalFinetuneDataset(Dataset):
    """
    Multimodal fine-tuning dataset supporting different combinations of four modalities
    """
    def __init__(self, csv_file_path, vocab, modalities, max_len=128, gene_expression_cols=None):
        """
        Initialize the multimodal dataset

        Args:
            csv_file_path: CSV file path
            vocab: SMILES vocabulary
            modalities: list of modalities to use, e.g. ['mol', 'smiles', 'lincs_embed', 'lincs_graph']
            max_len: maximum SMILES length
            gene_expression_cols: column names of gene expression data
        """
        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Extract labels
        if 'Label' in df.columns:
            self.labels = df['Label'].astype(int).tolist()
        elif 'label' in df.columns:
            self.labels = df['label'].astype(int).tolist()
        else:
            raise ValueError("CSV文件必须包含'Label'或'label'列")

        # Initialize data for each modality
        self.modalities = modalities
        self.data = {}

        # SMILES data
        if 'smiles' in modalities:
            self.smiles_list = df['SMILES'].tolist()
            self.vocab = vocab
            self.max_len = max_len

        # Molecular graph data
        if 'mol' in modalities:
            from rdkit import Chem
            from torch_geometric.data import Data

            self.mol_data_list = []
            for smiles in df['SMILES'].tolist():
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        # Create atom features
                        atom_features = []
                        for atom in mol.GetAtoms():
                            atom_features.append([
                                atom.GetAtomicNum(),
                                atom.GetChiralTag(),
                                atom.GetFormalCharge(),
                                atom.GetHybridization().real,
                                atom.GetNumExplicitHs(),
                                atom.GetIsAromatic()
                            ])

                        # Create edge index and edge features
                        edge_indices = []
                        edge_features = []
                        for bond in mol.GetBonds():
                            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            edge_indices.append([start, end])
                            edge_indices.append([end, start])  # add reverse edge

                            # Edge features: bond type and direction - ensure at least 2 features
                            bond_type = bond.GetBondTypeAsDouble()
                            bond_dir = 0.0  # default direction
                            edge_features.append([bond_type, bond_dir])
                            edge_features.append([bond_type, bond_dir])

                        if len(edge_indices) > 0:
                            x = torch.tensor(atom_features, dtype=torch.float)
                            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                            edge_attr = torch.tensor(edge_features, dtype=torch.float)

                            # Ensure edge_attr is a 2D tensor [num_edges, num_features] with at least 2 features
                            if edge_attr.dim() == 1:
                                edge_attr = edge_attr.unsqueeze(1)
                            if edge_attr.size(1) < 2:
                                # If edge feature dim < 2, expand to 2 dims
                                padding = torch.zeros(edge_attr.size(0), 2 - edge_attr.size(1), dtype=edge_attr.dtype, device=edge_attr.device)
                                edge_attr = torch.cat([edge_attr, padding], dim=1)

                            mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                        else:
                            # If there are no edges, create a graph with only nodes
                            x = torch.tensor(atom_features, dtype=torch.float)
                            edge_index = torch.empty((2, 0), dtype=torch.long)
                            edge_attr = torch.empty((0, 2), dtype=torch.float)  # ensure edge feature dim is 2
                            mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

                        self.mol_data_list.append(mol_data)
                    else:
                        # If SMILES cannot be parsed, create an empty graph
                        x = torch.empty((0, 6), dtype=torch.float)
                        edge_index = torch.empty((2, 0), dtype=torch.long)
                        edge_attr = torch.empty((0, 2), dtype=torch.float)  # ensure edge feature dim is 2
                        mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                        self.mol_data_list.append(mol_data)
                except Exception as e:
                    # If generating the molecular graph fails, create an empty graph
                    x = torch.empty((0, 6), dtype=torch.float)
                    edge_index = torch.empty((2, 0), dtype=torch.long)
                    edge_attr = torch.empty((0, 2), dtype=torch.float)  # ensure edge feature dim is 2
                    mol_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
                    self.mol_data_list.append(mol_data)

        # Both LINCS embedding data and LINCS graph data need gene expression data
        if 'lincs_embed' in modalities or 'lincs_graph' in modalities:
            if gene_expression_cols is None:
                gene_cols = [col for col in df.columns if col not in ['SMILES', 'Label', 'label', 'pert_id']]
            else:
                gene_cols = gene_expression_cols

            # Truncate to 978 genes (dimension expected by the model)
            if len(gene_cols) > 978:
                gene_cols = gene_cols[:978]
                logger.info(f"LINCS 数据截断到 978 个基因")

            self.gene_expression_data = df[gene_cols].values.astype(np.float32)

        # LINCS graph data (placeholder here; actual graph structure passed during training)
        if 'lincs_graph' in modalities:
            # Graph structure data is usually defined outside the model
            pass

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'label': torch.tensor(self.labels[idx], dtype=torch.long)}

        # Add SMILES data
        if 'smiles' in self.modalities:
            smiles = self.smiles_list[idx]
            token_ids = tokenize_smiles(smiles, self.vocab, self.max_len)
            sample['smiles_tokens'] = torch.tensor(token_ids, dtype=torch.long)

        # Add molecular graph data
        if 'mol' in self.modalities:
            sample['mol_data'] = self.mol_data_list[idx]

        # Add LINCS embedding data
        if 'lincs_embed' in self.modalities or 'lincs_graph' in self.modalities:
            # lincs_graph also needs lincs_embed as node features
            sample['lincs_embed'] = torch.tensor(self.gene_expression_data[idx], dtype=torch.float)

        return sample