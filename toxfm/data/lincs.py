import pandas as pd
import torch


def load_lincs_data(filepath):
    """Load LINCS gene expression data."""
    try:
        df = pd.read_csv(filepath)
        # Extract gene expression columns (all columns except pert_id and SMILES)
        gene_cols = [col for col in df.columns if col not in ['pert_id', 'SMILES']]
        gene_expression = df[gene_cols].values
        return df, gene_cols, gene_expression
    except Exception as e:
        print(f"加载LINCS数据时出错: {e}")
        return None, [], None


def mask_lincs_features(lincs_features, mask_rate=0.15):
    """Mask LINCS gene expression features for pretraining."""
    # Create the mask
    mask = torch.rand(lincs_features.shape[0], lincs_features.shape[1]) < mask_rate
    
    # Store the original values
    original_values = lincs_features.clone()
    
    # Mask the features (fill with 0)
    masked_features = lincs_features.clone()
    masked_features[mask] = 0
    
    return masked_features, original_values, mask


def mask_protein_features(protein_features, mask_rate=0.15):
    """Mask protein features for pretraining."""
    # Create the mask
    mask = torch.rand(protein_features.shape[0], protein_features.shape[1]) < mask_rate

    # Store the original values
    original_values = protein_features.clone()

    # Mask the features
    masked_features = protein_features.clone()
    masked_features[mask] = 0  # Mask with zeros

    return masked_features, original_values, mask