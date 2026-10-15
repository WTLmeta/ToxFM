import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


def compute_metrics(pred):
    """Compute evaluation metrics - following cv_5fold_final.py"""
    predictions, labels = pred

    # predictions is the probability of the positive class (1D array)
    if isinstance(predictions, np.ndarray) and predictions.ndim == 2:
        # If it is a 2D array, take the positive class probability
        probs = predictions[:, 1]
    else:
        probs = predictions

    # Convert to classes using threshold 0.5
    pred_classes = (probs > 0.5).astype(int)

    # Compute AUC
    try:
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.5

    # Compute AUPR
    try:
        aupr = average_precision_score(labels, probs)
    except:
        aupr = 0.0

    # Compute other metrics
    accuracy = accuracy_score(labels, pred_classes)
    precision = precision_score(labels, pred_classes, average='binary', zero_division=0)
    recall = recall_score(labels, pred_classes, average='binary', zero_division=0)
    f1 = f1_score(labels, pred_classes, average='binary', zero_division=0)

    # Compute specificity
    tn = np.sum((pred_classes == 0) & (labels == 0))
    fp = np.sum((pred_classes == 1) & (labels == 0))
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
    else:
        specificity = 0.0

    # Balanced accuracy
    balance_acc = (recall + specificity) / 2.0

    return {
        'auc': auc,
        'aupr': aupr,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'balance_acc': balance_acc,
        'f1': f1
    }


class FocalLoss(torch.nn.Module):
    """Focal Loss for binary/multi-class classification"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: [batch_size, num_classes]
            targets: [batch_size]
        """
        probs = torch.softmax(logits, dim=1)
        p = probs[:, 1]  # positive class probability
        p = torch.clamp(p, 1e-8, 1.0 - 1e-8)

        # p_t: probability of the correct class
        p_t = torch.where(targets == 1, p, 1 - p)

        # alpha_t: class weight
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Focal weight
        focal_weight = alpha_t * torch.pow(1 - p_t, self.gamma)

        loss = -focal_weight * torch.log(p_t)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss