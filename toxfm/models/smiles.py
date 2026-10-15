import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
import random


class SMILESVocabulary:
    """
    Class for managing the SMILES vocabulary.
    """

    def __init__(self):
        # Define the basic SMILES characters
        self.chars = [
            '#', '%', '(', ')', '+', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '=', '@', 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'S', 'T', 'V', 'X', 'Z',
            '[', '\\', ']', 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's', 't'
        ]

        # Add special tokens
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.cls_token = '<CLS>'
        self.sep_token = '<SEP>'
        self.mask_token = '<MASK>'

        # Build the vocabulary
        self.char_to_idx = {char: idx for idx, char in enumerate(
            [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token] + self.chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        self.pad_idx = self.char_to_idx[self.pad_token]
        self.mask_idx = self.char_to_idx[self.mask_token]
        self.cls_idx = self.char_to_idx[self.cls_token]
        self.sep_idx = self.char_to_idx[self.sep_token]
        self.unk_idx = self.char_to_idx[self.unk_token]


def tokenize_smiles(smiles, vocab, max_len=128):
    """
    Convert a SMILES string into a token sequence.
    """
    tokens = [vocab.cls_token]
    tokens.extend(list(smiles))
    tokens.append(vocab.sep_token)

    if len(tokens) > max_len:
        tokens = tokens[:max_len - 1] + [vocab.sep_token]

    token_ids = [vocab.char_to_idx.get(token, vocab.unk_idx) for token in tokens]

    if len(token_ids) < max_len:
        token_ids.extend([vocab.pad_idx] * (max_len - len(token_ids)))

    return token_ids


def augment_smiles(smiles, augment_type='random'):
    """
    SMILES data augmentation.

    Args:
        smiles: SMILES string
        augment_type: augmentation type ('random', 'mask', 'substitute', 'delete')

    Returns:
        augmented SMILES string
    """
    if augment_type == 'random':
        return random_rotate_smiles(smiles)
    elif augment_type == 'mask':
        return mask_smiles_tokens(smiles, mask_ratio=0.1)
    elif augment_type == 'substitute':
        return substitute_smiles_tokens(smiles, substitute_ratio=0.05)
    elif augment_type == 'delete':
        return delete_smiles_tokens(smiles, delete_ratio=0.05)
    return smiles


def random_rotate_smiles(smiles):
    """
    Randomly rotate the SMILES string (start from a different atom while keeping the chemical structure unchanged).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles

        num_atoms = mol.GetNumAtoms()
        if num_atoms <= 1:
            return smiles

        start_atom = random.randint(0, num_atoms - 1)
        new_smiles = Chem.MolToSmiles(mol, rootedAtAtom=start_atom)

        return new_smiles
    except Exception:
        # If parsing or rotation fails, return the original SMILES
        return smiles


def mask_smiles_tokens(smiles, mask_ratio=0.1):
    """
    Randomly mask some SMILES tokens.
    """
    try:
        tokens = list(smiles)
        num_mask = max(1, int(len(tokens) * mask_ratio))
        mask_indices = random.sample(range(len(tokens)), min(num_mask, len(tokens)))

        for idx in mask_indices:
            if tokens[idx] not in ['[', ']', '(', ')', '@', '+', '-', '.', '/', '\\']:
                tokens[idx] = '*'

        return ''.join(tokens)
    except:
        return smiles


def substitute_smiles_tokens(smiles, substitute_ratio=0.05):
    """
    Randomly substitute some SMILES tokens.
    """
    try:
        tokens = list(smiles)
        num_substitute = max(1, int(len(tokens) * substitute_ratio))
        substitute_indices = random.sample(range(len(tokens)), min(num_substitute, len(tokens)))

        for idx in substitute_indices:
            if tokens[idx] not in ['[', ']', '(', ')', '@', '+', '-', '.', '/', '\\']:
                tokens[idx] = random.choice(['C', 'N', 'O', 'S'])

        return ''.join(tokens)
    except:
        return smiles


def delete_smiles_tokens(smiles, delete_ratio=0.05):
    """
    Randomly delete some SMILES tokens.
    """
    try:
        tokens = list(smiles)
        num_delete = max(1, int(len(tokens) * delete_ratio))
        delete_indices = random.sample(range(len(tokens)), min(num_delete, len(tokens)))

        for idx in sorted(delete_indices, reverse=True):
            if tokens[idx] not in ['[', ']', '(', ')', '@', '+', '-', '.', '/', '\\']:
                del tokens[idx]

        return ''.join(tokens)
    except:
        return smiles


def mask_tokens(inputs, vocab, mask_prob=0.15, mask_ratio=0.8, random_ratio=0.1, unchanged_ratio=0.1):
    """
    Mask SMILES tokens (for the MLM task).
    """
    labels = inputs.clone()
    batch_size, seq_len = inputs.shape

    for i in range(batch_size):
        non_special_positions = []
        for j in range(seq_len):
            token_id = inputs[i, j].item()
            if token_id not in [vocab.pad_idx, vocab.cls_idx, vocab.sep_idx]:
                non_special_positions.append(j)

        num_tokens = len(non_special_positions)
        num_mask = int(num_tokens * mask_prob)

        if num_mask == 0:
            continue

        mask_indices = random.sample(non_special_positions, min(num_mask, len(non_special_positions)))

        num_mask_actual = len(mask_indices)
        num_mask_mask = int(num_mask_actual * mask_ratio)
        num_mask_random = int(num_mask_actual * random_ratio)

        mask_mask_indices = mask_indices[:num_mask_mask]
        random_indices = mask_indices[num_mask_mask:num_mask_mask + num_mask_random]
        unchanged_indices = mask_indices[num_mask_mask + num_mask_random:]

        for idx in mask_mask_indices:
            inputs[i, idx] = vocab.mask_idx

        for idx in random_indices:
            inputs[i, idx] = random.randint(5, vocab.vocab_size - 1)

        for idx in unchanged_indices:
            labels[i, idx] = inputs[i, idx]

    labels[inputs == vocab.pad_idx] = vocab.pad_idx

    return inputs, labels


class SMILESEmbedding(nn.Module):
    """
    SMILES embedding layer.
    """

    def __init__(self, vocab_size, embed_dim, max_len=128):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.embed_dim = embed_dim

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)

        x = self.token_embed(x) + self.pos_embed(pos)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class SMILESAttentionBlock(nn.Module):
    """
    Self-attention block.
    """

    def __init__(self, embed_dim, nhead=8, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x


class SMILESRNNEncoder(nn.Module):
    """
    SMILES pre-training model (uses an RNN instead of BERT/Transformer).
    Lighter-weight and suitable for sequence modeling.
    """

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2, max_len=128, rnn_type='lstm'):
        super().__init__()
        self.embedding = SMILESEmbedding(vocab_size, embed_dim, max_len)

        # RNN encoder (LSTM or GRU)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, 
                              batch_first=True, bidirectional=True, dropout=0.1)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True, dropout=0.1)
        else:
            raise ValueError(f"不支持的 RNN 类型：{rnn_type}")

        # Projection layer
        self.output_proj = nn.Linear(hidden_dim * 2, embed_dim)  # bidirectional, hence *2
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, x, return_embeddings=False):
        # Embed
        x = self.embedding(x)

        # RNN encode
        rnn_output, (hidden, cell) = self.rnn(x)

        # Use the final hidden states of the bidirectional RNN
        # hidden: [num_layers * 2, batch, hidden_dim]
        batch_size = x.size(0)

        # Concatenate the hidden states from both directions
        hidden_fwd = hidden[-2, :, :]   # forward last layer
        hidden_bwd = hidden[-1, :, :]   # backward last layer
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # [batch, hidden_dim*2]

        # Project to embed_dim
        output = self.output_proj(hidden_cat)  # [batch, embed_dim]

        if return_embeddings:
            # To be compatible with the Transformer interface, add a seq_len dimension
            return None, output.unsqueeze(1)  # [batch, 1, embed_dim]
        return output


class SMILESPretrainModel(nn.Module):
    """
    SMILES pre-training model (uses an RNN, supports the MLM task).
    """

    def __init__(self, vocab_size, embed_dim=256, nhead=8, num_layers=6, max_len=128):
        super().__init__()
        # To stay compatible with the original interface, ignore the nhead parameter (not needed for RNN)
        # num_layers maps to the number of RNN layers
        rnn_layers = max(1, num_layers // 2)  # RNN layers are usually fewer

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Bi-LSTM encoder
        self.rnn = nn.LSTM(embed_dim, embed_dim, num_layers=rnn_layers,
                          batch_first=True, bidirectional=True, dropout=0.1)

        # MLM prediction head
        self.mlm_head = nn.Linear(embed_dim * 2, vocab_size)  # bidirectional * 2

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def forward(self, x, return_embeddings=False):
        # Embed
        x = self.embedding(x)

        # RNN encode
        rnn_output, (hidden, cell) = self.rnn(x)
        # rnn_output: [batch, seq_len, hidden_dim*2]

        if return_embeddings:
            # Return per-position embeddings (for MLM)
            return None, rnn_output  # [batch, seq_len, embed_dim*2]
        return rnn_output


class SMILESContrastivePretrain(nn.Module):
    """
    SMILES contrastive pre-training module (uses an RNN encoder).

    Combines Masked Language Modeling and contrastive learning.
    """

    def __init__(self, vocab_size, embed_dim=256, nhead=8, num_layers=6, 
                 max_len=128, temperature=0.07):
        super().__init__()

        # SMILES encoder (uses RNN, shared with MLM)
        self.encoder = SMILESPretrainModel(vocab_size, embed_dim, nhead, num_layers, max_len)

        self.temperature = temperature
        self.embed_dim = embed_dim

    def forward(self, input_ids, augmented_input_ids=None, masked_input_ids=None, labels=None, 
                return_loss=True):
        """
        Forward pass (MLM only).

        Args:
            input_ids: input_ids of the original SMILES [batch_size, seq_len]
            augmented_input_ids: input_ids of the augmented SMILES [batch_size, seq_len] (unused, kept for compatibility)
            masked_input_ids: masked version of input_ids (used for MLM)
            labels: MLM labels
            return_loss: whether to return the loss

        Returns:
            losses: loss dictionary {'mlm': mlm_loss, 'total': total_loss}
        """
        losses = {}

        # MLM task
        if masked_input_ids is not None and labels is not None:
            _, mlm_embeddings = self.encoder(masked_input_ids, return_embeddings=True)
            mlm_logits = self.encoder.mlm_head(mlm_embeddings)
            mlm_loss = nn.CrossEntropyLoss()(
                mlm_logits.view(-1, self.encoder.vocab_size),
                labels.view(-1)
            )
            losses['mlm'] = mlm_loss
            losses['total'] = mlm_loss
        else:
            losses['total'] = torch.tensor(0.0, device=input_ids.device)

        if return_loss:
            return losses

        return None