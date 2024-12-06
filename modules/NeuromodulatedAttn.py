from torch import nn
import torch
import torch.nn.functional as F

class NeuromodulatedAttention(nn.Module):
    def __init__(self, d_model, dropout=0.3, **kwargs):
        """
        Neuromodulated Attention mechanism inspired by dopamine (reward modulation)
        and serotonin (uncertainty modulation).
        Args:
            d_model (int): Dimensionality of the input query/key/value.
            dropout (float): Dropout rate for regularization.
        """
        super(NeuromodulatedAttention, self).__init__()
        self.d_model = d_model
        self.batch_first = True
        self._qkv_same_embed_dim = False
        # Dopamine network for task-relevant signals
        self.dopamine_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Serotonin network for uncertainty signals
        self.serotonin_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # Dynamic scaling factors
        self.dopamine_scale = nn.Parameter(torch.tensor(1.0))
        self.serotonin_scale = nn.Parameter(torch.tensor(1.0))

        # Dropout for attention probabilities
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        """
        Forward pass of the neuromodulated attention mechanism.
        Args:
            Q (Tensor): Query matrix of shape (batch_size, seq_len, d_model).
            K (Tensor): Key matrix of shape (batch_size, seq_len, d_model).
            V (Tensor): Value matrix of shape (batch_size, seq_len, d_model).
        Returns:
            Tensor: Output of the attention mechanism with neuromodulation.
        """
        # Compute scaled dot-product attention scores
        scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=Q.device))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # Shape: (batch_size, seq_len, seq_len)

        # Compute entropy from attention scores (uncertainty)
        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)  # Convert scores to probabilities
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-6), dim=-1)  # Shape: (batch_size, seq_len)

        # Project entropy to d_model dimensions for serotonin
        entropy_projected = entropy.unsqueeze(-1).expand(-1, -1, self.d_model)  # Shape: (batch_size, seq_len, d_model)

        # Compute serotonin signal using entropy_projected
        serotonin_signal = torch.sigmoid(self.serotonin_network(entropy_projected)).squeeze(-1)  # Shape: (batch_size, seq_len)

        # Compute dopamine signal directly from Q (task-relevance)
        dopamine_signal = torch.sigmoid(self.dopamine_network(Q)).squeeze(-1)  # Shape: (batch_size, seq_len)

        # Modulate attention scores
        modulation_signal = (self.dopamine_scale * dopamine_signal.unsqueeze(-1)) - \
                            (self.serotonin_scale * serotonin_signal.unsqueeze(-1))  # Shape: (batch_size, seq_len, seq_len)

        modulated_scores = attention_scores + modulation_signal

        # Normalize scores for numerical stability
        modulated_scores = (modulated_scores - modulated_scores.mean(dim=-1, keepdim=True)) / (
            modulated_scores.std(dim=-1, keepdim=True) + 1e-6
        )
        # print(f"{modulated_scores.shape=}, {modulation_signal.shape=}")

        # Compute final attention probabilities and apply dropout
        attention_probs = torch.nn.functional.softmax(modulated_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # Compute the final attention output
        output = torch.matmul(attention_probs, V)  # Shape: (batch_size, seq_len, d_model)

        return output

class NeuromodulatedAttentionWithElectrodes(nn.Module):
    def __init__(self, d_model, num_electrodes, num_classes, dropout=0.3):
        super(NeuromodulatedAttentionWithElectrodes, self).__init__()
        self.d_model = d_model

        # Learnable electrode embeddings
        self.electrode_embedding = nn.Embedding(num_electrodes, d_model)

        # Dopamine and Serotonin networks
        self.dopamine_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        self.serotonin_network = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self.dopamine_scale = nn.Parameter(torch.tensor(1.0))
        self.serotonin_scale = nn.Parameter(torch.tensor(1.0))
        self.attention_dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_electrodes * d_model, num_classes)
        )

        self.electrode_ids = []

    def forward(self, Q, K, V):
        # Add electrode embeddings to queries, keys, and values
        electrode_embeddings = self.electrode_embedding(self.electrode_ids)
        Q += electrode_embeddings
        K += electrode_embeddings
        V += electrode_embeddings

        # Compute scaled dot-product attention scores
        scale = torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32, device=Q.device))
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Entropy for serotonin signal
        attention_probs = torch.softmax(attention_scores, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-6), dim=-1)

        # Project entropy to d_model dimension
        entropy_projected = entropy.unsqueeze(-1).expand(-1, -1, self.d_model)

        # Compute dopamine and serotonin signals
        dopamine_signal = torch.sigmoid(self.dopamine_network(Q)).squeeze(-1)
        serotonin_signal = torch.sigmoid(self.serotonin_network(entropy_projected)).squeeze(-1)

        # Combine dopamine and serotonin signals
        modulation_signal = (self.dopamine_scale * dopamine_signal.unsqueeze(-1)) - \
                            (self.serotonin_scale * serotonin_signal.unsqueeze(-1))
        modulated_scores = attention_scores + modulation_signal

        # Normalize modulated scores
        modulated_scores = (modulated_scores - modulated_scores.mean(dim=-1, keepdim=True)) / (
            modulated_scores.std(dim=-1, keepdim=True) + 1e-6
        )

        # print(modulated_scores.shape, modulation_signal.shape)

        # Apply softmax to compute attention probabilities
        attention_probs = torch.nn.functional.softmax(modulated_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        # Compute the final attention output
        attention_output = torch.matmul(attention_probs, V)
        return self.fc(attention_output)

class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, **kwargs):
        super().__init__(d_model, nhead, **kwargs)
        self.attn_weights = None
        self.self_attn = NeuromodulatedAttention(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
        # Self-attention with output attn_weights
        src2 = self.self_attn(src, src, src)
        # Apply dropout and add & norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # Return output and attention weights
        return src