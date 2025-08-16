import torch
import torch.nn as nn
from torch import Tensor
import math


# emb_size = 512
# maxlen = 5000
# batch_size = 64
# seq_len = 100   (the length of an input sequence in the forward pass)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()

        # pos: (maxlen, 1) -> (5000, 1)
        # Creates a column vector of positions [0, 1, ..., 4999]
        pos = torch.arange(maxlen).unsqueeze(1)

        # i: (emb_size / 2,) -> (256,)
        # Creates a row vector for the even dimension indices [0, 2, ..., 510]
        i = torch.arange(0, emb_size, 2)

        # angle_rates: (maxlen, emb_size / 2) -> (5000, 256)
        # Calculates the arguments for sin/cos using broadcasting.
        # (5000, 1) / (256,) results in a (5000, 256) matrix.
        angle_rates = pos / (10000 ** (i.float() / emb_size))

        # pos_encoding: (maxlen, emb_size) -> (5000, 512)
        pos_encoding = torch.zeros(maxlen, emb_size)
        
        # Fills even indices (0, 2, ...) with sin values.
        # The slice pos_encoding[:, 0::2] has shape (5000, 256).
        pos_encoding[:, 0::2] = torch.sin(angle_rates)

        # Fills odd indices (1, 3, ...) with cos values.
        # The slice pos_encoding[:, 1::2] has shape (5000, 256).
        pos_encoding[:, 1::2] = torch.cos(angle_rates)

        # --- Finalizing and Storing ---
        # pos_encoding: (maxlen, 1, emb_size) -> (5000, 1, 512)
        # Adds a dimension for batch broadcasting in the forward pass.
        pos_encoding = pos_encoding.unsqueeze(1)
        
        # Registers 'pos_encoding' as a buffer. It's part of the model's state
        # but not a parameter to be trained.
        self.register_buffer('pos_encoding', pos_encoding)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: Tensor):
        # token_embedding (input): (seq_len, batch_size, emb_size) -> (100, 64, 512)
        seq_len = token_embedding.size(0)

        # Add positional encoding to token embedding.
        # self.pos_encoding[:seq_len, :] slices the buffer to get shape (100, 1, 512).
        # Broadcasting adds this to token_embedding (100, 64, 512).
        # The result has shape (100, 64, 512).
        output = token_embedding + self.pos_encoding[:seq_len, :]
        return self.dropout(output)
    
    

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        # Creates a lookup table of shape (vocab_size, emb_size)
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        # Input 'tokens' shape: (seq_len, batch_size)
        # Output shape: (seq_len, batch_size, emb_size)
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)