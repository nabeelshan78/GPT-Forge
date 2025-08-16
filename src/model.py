import torch
import torch.nn as nn
from torch import Tensor

from .modules import TokenEmbedding, PositionalEncoding
from .config import PAD_IDX


class CustomGPTModel(nn.Module):
    """
    A custom GPT-style Transformer model for language generation.
    """
    def __init__(self, 
                 vocab_size: int, 
                 embed_size: int, 
                 num_heads: int, 
                 num_layers: int, 
                 dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        # Input Embedding Pipeline
        self.embedding_pipeline = nn.Sequential(
            TokenEmbedding(vocab_size, embed_size),
            PositionalEncoding(embed_size, dropout=dropout, maxlen=max_seq_len)
        )

        # Core Transformer Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, 
            nhead=num_heads, 
            dropout=dropout, 
            batch_first=False     # Expects (seq_len, batch_size, embed_size)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output Projection Layer
        self.lm_head = nn.Linear(embed_size, vocab_size)

        # Initialize weights after all layers are defined
        self.init_weights()

    def init_weights(self):
        """Initializes model weights using Xavier uniform distribution."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def create_masks(src: Tensor, device):
        """
        Creates the causal (look-ahead) and padding masks for the source sequence.
        """
        seq_len = src.shape[0]
        
        # Causal mask: Prevents attending to future tokens.
        # Shape: (seq_len, seq_len)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)
        
        # Padding mask: Prevents attending to <pad> tokens.
        # Shape: (batch_size, seq_len)
        padding_mask = (src == PAD_IDX).transpose(0, 1)
        
        return causal_mask, padding_mask
    

    def forward(self, src: Tensor):
        """
        Defines the forward pass of the model.
        
        Args:
            src (Tensor): Input tensor of token IDs.
                          Shape: (seq_len, batch_size)
        
        Returns:
            Tensor: Output logits over the vocabulary.
                    Shape: (seq_len, batch_size, vocab_size)
        """
        # Create masks based on the input tensor.
        src_mask, src_padding_mask = self.create_masks(src, device=src.device)
        
        # Prepare input: Apply token embedding and positional encoding.
        # src shape: (seq_len, batch_size) -> (seq_len, batch_size, embed_size)
        src_emb = self.embedding_pipeline(src)

        # Pass through the main Transformer blocks.
        # Shape remains: (seq_len, batch_size, embed_size)
        output = self.transformer_encoder(
            src_emb, 
            mask=src_mask, 
            src_key_padding_mask=src_padding_mask
        )
        
        # Project to vocabulary space to get final logits.
        # output shape: (seq_len, batch_size, embed_size) -> (seq_len, batch_size, vocab_size)
        logits = self.lm_head(output)
        
        return logits