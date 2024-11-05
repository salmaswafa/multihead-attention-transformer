# add all  your Encoder and Decoder code here

import torch.nn as nn
import torch

from block import Block

class EncoderModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, num_layers):
        super().__init__()  # Initialize nn.Module
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Generates random embedding vectors for each token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        print(f'emb of word of index 2: {self.token_embedding_table(torch.tensor([2]))}')
        
    def forward(self, idx, targets = None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device = self.device))
        
        x = tok_emb + pos_emb
        attn_maps = []
        
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.append(attn_map)
            
        x = self.ln_f(x)
        
        return x, attn_maps