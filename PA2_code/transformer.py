# add all  your Encoder and Decoder code here

import torch.nn as nn
import torch
from torch.nn import functional as F

from block import Block
import globals

class EncoderModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, num_layers):
        super().__init__()  # Initialize nn.Module
        # Generates random embedding vectors for each token
        # create a table of random word embeddings for all the vocab we have in the dataset
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # create a table of random positional embeddings for all the word positions we have in the dataset block size
        # normally, this should not be trainable but here it is random and trainable
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        
    def forward(self, idx, targets = None):
        B, T = idx.shape
        
        # get token embeddings of the words from the table using the indices
        tok_emb = self.token_embedding_table(idx)
        # get positional embeddings of the words from the table using the indices
        pos_emb = self.position_embedding_table(torch.arange(T, device = globals.device))
        
        # add both together for the final embedding - word meaning + position (for better context)
        x = tok_emb + pos_emb
        attn_maps = []
        
        # send to each block
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.append(attn_map)
            
        # normalize output
        x = self.ln_f(x)
        
        return x, attn_maps
    
class DecoderModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, num_layers, applyAlibi = False):
        super().__init__()  # Initialize nn.Module
        self.applyAlibi = applyAlibi
        # Generates random embedding vectors for each token
        # create a table of random word embeddings for all the vocab we have in the dataset
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # create a table of random positional embeddings for all the word positions we have in the dataset block size
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, num_heads, applyMasking = True, applyAlibi = applyAlibi) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
    def forward(self, idx, targets = None):
        B, T = idx.shape
        
        # get token embeddings of the words from the table using the indices
        tok_emb = self.token_embedding_table(idx)
        
        if not self.applyAlibi:
            # get positional embeddings of the words from the table using the indices
            pos_emb = self.position_embedding_table(torch.arange(T, device = globals.device))
            
            # add both together for the final embedding - word meaning + position (for better context)
            x = tok_emb + pos_emb
        else:
            x = tok_emb
            
        attn_maps = []
        
        # send to each block
        for block in self.blocks:
            x, attn_map = block(x)
            attn_maps.append(attn_map)
            
        # normalize output
        x = self.ln_f(x)
        
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss, attn_maps