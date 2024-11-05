# add all  your Encoder and Decoder code here

import torch.nn as nn
import torch

from block import Block

class EncoderModel(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, num_heads, num_layers):
        super().__init__()  # Initialize nn.Module
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Generates random embedding vectors for each token
        # create a table of random word embeddings for all the vocab we have in the dataset
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # create a table of random positional embeddings for all the word positions we have in the dataset block size
        # TODO: normally, this should not be trainable but here it is random and trainable. should this also be ordered? -20, -10, 0, 10, 20? how?
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        
        print(f'emb of word of index 2: {self.token_embedding_table(torch.tensor([2]))}')
        
    def forward(self, idx, targets = None):
        B, T = idx.shape
        
        # get token embeddings of the words from the table using the indices
        tok_emb = self.token_embedding_table(idx)
        # get positional embeddings of the words from the table using the indices
        pos_emb = self.position_embedding_table(torch.arange(T, device = self.device))
        
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