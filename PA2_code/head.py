import torch
import torch.nn as nn
from torch.nn import functional as F

import globals

class Head(nn.Module):
    def __init__(self, head_size, d_model, applyMasking = False):
        super().__init__()
        # d_model = size of the embedding dimension
        # TODO: change 32 to d_model // num_heads
        # TODO: 3 linear models with 3 sets of parameters?
        self.query = nn.Linear(32, head_size, bias = False)
        self.key = nn.Linear(32, head_size, bias = False)
        self.value = nn.Linear(32, head_size, bias = False)
        self.applyMasking = applyMasking
        self.register_buffer('tril', torch.tril(torch.ones(globals.block_size, globals.block_size)))
    
    def forward(self, x):
        # Shape of x: (Batch size, Sequence length, Embedding dimension)
        B, T, C = x.shape
        # print(f'x.shape: {x.shape}')
        
        # calculate (derive) q, k, v from x
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Calculate attention scores using scaled-dot product attention mechanism
        similarity = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, T)
        
        if(self.applyMasking):
            # TODO: change to self.tril[:T][:T] like video?
            similarity = similarity.masked_fill(self.tril == 0, float('-inf'))
        
        attention = F.softmax(similarity, dim = -1) # (B, T, T)
        
        # Apply attention weights to the values to produce the output
        output = attention @ v
        return output, attention
        