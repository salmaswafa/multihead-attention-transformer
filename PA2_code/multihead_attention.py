import torch
import torch.nn as nn
from head import Head
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_model, applyMasking = False, applyAlibi = False):
        # d_model = size of the embedding dimension
        super().__init__()
        self.num_heads = num_heads
        self.applyAlibi = applyAlibi
        self.heads = nn.ModuleList([Head(head_size, d_model, applyMasking = applyMasking, applyAlibi = applyAlibi) for _ in range(num_heads)])
        
        if self.applyAlibi:
            self.m = []
            ratio = 2**(-2**(-math.log2(self.num_heads)+3))
            
            for i in range(self.num_heads):
                self.m.append(ratio**(i+1))
                    
            for i, head in enumerate(self.heads):
                head.alibi_slope = self.m[i]
                
        self.proj = nn.Linear(head_size * num_heads, d_model)
            
    
    def forward(self, x):
        #outputs = zip(*[head(x) for head in self.heads])
        
        outputs = []
        attn_maps = []
        
        # split input into chunks the same number of heads
        x_splits = torch.chunk(x, self.num_heads, dim=-1)
        
        # send each chunk to its relevant head
        for head, x in zip(self.heads, x_splits):
            x, attn_map = head(x)
            attn_maps.append(attn_map)
            outputs.append(x)
        
        # Concatenate all outputs into one tensor
        concat_outputs = torch.cat(outputs, dim=-1)
        # pass through final layer
        final_output = self.proj(concat_outputs) # another layer
        
        return final_output, attn_maps
