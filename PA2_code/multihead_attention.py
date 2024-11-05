import torch
import torch.nn as nn
from head import Head


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, d_model):
        # d_model = size of the embedding dimension
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([Head(head_size, d_model) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, d_model)
    
    def forward(self, x):
        #outputs = zip(*[head(x) for head in self.heads])
        
        outputs = []
        attn_maps = []
        
        x_splits = torch.chunk(x, self.num_heads, dim=-1)
        
        for head, x in zip(self.heads, x_splits):
            x, attn_map = head(x)
            attn_maps.append(attn_map)
            outputs.append(x)
        
        # Concatenate all outputs
        # concat_outputs = torch.cat(outputs, dim = -1)
        concat_outputs = torch.cat(outputs, dim=-1)
        final_output = self.proj(concat_outputs) # another layer
        
        return final_output, attn_maps