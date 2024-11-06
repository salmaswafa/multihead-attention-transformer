import torch.nn as nn
import globals

from feed_forward import FeedForward
from multihead_attention import MultiHeadAttention


class Block(nn.Module):
    def __init__(self, n_embd, num_heads, applyMasking = False):
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads = num_heads, head_size = head_size, d_model = n_embd, applyMasking = applyMasking)
        # 
        self.ffwd = FeedForward(d_model = n_embd, hidden_size=globals.n_hidden)
        # TODO: understand why and where this happens
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # normalize input
        x_ln1 = self.ln1(x)
        # send to multi-head attention
        sa_out, attn_maps = self.sa(x_ln1)
        # TODO: WHY ARE WE ADDING BELOW? Diagram says "Add & Norm" after multi-head and after ffwd - but why?
        # TODO: UNDERSTAND THE PART ABOUT: FEEDFORWARD LAYERS APPLIED TO EACH TOKEN INDIVIDUALLY
        # TODO: WHAT DOES THIS FFWD DO?
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, attn_maps
        
        