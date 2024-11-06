import torch.nn as nn

class FeedForward(nn.Module):
    """Feed forward netword applied after multi-head attention in the Transformer

    Args:
    """
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            # d_model = size of the embedding dimension
            # hidden_size = hidden layer dimension
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model),
            # could add another activation layer here
        )
        
    def forward(self, x):
        return self.net(x)