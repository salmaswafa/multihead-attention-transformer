import torch.nn as nn

class FeedForward(nn.Module):
    """Feed forward netword applied after multi-head attention in the Transformer

    Args:
    """
    # TODO: video has one linear and one relu only. why is there a second one here? should it have an activation after the last one?
    # nn.Linear(d_model, d_model),
    # nn.ReLU(),
    def __init__(self, d_model, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            # d_model = size of the embedding dimension
            # hidden_size = hidden layer dimension
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d_model),
        )
        
    def forward(self, x):
        return self.net(x)