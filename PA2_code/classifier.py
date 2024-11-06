# models.py

import torch
from torch import nn
import torch.nn.functional as F
# from sklearn.feature_extraction.text import CountVectorizer
# from sentiment_data import WordEmbeddings, read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset
import globals 
import numpy as np
from transformer import EncoderModel

# TODO: assignment says 1 layer but there is a hidden size in the globals
# add relu and another layer
# Two-layer fully connected neural network
class NN1DAN(nn.Module):
    def __init__(self, input_size, tokenizer):
        super().__init__()
        # self.embeddings = embs.get_initialized_embedding_layer(frozen=isFrozen)
        # [16,300] - since they are batches of 16 and the vector dimension is 300
        self.encoder = EncoderModel(vocab_size = tokenizer.vocab_size, n_embd = globals.n_embd, block_size = globals.block_size, num_heads = globals.n_head, num_layers = globals.n_layer)
        
        # print the number of parameters in the model
        print(f'Encoder parameters: {sum(p.numel() for p in self.encoder.parameters())}')
        
        self.fc1 = nn.Linear(input_size, 3)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # convert input into long and get embeddings
        # x = x.long()
        # x = self.embeddings(x)
        
        # input to encoder: indices of words in batch
        x, attn_maps = self.encoder(x)
        # output from encoder is the heavily processs, contextualized version of the word vectors (input tokens)
        # Input and output shapes are identical
        
        # To provide the embeddings to the classifier, use the mean of the embeddings across the sequence dimension
        # get the mean of all words in every sentence
        x = torch.mean(x, dim=1)

        # send through the network
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        x = self.log_softmax(x)
        return x, attn_maps