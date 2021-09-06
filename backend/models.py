import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle

import os

import numpy  as np

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, padding_idx):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.lstm      = nn.LSTM(emb_size, hidden_size, batch_first=True)
        
    def forward(self, batch_words):
        '''
        Inputs:
            batch_words: (batch x source_len)
        '''
        
        #(batch x source_len) -> (batch x source_len x emb_size)
        embedded = self.embedding(batch_words)
        
        output, hidden = self.lstm(embedded)
        return output
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        
    def forward(self, hidden, encoder_output, encoder_mask):
        '''
        Inputs:
            hidden: (batch x hidden_size)
            encoder_output: (batch x source_len x hidden_size)
        '''
        hidden = self.linear(hidden)
        hidden = hidden.unsqueeze(2)
        alphas = encoder_output.matmul(hidden)
        
        if encoder_mask is not None:
            alphas[encoder_mask] = -1e16
            
        scores = F.softmax(alphas, dim=1)
        c = (scores * encoder_output).sum(dim=1)
        return c
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, padding_idx):
        super(Decoder, self).__init__()
        
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.attn      = Attention(hidden_size)
        self.gru_cell  = nn.GRUCell(emb_size + hidden_size, hidden_size)
        self.linear    = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, batch_trans_in, encoder_output, hidden, encoder_mask=None):
        '''
        Inputs:
            batch_trans_in: (batch x target_len)
            encoder_output: (batch x source_len x hidden_size)
            hidden: (batch x hidden_size)
        '''
        embedded  = self.embedding(batch_trans_in)
        timesteps = embedded.size(1)
        
        output = []
        
        for t in range(timesteps):
            x = embedded[:, t]
            c = self.attn(hidden, encoder_output, encoder_mask)
            inp = torch.cat([x, c], dim=1)
            hidden = self.gru_cell(inp, hidden)
            output.append(hidden)
        
        output = torch.stack(output, dim=1)
        logits = self.linear(output)
        return logits.view(-1, self.vocab_size), hidden