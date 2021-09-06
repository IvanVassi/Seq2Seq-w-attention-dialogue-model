import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import csv
import codecs
import numpy  as np
import re
import unicodedata
import pickle

from models import Encoder
from models import Decoder

import re

from celery import Celery

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pad_idx = 0
unk_idx = 1
sos_idx = 2
eos_idx = 3
    
# Load tokenizer for preprocessing
with open('word2id.pkl', 'rb') as f:
    word2id = pickle.load(f)
id2word = {value: key for key, value in word2id.items()}

# Load weights into new model
encoder = Encoder(len(word2id), emb_size=256, hidden_size=512, padding_idx=0).to(device)
decoder = Decoder(len(word2id), emb_size=256, hidden_size=512, padding_idx=0).to(device)
encoder.load_state_dict(torch.load('encoder.pt', map_location=torch.device('cpu')))
encoder.eval()
decoder.load_state_dict(torch.load('decoder.pt', map_location=torch.device('cpu')))
decoder.eval()

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def prepr(a):
    a = unicodeToAscii(a)
    a = normalizeString(a)
    a = a.split()
    a = [word2id.get(word, unk_idx) for word in a]
    a = torch.LongTensor(a).to(device)
    return a

def evaluation(source1):
    source1 = prepr(source1)
    seq = source1.unsqueeze(0)
    encoder_output = encoder(seq)
    
    hidden = torch.zeros(1, 512).to(device)
    
    generated = [sos_idx]
    
    for i in range(20):
        
        generated_in = torch.LongTensor([[generated[-1]]]).to(device)
        
        logit, hidden = decoder(generated_in, encoder_output, hidden)
        next_idx = logit.max(1)[1][0].item()
        generated.append(next_idx)
        
        if next_idx == eos_idx:
            break
            
    seq = [id2word.get(idx, "unk") for idx in seq[0].tolist()]
    target = [id2word.get(idx, "unk") for idx in generated if idx not in [sos_idx, eos_idx]]
    
    seq = ' '.join(seq)
    target = ' '.join(target).replace("<unk>", "well")
    
    return target


celeryapp = Celery(
    'worker', 
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)


@celeryapp.task
def predict(seq):
    result = evaluation(seq)
    return result