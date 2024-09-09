import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

import torch
import torch.nn as nn
import numpy as np

import torch

class ResidualNorm(nn.Module):
    def __init__ (self, size, dropout):
        super(ResidualNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))

class MLP(nn.Module):
    def __init__(self, model_depth, ff_depth, dropout):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(model_depth, ff_depth)
        self.w2 = nn.Linear(ff_depth, model_depth)
        self.dropout = nn.Dropout(dropout)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.w2(self.dropout(self.silu(self.w1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return x

################################################################
# attention

class CrossAttention(nn.Module):
    def __init__(self, query_input_dim, key_input_dim, output_dim):
        super(CrossAttention, self).__init__()
        # TODO: multi-head cross attention
        
        self.out_dim = output_dim
        self.W_Q = nn.Linear(query_input_dim, output_dim)
        self.W_K = nn.Linear(key_input_dim, output_dim)
        self.W_V = nn.Linear(key_input_dim, output_dim)
        self.scale_val = self.out_dim ** 0.5
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query_input, key_input, value_input, query_input_mask=None, key_input_mask=None):
        query = self.W_Q(query_input)
        key = self.W_K(key_input)
        value = self.W_V(value_input)

        attn_weights = torch.matmul(query, key.transpose(1, 2)) / self.scale_val
        attn_mask = query_input_mask.unsqueeze(-1) * key_input_mask.unsqueeze(-1).transpose(1, 2)
        attn_weights = attn_weights.masked_fill(attn_mask == False, -1e9)
        attn_weights = self.softmax(attn_weights)
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights
        

class MultiHeadAttention (nn.Module):
    def __init__ (self, 
                  num_heads, 
                  embed_dim, 
                  bias=False
                 ):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dk = embed_dim//num_heads
        self.WQ = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.WK = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.WV = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.WO = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward (self, x, kv, mask=None):
        batch_size = x.size(0)
        Q = self.WQ(x ).view(batch_size, -1, self.num_heads, self.dk).transpose(1,2)
        K = self.WK(kv).view(batch_size, -1, self.num_heads, self.dk).transpose(1,2)
        V = self.WV(kv).view(batch_size, -1, self.num_heads, self.dk).transpose(1,2)

        if mask is not None:
            if len(mask.shape) == 2:
                mask = torch.einsum('bi,bj->bij', mask, mask)
        x = attention(Q, K, V, mask=mask)

        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads*self.dk)
        return self.WO(x)

def attention (Q,K,V, mask=None):
    dk = Q.size(-1)
    T = (Q @ K.transpose(-2, -1))/math.sqrt(dk)
    # print(f'T.shape: {T.shape}')
    # print(f'mask.shape: {mask.shape}')
    if mask is not None:
        T = T.masked_fill_(mask.unsqueeze(1)==0, -1e9)
    T = F.softmax(T, dim=-1)
    return T @ V


################################################################
# encoder

class Encoder(nn.Module):
    def __init__ (self, 
                  n_layers, 
                  n_heads, 
                  model_depth, 
                  ff_depth, 
                  dropout
                 ):
        super(Encoder, self).__init__()
        
        self.layers = nn.ModuleList([EncoderLayer(n_heads, model_depth, ff_depth, dropout) for i in range(n_layers)])
        self.lnorm = LayerNorm(model_depth)

    def forward (self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.lnorm(x)


class EncoderLayer (nn.Module):
    def __init__ (self, 
                  n_heads, 
                  model_depth, 
                  ff_depth, 
                  dropout=0.0
                 ):
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(embed_dim=model_depth, num_heads=n_heads)
        self.resnorm1 = ResidualNorm(model_depth, dropout)
        self.ff = MLP(model_depth, ff_depth, dropout)
        self.resnorm2 = ResidualNorm(model_depth, dropout)

    def forward (self, x, mask):
        x = self.resnorm1(x, lambda arg: self.self_attn(arg, arg, mask))
        x = self.resnorm2(x, self.ff)
        return x

################################################################
# embedder

class Embedding(nn.Module):
    def __init__(self, vocab_size, model_depth):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, model_depth)
        self.model_depth = model_depth
        self.positional = PositionalEncoding(model_depth)

    def forward(self, x):
        emb = self.lut(x) * math.sqrt(self.model_depth)
        return self.positional(emb)

class PositionalEncoding(nn.Module):
    def __init__(self, model_depth, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, model_depth)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, model_depth, 2) *
                             -(math.log(10000.0) / model_depth))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).unsqueeze(0)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        # pe.shape: (1, 1, 5000, model_depth)

    def forward(self, x):
        return x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        # return x + Variable(self.pe[:, :, :x.size(2), :], requires_grad=False)

################################################################
# transformer

class Generator (nn.Module):
    def __init__(self, 
                 model_depth, 
                 vocab_size
                ):
        super(Generator, self).__init__()
        self.ff = nn.Linear(model_depth, vocab_size)

    def forward(self, x):
        return F.log_softmax(self.ff(x), dim=-1)


class MSATransformer(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 n_layers=2, 
                 n_heads=4, 
                 model_depth=64, 
                 ff_depth=64, 
                 dropout=0.0,
                ):
        super(MSATransformer, self).__init__()
        
        self.model_depth = model_depth
        self.encoder = Encoder(n_layers=n_layers, 
                               n_heads=n_heads, 
                               model_depth=model_depth, 
                               ff_depth=ff_depth, 
                               dropout=dropout,
                              )
        if vocab_size is not None:
            if isinstance(vocab_size, int):
                self.set_vocab_size(vocab_size)
            else:
                self.set_vocab_size(vocab_size[0], vocab_size[1])

    def set_vocab_size(self, src_vocab_size):
        self.src_embedder = Embedding(src_vocab_size, self.model_depth)
        self.generator = Generator(self.model_depth, src_vocab_size)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_mask=None):
        enc_out = self.encoder(self.src_embedder(src), src_mask)
        return enc_out
