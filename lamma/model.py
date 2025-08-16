import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # query heads
    n_kv_heads: Optional[int] = None # key-value heads, if None, use n_heads
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    
    # Needed for kv cache
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    device: str = None # Device to run the model on, e.g., 'cuda' or 'cpu'
    

def precompute_theta_pos_frenquencies(
    head_dim: int,
    seq_len: int,
    device: str,
    theta: float = 10000.0
) -> torch.Tensor:
    # As write in the paper, the dimension of the embedding must be even
    assert head_dim %2 == 0, 'Dimensions must be divisible by 2'
    # build the theta parameters
    # according to the formula theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,3...,dim/2]
    # Shape (head_dim/2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # SHape (head_dim/2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the m parameter in the paper)
    # Shape (seq_len)
    m = torch.arange(seq_len).float().to(device)
    # Multiply each theta by each position using the outer product
    # Shape (Seq_len) outer_product* (Head_dim/2) -> (seq_len, head_dim/2)
    freqs = torch.outer(m, theta).float()
    # we can compute complex numbers in the polar form c = R * exp(i * m * theta) where R = 1 as follows:
    # (seq_len, head_dim/2) -> (seq_len, head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotaty_embedding(
    x: torch.Tensor,
    freqs_complex: torch.Tensor,
    device: str
) -> torch.Tensor:
    # x: (bs, seq_len, H, head_dim) -> (bs, seq_len, H, head_dim/2, 2)
    # view_as_complex: (bs, seq_len, H, head_dim/2, 2) -> (bs, seq_len, H, head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (seq_len, head_dim/2) -> (1, seq_len, 1,head_dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (bs, seq_len, H, head_dim/2) * (1, seq_len, 1,head_dim/2) -> (bs, seq_len, H, head_dim/2)
    x_rotated = x_complex * freqs_complex
    # (bs, seq_len, H, head_dim/2) -> (bs, seq_len, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (bs, seq_len, H, head_dim/2, 2) -> (bs, seq_len, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).device(device)


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # (dim), the gamma parameter for scaling
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x: torch.Tensor):
        # x.pow(2).mean(-1, keepdim=True): (bs, seq_len, dim) -> (bs, seq_len, 1)
        # x: (bs, seq_len, dim) * (bs, seq_len, 1) -> (bs, seq_len, dim)
        # rsqrt: 1/sqrt(x)
        # pow: x^y
        # mean: mean of x along the last dimension
        # keepdim: keep the last dimension
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.weight * self._norm(x.float()).type_as(x)
    
class Transformer(nn.Module):
    
    def __init__(self, model_args: ModelArgs):
        super().__init__()
        self.args = model_args
        self.vocab_size = model_args.vocab_size
        self.nlayers = model_args.n_layers
        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)
        
        # EncoderBlock 后续实现，先实现大逻辑
        self.layers = nn.ModuleList([EncoderBlock(model_args) for _ in range(self.nlayers)])
        # RMSNorm 后面实现
        self.norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)
        
        # 旋转编码后面实现
        self.freqs_complex = precompute_theta_pos_frenquencies(
            self.args.dim//self.args.n_heads,
            self.args.max_seq_len * 2,
            device = self.args.device
        )
    
    def forward(self, token: torch.Tensor, start_pos: int) -> torch.Tensor:
        
        #(bs, seq_len)
        batch_size, seq_len = token.shape
        assert seq_len == 1, "Only one token at a time can be processed"
        
        #(bs, seq_len) -> (bs, seq_len, dim)
        h = self.tok_embeddings(token)  # (bs, seq_len, dim)
        
        # retrieve the pairs(m, theta) corresponding to the positions (start_pos, start_pos+seq_len)
        freqs_complex = self.freqs_complex[start_pos:start_pos+seq_len]
        
        # apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output