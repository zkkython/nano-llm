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