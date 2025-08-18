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


def repeat_kv(x: torch.Tensor, n_sep: int) -> torch.Tensor:
    bs, seq_len, n_kv_heads, dim =x.shape
    if n_sep == 1:
        return x
    else:
        return (
            x.unsqueeze(3)
            .expand(bs, seq_len, n_kv_heads, n_sep, dim)
            .reshape(bs, seq_len, n_kv_heads*n_sep, dim)
        )


class SelfAttention(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads_q = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads_q
        self.n_heads_kv = args.n_kv_heads or args.n_heads
        self.n_rep = self.n_heads_kv // self.n_heads_q
        self.wq = nn.Linear(self.dim, self.n_heads_q * self.head_dim, bias=False)
        self.wk = nn.Linear(self.dim, self.n_heads_kv * self.head_dim, bias=False)
        self.wv = nn.Linear(self.dim, self.n_heads_kv * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads_q * self.head_dim, self.dim, bias=False)
        
        self.cache_k = torch.zeros((args.max_batch_size,
                                    args.max_seq_len, 
                                    self.n_heads_kv, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, 
                                    self.n_heads_kv, self.head_dim)
                                   ).to(args.device)
        
    # 单个token单个token的处理，所以此时的start_pos代表的是token位置索引，比如start_pos=0，则表示的是第一个token
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        # x: (bs, seq_len, dim)
        # start_pos: int, the position of the start token
        # freqs_complex: (seq_len, head_dim/2)
        bs, seq_len, _ = x.shape # (bs, 1, dim)
        # (bs, seq_len, dim) -> (bs, seq_len, n_heads_q * head_dim)
        xq = self.wq(x)
        # (bs, seq_len, n_heads_q * head_dim) -> (bs, seq_len, n_heads_q, head_dim) 
        xq = xq.view(bs, seq_len, self.n_heads_q, self.head_dim) 
        # (bs, seq_len, dim) -> (bs, seq_len, n_heads_kv, head_dim)
        xk = self.wk(x)
        # (bs, seq_len, n_heads_kv * head_dim) -> (bs, seq_len, n_heads_kv, head_dim)
        xk = xk.view(bs, seq_len, self.n_heads_kv, self.head_dim)
        # (bs, seq_len, dim) -> (bs, seq_len, n_heads_kv * head_dim)
        xv = self.wv(x)
        # (bs, seq_len, n_heads_kv * head_dim) -> (bs, seq_len, n_heads_kv, head_dim)
        xv = xv.view(bs, seq_len, self.n_heads_kv, self.head_dim)
        
        xk = apply_rotaty_embedding(xk, freqs_complex, x.device)
        xv = apply_rotaty_embedding(xv, freqs_complex, x.device)
        
        # 将增量的kv 加入到kv缓存中
        self.cache_k[:, start_pos:start_pos+seq_len] = xk
        self.cache_v[:, start_pos:start_pos+seq_len] = xv
        # 针对一个query，把历史全量的k取出来
        # (bs, history_total_seq_len, n_heads_kv, head_dim)
        keys = self.cache_k[:, 0:start_pos+seq_len]
        # (bs, history_total_seq_len, n_heads_kv, head_dim)
        values= self.cache_v[:, 0:start_pos+seq_len]
        
        # (bs, history_total_seq_len, n_heads_kv, head_dim) -> (bs, history_total_seq_len, n_heads_q, head_dim)
        keys = repeat_kv(keys, self.n_rep)
        # (bs, history_total_seq_len, n_heads_kv, head_dim) -> (bs, history_total_seq_len, n_heads_q, head_dim)
        values = repeat_kv(values, self.n_rep)
        
        # (bs, 1, n_heads_q, head_dim)  -> (bs, n_heads_q, 1, head_dim) 
        xq = xq.transpose(1, 2)
        # (bs, history_total_seq_len, n_heads_q, head_dim) -> (bs, n_heads_q, history_total_seq_len, head_dim)
        keys = keys.transpose(1, 2)
        # (bs, history_total_seq_len, n_heads_q, head_dim) -> (bs, n_heads_q, history_total_seq_len, head_dim)
        values = values.transpose(1, 2)
        # (bs, n_heads_q, 1, head_dim)  * (bs, n_heads_q, head_dim, history_total_seq_len) = (bs, n_heads_q, 1, history_total_seq_len)
        softmax = xq.matmul(keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (bs, n_heads_q, 1, history_total_seq_len) * (bs, n_heads_q, history_total_seq_len, head_dim) = (bs, n_heads_q, 1, head_dim)
        output = softmax.matmul(values)
        # (bs, n_heads_q, 1, head_dim) -> (bs, 1, n_heads_q*head_dim)
        output = output.transpose(1, 2).contiguous().view(bs, seq_len, -1)
        #(bs, 1, n_heads_q*head_dim) * (bs, seq_len, n_heads_q*head_dim, dim)   = (bs, seq_len, dim)
        return self.wo(output)
        
        
        
        
        
class EncoderBlock(nn.Module):
    
    def __init__(self, 
                 args: ModelArgs):
        super().__init__()
        self.args = args
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = self.dim // self.n_heads
        
        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)
        # norm before the self attention
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # norm after self attention, before the feed forward
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor) -> torch.Tensor:
        # x: (bs, seq_len, dim)
        # apply the norm before the self attention
        norm = self.attention_norm(x)
        # apply the self attention
        
        # (bs, seq_len, dim) + (bs, seq_len, dim)  -> (bs, seq_len, dim)
        h = x + self.attention(norm, start_pos, freqs_complex)
        # apply the norm after the self attention
        ffn_norm = self.ffn_norm(x)
        # apply the feed forward
        out = h + self.feed_forward(ffn_norm)
        return out
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