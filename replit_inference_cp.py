import logging
import transformers
import sys
logging.basicConfig(format='%(message)s')
log = logging.getLogger(__name__)
import os
import deepspeed
import sys
sys.stderr.write("*************************************************************************")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import time
import json
import uuid
from argparse import ArgumentParser
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import termcolor
import torch
import transformers
from flask import jsonify
import kserve
from typing import Dict
#from blocks import MPTBlock
#from .attention import ATTN_CLASS_REGISTRY

from dotenv import load_dotenv
import math
import warnings
from typing import Optional
import torch
import torch.nn as nn
from einops import rearrange
from torch import nn

load_dotenv()
HF_ACCESS_TOKEN = os.getenv('HF_ACCESS_TOKEN')

API_KEY = "your_api_key_here"
global world_size
world_size = int(os.getenv('WORLD_SIZE', '4'))
#deepspeed.init_distributed(dist_backend="mpi")

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
#from .attention import ATTN_CLASS_REGISTRY
def _cast_if_autocast_enabled(tensor):
    if torch.is_autocast_enabled():
        if tensor.device.type == 'cuda':
            dtype = torch.get_autocast_gpu_dtype()
        elif tensor.device.type == 'cpu':
            dtype = torch.get_autocast_cpu_dtype()
        else:
            raise NotImplementedError()
        return tensor.to(dtype=dtype)
    return tensor

class LPLayerNorm(torch.nn.LayerNorm):

    def __init__(self, normalized_shape, eps=1e-05, elementwise_affine=True, device=None, dtype=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine, device=device, dtype=dtype)

    def forward(self, x):
        module_device = x.device
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        downcast_bias = _cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
        with torch.autocast(enabled=False, device_type=module_device.type):
            return torch.nn.functional.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)

def rms_norm(x, weight=None, eps=1e-05):
    output = x / torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    if weight is not None:
        return output * weight
    return output


class RMSNorm(torch.nn.Module):

    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__()
        self.eps = eps
        if weight:
            self.weight = torch.nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
        else:
            self.register_parameter('weight', None)

    def forward(self, x):
        return rms_norm(x.float(), self.weight, self.eps).to(dtype=x.dtype)

class LPRMSNorm(RMSNorm):

    def __init__(self, normalized_shape, eps=1e-05, weight=True, dtype=None, device=None):
        super().__init__(normalized_shape=normalized_shape, eps=eps, weight=weight, dtype=dtype, device=device)

    def forward(self, x):
        downcast_x = _cast_if_autocast_enabled(x)
        downcast_weight = _cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
        with torch.autocast(enabled=False, device_type=x.device.type):
            return rms_norm(downcast_x, downcast_weight, self.eps).to(dtype=x.dtype)



#from .norm import NORM_CLASS_REGISTRY
NORM_CLASS_REGISTRY = {'layernorm': torch.nn.LayerNorm, 'low_precision_layernorm': LPLayerNorm, 'rmsnorm': RMSNorm, 'low_precision_rmsnorm': LPRMSNorm}
class MPTMLP(nn.Module):

    def __init__(self, d_model: int, expansion_ratio: int, device: Optional[str]=None):
        super().__init__()
        self.up_proj = nn.Linear(d_model, expansion_ratio * d_model, device=device)
        self.act = nn.GELU(approximate='none')
        self.down_proj = nn.Linear(expansion_ratio * d_model, d_model, device=device)
        self.down_proj._is_residual = True

    def forward(self, x):
        return self.down_proj(self.act(self.up_proj(x)))
    

class MPTBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, expansion_ratio: int, attn_config: Dict={'attn_type': 'multihead_attention', 'attn_pdrop': 0.0, 'attn_impl': 'triton', 'qk_ln': False, 'clip_qkv': None, 'softmax_scale': None, 'prefix_lm': False, 'attn_uses_sequence_id': False, 'alibi': False, 'alibi_bias_max': 8}, resid_pdrop: float=0.0, norm_type: str='low_precision_layernorm', verbose: int=0, device: Optional[str]=None, **kwargs):
        del kwargs
        super().__init__()
        norm_class = NORM_CLASS_REGISTRY[norm_type.lower()]
        attn_class = ATTN_CLASS_REGISTRY[attn_config['attn_type']]
        self.norm_1 = norm_class(d_model, device=device)
        self.attn = attn_class(attn_impl=attn_config['attn_impl'], clip_qkv=attn_config['clip_qkv'], qk_ln=attn_config['qk_ln'], softmax_scale=attn_config['softmax_scale'], attn_pdrop=attn_config['attn_pdrop'], d_model=d_model, n_heads=n_heads, verbose=verbose, device=device)
        self.norm_2 = norm_class(d_model, device=device)
        self.ffn = MPTMLP(d_model=d_model, expansion_ratio=expansion_ratio, device=device)
        self.resid_attn_dropout = nn.Dropout(resid_pdrop)
        self.resid_ffn_dropout = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor, past_key_value: Optional[Tuple[torch.Tensor]]=None, attn_bias: Optional[torch.Tensor]=None, attention_mask: Optional[torch.ByteTensor]=None, is_causal: bool=True) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        a = self.norm_1(x)
        (b, _, past_key_value) = self.attn(a, past_key_value=past_key_value, attn_bias=attn_bias, attention_mask=attention_mask, is_causal=is_causal)
        x = x + self.resid_attn_dropout(b)
        m = self.norm_2(x)
        n = self.ffn(m)
        x = x + self.resid_ffn_dropout(n)
        return (x, past_key_value)


def _reset_is_causal(num_query_tokens: int, num_key_tokens: int, original_is_causal: bool):
    if original_is_causal and num_query_tokens != num_key_tokens:
        if num_query_tokens != 1:
            raise NotImplementedError('MPT does not support query and key with different number of tokens, unless number of query tokens is 1.')
        else:
            return False
    return original_is_causal

def scaled_multihead_dot_product_attention(query, key, value, n_heads, softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False):
    q = rearrange(query, 'b s (h d) -> b h s d', h=n_heads)
    k = rearrange(key, 'b s (h d) -> b h d s', h=1 if multiquery else n_heads)
    v = rearrange(value, 'b s (h d) -> b h s d', h=1 if multiquery else n_heads)
    min_val = torch.finfo(q.dtype).min
    (b, _, s_q, d) = q.shape
    s_k = k.size(-1)
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(d)
    attn_weight = q.matmul(k) * softmax_scale
    if attn_bias is not None:
        if attn_bias.size(-1) != 1 and attn_bias.size(-1) != s_k or (attn_bias.size(-2) != 1 and attn_bias.size(-2) != s_q):
            raise RuntimeError(f'attn_bias (shape: {attn_bias.shape}) is expected to broadcast to shape: {attn_weight.shape}.')
        attn_weight = attn_weight + attn_bias
    if key_padding_mask is not None:
        if attn_bias is not None:
            warnings.warn('Propogating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unneccessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        attn_weight = attn_weight.masked_fill(~key_padding_mask.view((b, 1, 1, s_k)), min_val)
    if is_causal:
        s = max(s_q, s_k)
        causal_mask = attn_weight.new_ones(s, s, dtype=torch.float16)
        causal_mask = causal_mask.tril()
        causal_mask = causal_mask.to(torch.bool)
        causal_mask = ~causal_mask
        causal_mask = causal_mask[-s_q:, -s_k:]
        attn_weight = attn_weight.masked_fill(causal_mask.view(1, 1, s_q, s_k), min_val)
    attn_weight = torch.softmax(attn_weight, dim=-1)
    if dropout_p:
        attn_weight = torch.nn.functional.dropout(attn_weight, p=dropout_p, training=training, inplace=True)
    out = attn_weight.matmul(v)
    out = rearrange(out, 'b h s d -> b s (h d)')
    if needs_weights:
        return (out, attn_weight)
    return (out, None)

def check_valid_inputs(*tensors, valid_dtypes=[torch.float16, torch.bfloat16]):
    for tensor in tensors:
        if tensor.dtype not in valid_dtypes:
            raise TypeError(f'tensor.dtype={tensor.dtype!r} must be in valid_dtypes={valid_dtypes!r}.')
        if not tensor.is_cuda:
            raise TypeError(f'Inputs must be cuda tensors (tensor.is_cuda={tensor.is_cuda!r}).')

def flash_attn_fn(query, key, value, n_heads, softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False):
    try:
        from flash_attn import bert_padding, flash_attn_interface
    except:
        raise RuntimeError('Please install flash-attn==1.0.3.post0')
    check_valid_inputs(query, key, value)
    if attn_bias is not None:
        raise NotImplementedError(f'attn_bias not implemented for flash attn.')
    (batch_size, seqlen) = query.shape[:2]
    if key_padding_mask is None:
        key_padding_mask = torch.ones_like(key[:, :, 0], dtype=torch.bool)
    query_padding_mask = key_padding_mask[:, -query.size(1):]
    (query_unpad, indices_q, cu_seqlens_q, max_seqlen_q) = bert_padding.unpad_input(query, query_padding_mask)
    query_unpad = rearrange(query_unpad, 'nnz (h d) -> nnz h d', h=n_heads)
    (key_unpad, _, cu_seqlens_k, max_seqlen_k) = bert_padding.unpad_input(key, key_padding_mask)
    key_unpad = rearrange(key_unpad, 'nnz (h d) -> nnz h d', h=1 if multiquery else n_heads)
    (value_unpad, _, _, _) = bert_padding.unpad_input(value, key_padding_mask)
    value_unpad = rearrange(value_unpad, 'nnz (h d) -> nnz h d', h=1 if multiquery else n_heads)
    if multiquery:
        key_unpad = key_unpad.expand(key_unpad.size(0), n_heads, key_unpad.size(-1))
        value_unpad = value_unpad.expand(value_unpad.size(0), n_heads, value_unpad.size(-1))
    dropout_p = dropout_p if training else 0.0
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    output_unpad = flash_attn_interface.flash_attn_unpadded_func(query_unpad, key_unpad, value_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale=softmax_scale, causal=reset_is_causal, return_attn_probs=needs_weights)
    output = bert_padding.pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'), indices_q, batch_size, seqlen)
    return (output, None)

def triton_flash_attn_fn(query, key, value, n_heads, softmax_scale=None, attn_bias=None, key_padding_mask=None, is_causal=False, dropout_p=0.0, training=False, needs_weights=False, multiquery=False):
    try:
        from flash_attn import flash_attn_triton
    except:
        raise RuntimeError('Please install flash-attn==1.0.3.post0 and triton==2.0.0.dev20221202')
    check_valid_inputs(query, key, value)
    if dropout_p:
        raise NotImplementedError(f'Dropout not implemented for attn_impl: triton.')
    if needs_weights:
        raise NotImplementedError(f'attn_impl: triton cannot return attn weights.')
    if key_padding_mask is not None:
        warnings.warn('Propagating key_padding_mask to the attention module ' + 'and applying it within the attention module can cause ' + 'unnecessary computation/memory usage. Consider integrating ' + 'into attn_bias once and passing that to each attention ' + 'module instead.')
        (b_size, s_k) = key_padding_mask.shape[:2]
        if attn_bias is None:
            attn_bias = query.new_zeros(b_size, 1, 1, s_k)
        attn_bias = attn_bias.masked_fill(~key_padding_mask.view((b_size, 1, 1, s_k)), torch.finfo(query.dtype).min)
    query = rearrange(query, 'b s (h d) -> b s h d', h=n_heads)
    key = rearrange(key, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)
    value = rearrange(value, 'b s (h d) -> b s h d', h=1 if multiquery else n_heads)
    if multiquery:
        key = key.expand(*key.shape[:2], n_heads, key.size(-1))
        value = value.expand(*value.shape[:2], n_heads, value.size(-1))
    reset_is_causal = _reset_is_causal(query.size(1), key.size(1), is_causal)
    attn_output = flash_attn_triton.flash_attn_func(query, key, value, attn_bias, reset_is_causal, softmax_scale)
    output = attn_output.view(*attn_output.shape[:2], -1)
    return (output, None)



class MultiheadAttention(nn.Module):
    """Multi-head self attention.
    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(self, d_model: int, n_heads: int, attn_impl: str='triton', clip_qkv: Optional[float]=None, qk_ln: bool=False, softmax_scale: Optional[float]=None, attn_pdrop: float=0.0, low_precision_layernorm: bool=False, verbose: int=0, device: Optional[str]=None):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.d_model / self.n_heads)
        self.attn_dropout_p = attn_pdrop
        self.Wqkv = nn.Linear(self.d_model, 3 * self.d_model, device=device)
        fuse_splits = (d_model, 2 * d_model)
        self.Wqkv._fused = (0, fuse_splits)
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(self.d_model, device=device)
            self.k_ln = layernorm_class(self.d_model, device=device)
        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn('While `attn_impl: triton` can be faster than `attn_impl: flash` ' + 'it uses more memory. When training larger models this can trigger ' + 'alloc retries which hurts performance. If encountered, we recommend ' + 'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn('Using `attn_impl: torch`. If your model does not use `alibi` or ' + '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' + 'we recommend using `attn_impl: triton`.')
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        (query, key, value) = qkv.chunk(3, dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        if past_key_value is not None:
            if len(past_key_value) != 0:
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)
            past_key_value = (key, value)
        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]
        (context, attn_weights) = self.attn_fn(query, key, value, self.n_heads, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights)
        return (self.out_proj(context), attn_weights, past_key_value)

class MultiQueryAttention(nn.Module):
    """Multi-Query self attention.
    Using torch or triton attention implemetation enables user to also use
    additive bias.
    """

    def __init__(self, d_model: int, n_heads: int, attn_impl: str='triton', clip_qkv: Optional[float]=None, qk_ln: bool=False, softmax_scale: Optional[float]=None, attn_pdrop: float=0.0, low_precision_layernorm: bool=False, verbose: int=0, device: Optional[str]=None):
        super().__init__()
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.qk_ln = qk_ln
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.softmax_scale = softmax_scale
        if self.softmax_scale is None:
            self.softmax_scale = 1 / math.sqrt(self.head_dim)
        self.attn_dropout_p = attn_pdrop
        self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim, device=device)
        fuse_splits = (d_model, d_model + self.head_dim)
        self.Wqkv._fused = (0, fuse_splits)
        if self.qk_ln:
            layernorm_class = LPLayerNorm if low_precision_layernorm else nn.LayerNorm
            self.q_ln = layernorm_class(d_model, device=device)
            self.k_ln = layernorm_class(self.head_dim, device=device)
        if self.attn_impl == 'flash':
            self.attn_fn = flash_attn_fn
        elif self.attn_impl == 'triton':
            self.attn_fn = triton_flash_attn_fn
            if verbose:
                warnings.warn('While `attn_impl: triton` can be faster than `attn_impl: flash` ' + 'it uses more memory. When training larger models this can trigger ' + 'alloc retries which hurts performance. If encountered, we recommend ' + 'using `attn_impl: flash` if your model does not use `alibi` or `prefix_lm`.')
        elif self.attn_impl == 'torch':
            self.attn_fn = scaled_multihead_dot_product_attention
            if torch.cuda.is_available() and verbose:
                warnings.warn('Using `attn_impl: torch`. If your model does not use `alibi` or ' + '`prefix_lm` we recommend using `attn_impl: flash` otherwise ' + 'we recommend using `attn_impl: triton`.')
        else:
            raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, x, past_key_value=None, attn_bias=None, attention_mask=None, is_causal=True, needs_weights=False):
        qkv = self.Wqkv(x)
        if self.clip_qkv:
            qkv.clamp_(min=-self.clip_qkv, max=self.clip_qkv)
        (query, key, value) = qkv.split([self.d_model, self.head_dim, self.head_dim], dim=2)
        key_padding_mask = attention_mask
        if self.qk_ln:
            dtype = query.dtype
            query = self.q_ln(query).to(dtype)
            key = self.k_ln(key).to(dtype)
        if past_key_value is not None:
            if len(past_key_value) != 0:
                key = torch.cat([past_key_value[0], key], dim=1)
                value = torch.cat([past_key_value[1], value], dim=1)
            past_key_value = (key, value)
        if attn_bias is not None:
            attn_bias = attn_bias[:, :, -query.size(1):, -key.size(1):]
        (context, attn_weights) = self.attn_fn(query, key, value, self.n_heads, softmax_scale=self.softmax_scale, attn_bias=attn_bias, key_padding_mask=key_padding_mask, is_causal=is_causal, dropout_p=self.attn_dropout_p, training=self.training, needs_weights=needs_weights, multiquery=True)
        return (self.out_proj(context), attn_weights, past_key_value)
    


def attn_bias_shape(attn_impl, n_heads, seq_len, alibi, prefix_lm, causal, use_sequence_id):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            if (prefix_lm or not causal) or use_sequence_id:
                return (1, n_heads, seq_len, seq_len)
            return (1, n_heads, 1, seq_len)
        elif prefix_lm or use_sequence_id:
            return (1, 1, seq_len, seq_len)
        return None
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')

def build_attn_bias(attn_impl, attn_bias, n_heads, seq_len, causal=False, alibi=False, alibi_bias_max=8):
    if attn_impl == 'flash':
        return None
    elif attn_impl in ['torch', 'triton']:
        if alibi:
            (device, dtype) = (attn_bias.device, attn_bias.dtype)
            attn_bias = attn_bias.add(build_alibi_bias(n_heads, seq_len, full=not causal, alibi_bias_max=alibi_bias_max, device=device, dtype=dtype))
        return attn_bias
    else:
        raise ValueError(f'attn_impl={attn_impl!r} is an invalid setting.')

def gen_slopes(n_heads, alibi_bias_max=8, device=None):
    _n_heads = 2 ** math.ceil(math.log2(n_heads))
    m = torch.arange(1, _n_heads + 1, dtype=torch.float32, device=device)
    m = m.mul(alibi_bias_max / _n_heads)
    slopes = 1.0 / torch.pow(2, m)
    if _n_heads != n_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:n_heads]
    return slopes.view(1, n_heads, 1, 1)

def build_alibi_bias(n_heads, seq_len, full=False, alibi_bias_max=8, device=None, dtype=None):
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, 1, seq_len)
    if full:
        alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.int32, device=device).view(1, 1, seq_len, 1)
        alibi_bias = alibi_bias.abs().mul(-1)
    slopes = gen_slopes(n_heads, alibi_bias_max, device=device)
    alibi_bias = alibi_bias * slopes
    return alibi_bias.to(dtype=dtype)

ATTN_CLASS_REGISTRY = {'multihead_attention': MultiheadAttention, 'multiquery_attention': MultiQueryAttention}

def disable_torch_init():
    def noop(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop
    transformers.modeling_utils._init_weights = False

def get_replit(model, trust_remote_code=True):
    #config = AutoConfig.from_pretrained("replit/replit-code-v1-3b", trust_remote_code=True)
    #config.attn_config['attn_impl'] = 'triton'
    model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=HF_ACCESS_TOKEN, trust_remote_code=trust_remote_code)
    model = model.to("cuda", dtype=torch.float16)
    model.eval()
    model.seqlen = 2048
    print("model dtype:")
    print(model.dtype)
    #config = {
    #    "kernel_inject": True,
    #   "tensor_parallel": {"tp_size": 2},
    #   "dtype": "fp16",
    #   "enable_cuda_graph" : False
    #    }
    #world_size = int(os.getenv('WORLD_SIZE', '2'))
    #model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float, replace_with_kernel_inject=True)

    return model

def generate_text(tokenizer, model, prompt, max_tokens, n, temperature, stop):
    batch = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    batch = {k: v.cuda() for k, v in batch.items()}
    generated_texts = []
    for _ in range(n):
        generated = model.module.generate(batch["input_ids"].to("cuda"), do_sample=True, min_new_tokens=max_tokens, max_new_tokens=max_tokens, temperature=temperature, eos_token_id=tokenizer.encode(stop)[0] if stop else None)
        generated_texts.append(tokenizer.decode(generated[0]))
    return generated_texts


class KServeModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        self.ready = True

    def predict(self, request: Dict, headers: Dict[str, str] = None) -> Dict:
        api_key = headers.get("authorization")
        print(api_key)
        if api_key != f"Bearer {API_KEY}":
            return {"error": "Invalid API key"}

        instances = request.get("instances", [])
        if not instances or not all(isinstance(instance, dict) for instance in instances):
            return jsonify({"error": "Invalid input parameters"})
        st = time.time()
        generated_texts = []
        for instance in instances:
            prompt = instance.get("prompt", "")
            max_tokens = instance.get("max_tokens", 1024)
            n = instance.get("n", 1)
            temperature = instance.get("temperature", 0.7)
            stop = instance.get("stop", None)

            if not isinstance(prompt, str) or not isinstance(max_tokens, int) or not isinstance(n, int) or not isinstance(temperature, (int, float)) or (stop is not None and not isinstance(stop, str)):
                return jsonify({"error": "Invalid input parameters"}), 400
            generated_texts.extend(generate_text(tokenizer, model, prompt, max_tokens, n, temperature, stop))
        et = time.time()
        print(f"inference time is : {et-st}")
        return {"inference time" : et-st ,"predictions": [{"text": text} for text in generated_texts]}
    

def main():
    global model, tokenizer
    #local_rank = int(os.getenv('LOCAL_RANK', '0'))
    #world_size = int(os.getenv('WORLD_SIZE', '2'))
    #parser = ArgumentParser()
    #parser.add_argument("model", type=str, help="model to load, such as replit-code-v1-3b")
    model_path = r"/mnt/pvc/replit-code-v1"

    print("-- Loading original model weights from PVC (this will take a few seconds)")
    start = time.time()
    load = os.path.join("/mnt/pvc/replit-code-v1/", "model.bin")
    config = {
        "kernel_inject": False,
        "tensor_parallel": {"tp_size": 4},
        "dtype": "fp16",
        "enable_cuda_graph" : False,
        "injection_policy" : {MPTBlock: ('attn.out_proj', 'ffn.down_proj')}
        }
    disable_torch_init()
    num_of_gpus = torch.cuda.device_count()
    print(num_of_gpus)
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)
    print("-- Loading model weights from PVC (this will take a few seconds)")
    #os.chdir(r"/app/replit-code-v1-3b")
    #model = get_replit("replit/replit-code-v1-3b")
    model = get_replit(model_path)
    model = deepspeed.init_inference(model=model, config=config)
    print(dir(model))
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
    print(dir(model.injection_dict))
    print("_config")
    print(model._config)
    print("_apply_injection_policy")
    print(model._apply_injection_policy)
    print("named modules : ")
    print(model.named_modules)
    print("-------------------------------------------------------------------------")
    print("named_parameters")
    print(model.named_parameters)
    print("-------------------------------------------------------------------------")
    print("model.module:")
    print(model.module)
    print("-------------------------------------------------------------------------")
    print("model.injection_dict")
    print(model.injection_dict)
    print("-------------------------------------------------------------------------")
    print("model.inference_mp_group")
    print(model.inference_mp_group)
    print("-------------------------------------------------------------------------")
    print("model.mpu")
    print(model.mpu)
    print("-------------------------------------------------------------------------")
    print("model.mp_group")

    print(model.mp_group)
    print("-------------------------------------------------------------------------")    
    print("model.config")
    print(model.config)

    #model = deepspeed.init_inference(model, mp_size=world_size, dtype=torch.float16, injection_policy={MPTBlock: ('attn.out_proj.weight', 'ffn.down_proj.weight')})
    #model = deepspeed.init_inference(model, mp_size=world_size, replace_with_kernel_inject=False, replace_method="auto")
    print("-- Loading tokenize model")
    tokenizer = AutoTokenizer.from_pretrained(model_path, torch_dtype=torch.float16, use_auth_token=HF_ACCESS_TOKEN, trust_remote_code=True)
    #model = deepspeed.init_inference(model=model, dtype=torch.float16, replace_with_kernel_inject=True)
    end = time.time()
    print("----------------------------------------------------")
    print(f"Time for loading the model: {end-start}")
    kserve_model = KServeModel("replit-code-v1")
    s = time.time()
    print("-- Ready to serve inference requests :)")
    kserve.ModelServer().start([kserve_model])
    e = time.time()
    print(f"time for inference: {e-s}")

if __name__ == "__main__":
    main()
