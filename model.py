"""Small character Transformer, following Karpathy's GPT lecture architecture.

Study reference: https://github.com/karpathy/ng-video-lecture
The cache is local to a request: cache[layer][head] = (K, V).
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int = 65
    context: int = 128
    width: int = 128
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        size = config.width // config.heads
        self.key = nn.Linear(config.width, size, bias=False)
        self.query = nn.Linear(config.width, size, bias=False)
        self.value = nn.Linear(config.width, size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal", torch.ones(config.context, config.context,
                                                 dtype=torch.bool).tril())

    def forward(self, x, past=None, use_cache=False):
        q, k, v = self.query(x), self.key(x), self.value(x)
        offset = 0 if past is None else past[0].size(1)
        if past is not None:
            k = torch.cat((past[0], k), dim=1)
            v = torch.cat((past[1], v), dim=1)
        scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
        # New query rows have absolute positions offset ... offset + T - 1.
        allowed = self.causal[offset:offset + x.size(1), :k.size(1)]
        weights = self.dropout(F.softmax(scores.masked_fill(~allowed, -torch.inf), dim=-1))
        return weights @ v, (k, v) if use_cache else None


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.heads)])
        self.proj = nn.Linear(config.width, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.ff = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.ReLU(),
                                nn.Linear(4 * config.width, config.width),
                                nn.Dropout(config.dropout))
        self.ln1 = nn.LayerNorm(config.width)
        self.ln2 = nn.LayerNorm(config.width)

    def forward(self, x, past=None, use_cache=False):
        normalized = self.ln1(x)
        outputs, caches = [], []
        for i, head in enumerate(self.heads):
            out, kv = head(normalized, None if past is None else past[i], use_cache)
            outputs.append(out)
            caches.append(kv)
        x = x + self.dropout(self.proj(torch.cat(outputs, dim=-1)))
        return x + self.ff(self.ln2(x)), caches if use_cache else None


class TinyGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.width % config.heads:
            raise ValueError("width must be divisible by heads")
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.width)
        self.position_embedding = nn.Embedding(config.context, config.width)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.lm_head = nn.Linear(config.width, config.vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, cache=None, use_cache=False, last_only=False):
        if cache is not None and not use_cache:
            raise ValueError("passing cache requires use_cache=True")
        if self.training and use_cache:
            raise ValueError("KV caching is for evaluation; call model.eval()")
        offset = 0 if cache is None else cache[0][0][0].size(1)
        length = tokens.size(1)
        if length == 0 or offset + length > self.config.context:
            raise ValueError("input must be nonempty and fit within the context window")
        positions = torch.arange(offset, offset + length, device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, None if cache is None else cache[i], use_cache)
            new_cache.append(kv)
        # Both inference paths project only the last position to vocabulary logits.
        x = x[:, -1:] if last_only else x
        return self.lm_head(self.norm(x)), new_cache if use_cache else None

    @torch.inference_mode()
    def generate(self, tokens, new_tokens, use_cache=True, temperature=0.0):
        if self.training:
            raise ValueError("generation requires model.eval()")
        if new_tokens < 0 or tokens.size(1) == 0 or tokens.size(1) + new_tokens > self.config.context:
            raise ValueError("prompt plus output must fit within the context window")
        cache = None
        for _ in range(new_tokens):
            current = tokens[:, -1:] if cache is not None else tokens
            logits, cache = self(current, cache, use_cache, last_only=True)
            if temperature > 0:
                next_token = torch.multinomial(F.softmax(logits[:, -1] / temperature, -1), 1)
            else:
                next_token = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


def cache_bytes(cache):
    return sum(t.numel() * t.element_size() for layer in cache for pair in layer for t in pair)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyGPT(Config(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint
