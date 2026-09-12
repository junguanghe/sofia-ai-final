# Understanding KV Caching

Accelerating a tiny Shakespeare Transformer on a laptop CPU

**Junguang He - MSCS2201 Mini Research Problem**

- Motivation: ordinary generation repeatedly computes the same history.
- Approach: train one small model, then compare generation with and without a request-local KV cache.
- Result: **1.72-2.40x faster model execution** across three prompt lengths, using **158-254 KiB** of final KV tensors.
- Correctness: maximum tested logit difference **0.0000072**. Greedy outputs matched in the tests.

---

# Background and related work

| Reference | Relevant idea | Role in this project |
| --- | --- | --- |
| Vaswani et al. (2017) [1] | Scaled dot-product attention and causal masking | Explains why future characters cannot change historical states |
| Karpathy's GPT lecture [2] | A small character-level decoder Transformer | Architecture and training reference |
| Kwon et al. (2023) [3] | KV reuse during generation and cache memory management | Background for caching and its memory cost |

This experiment studies ordinary KV caching within one request. It does not implement vLLM's PagedAttention or cross-request prefix caching.

---

# What the cache reuses

| Prediction | Recompute history | KV cache |
| --- | --- | --- |
| ABC predicts D | Process ABC | Process ABC and save each layer's K, V |
| ABCD predicts E | Process ABCD | Process only D, using cached K, V from ABC |
| ABCDE predicts next | Process ABCDE | Process only E, extending the cache |

```text
attention = softmax(Q @ K.T / sqrt(head_size)) @ V
```

- During decode, Q has one new position. K and V cover the whole available history.
- Historical states stay unchanged under causal attention and fixed positions. The new query still attends to historical keys.

---

# Implementation and correctness

```python
q, k, v = self.query(x), self.key(x), self.value(x)
if past is not None:
    k = torch.cat((past[0], k), dim=1)
    v = torch.cat((past[1], v), dim=1)
```

- Continue position numbers from the cache length. Select the matching causal-mask rows.
- Use eval mode and inference mode. Start each request with an empty cache.
- Five checks passed on the trained model, covering logits, greedy output, memory, causality/request isolation and window boundaries.
- Maximum absolute logit difference: **0.0000072**, with absolute tolerance 0.00001 and relative tolerance 0.0001. This is numerical agreement on the tested inputs.

---

# Experimental setup

| Item | Setting |
| --- | --- |
| Model | 2 layers, width 128, 4 heads, 429,121 parameters |
| Training | Tiny Shakespeare, 90/10 split, 3,000 steps, batch 8 |
| Training outcome | About 5 minutes. Validation loss 4.28 to 1.83 nats/character |
| Hardware | Ryzen 5 3500U CPU, 4 PyTorch threads, FP32 |
| Workload | Prompts of 16, 32 or 64 characters, 64 output predictions, batch 1 |

- Both paths replay identical held-out continuations. Two warmups per configuration, then five rounds in shuffled order.
- Timing includes model calls only. Loading, tokenization, sampling and text assembly are excluded. Prefill predicts output #1, followed by 63 decode calls.

---

# Decode speed on this CPU

![Decode speed comparison](../results/figures/decode_speed.png)

Bars show medians. Error bars show the full range across five runs.

Cached decode reached about **544-561 predictions/s**, versus **227-322** without caching. The small model and short sequences limit how widely this result can be generalized.

---

# Execution time and cache memory

| Prompt | Uncached total | Cached total | Speedup | Final KV size |
| --- | --- | --- | --- | --- |
| 16 chars | 198.3 ms | 115.0 ms | 1.72x | 158 KiB |
| 32 chars | 229.6 ms | 118.9 ms | 1.93x | 190 KiB |
| 64 chars | 282.1 ms | 117.5 ms | 2.40x | 254 KiB |

Totals are medians of prefill + decode model execution time.

- KV bytes = 2 x layers x batch x cached positions x width x 4. Final cache length is prompt + 63. The last predicted character has not been fed back.
- This implementation copies tensors with cat and requires prompt + output <= 128. KV size excludes temporary allocations and model weights.
- One CPU, one trained model and five timing rounds. Timing outliers remain in the results. These measurements do not establish GPU or production-serving performance.

---

# References and code

- [1] Vaswani et al. (2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762). Sections 3.2 and 3.5.
- [2] Andrej Karpathy. [Let's build GPT: lecture code](https://github.com/karpathy/ng-video-lecture/blob/52201428ed7b46804849dea0b3ccf0de9df1a5c3/gpt.py). Architecture reference, reduced model size.
- [3] Kwon et al. (2023). [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180). Sections 2.1-2.2 and 3.
- Dataset: [Tiny Shakespeare in char-rnn](https://github.com/karpathy/char-rnn/blob/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare/input.txt).

**Code, raw measurements and reproduction commands:**
[github.com/junguanghe/sofia-ai-final](https://github.com/junguanghe/sofia-ai-final)

Project work: cache implementation and a small CPU experiment.
