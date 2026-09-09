# Speaker notes (about 10 minutes)

Use these as prompts, then explain the project in your own words. The statements
describe the completed local experiment; the slides and code still need to be
committed and pushed before the GitHub submission link contains them.

## Slide 1 - 50 seconds

“I studied one basic inference optimization, KV caching. I used a very small
character-level Transformer so the whole experiment could run on my laptop CPU.
Both versions use the same trained weights. Across the three workloads, cached
model execution was about 1.7 to 2.4 times faster, with less than 256 KiB of final
KV tensors. The main learning objective was understanding what can be reused.”

## Slide 2 - 70 seconds

“The Transformer paper provides the attention mechanism and causal masking.
Karpathy's lecture provides the small decoder architecture I used as a reference.
The vLLM paper explains autoregressive generation and why KV cache memory matters
for serving. My implementation uses simple contiguous tensors. PagedAttention
solves a larger memory-management problem that this experiment does not test.”

## Slide 3 - 100 seconds

Walk through the ABC example slowly. The first forward predicts D and saves the
keys and values for ABC in every layer. The next forward accepts D, calculates
its query/key/value and uses cached history to predict E. Explain the formula:
the new query matches all historical keys and combines their values. Historical
queries are unnecessary because only the new position's output is needed.

Emphasize that causal masking protects historical states from future tokens.
Caching saves repeated computation. It does not remove attention over history.

## Slide 4 - 90 seconds

Explain `past`, `torch.cat`, and the cache nesting by layer and head. Mention the
two implementation details that matter most: position offset and mask rows.
Eval mode switches off dropout, while inference mode disables gradient tracking.
The tests compare all-position logits as well as greedy sequences. The observed
small numerical difference comes from different floating-point execution shapes.
It stayed within the specified tolerances on all tested inputs.

## Slide 5 - 80 seconds

“I trained once, on the first 90 percent of Tiny Shakespeare. The final 10 percent
was reserved for validation. The vocabulary has 65 characters. Training took
about five minutes and validation cross-entropy fell to 1.83. This is still a
small imperfect language model. Its writing quality is not the variable in the
cache experiment.”

Explain fixed continuations, warmup, shuffled order and fixed threads. Clarify
that the timing measures forward calls, including cache operations, but excludes
sampling and application overhead. There are 63 decode calls after prefill for
64 output predictions.

## Slide 6 - 60 seconds

Explain both axes and the error bars. Cached decode is roughly flat over this
short length range, while recomputation slows down as more history is processed.
Do not claim cached decode has constant complexity: it still attends over all
keys. At this small size other costs can dominate, and the timings have outliers.

## Slide 7 - 100 seconds

The speedup is the uncached median total divided by the cached median total.
Explain the two factors in the memory formula that are often confused: the
leading 2 means K and V, and head count is already included in model width.
At prompt 64, final cache length is 127, which is 254 KiB. The 128th predicted
position has not yet been processed. Mention cat copies, the fixed window, the
single CPU and the lack of a production-serving comparison.

## Slide 8 - 30 seconds

Point to the repository containing the implementation, measurements, figures and
commands. Credit the architecture reference and AI assistance accurately. End
with the main finding: retaining historical K/V avoided enough recomputation to
speed up this small CPU workload, at a directly measurable memory cost.

## Questions to rehearse

- Does caching change the learned model? No. The same parameters implement the
  same causal computation, subject to floating-point differences.
- Why no Q cache? Only the latest position's attention output is required during
  generation, so its new query is sufficient.
- Is this prefix caching? It is reuse within one request, not sharing prompt
  computation across different requests.
- Why not compare quality scores? Both paths use the same checkpoint. Numerical
  equivalence is the relevant check; generation quality belongs to training.
- Can it generate indefinitely? No. This implementation rejects requests beyond
  128 total characters, because learned absolute positions have a fixed range.
- Does 254 KiB mean the whole process uses that much memory? No. It counts only
  the final K/V tensors, excluding weights and temporary allocations.
- Why use fixed continuations? To hold the input work constant while comparing
  the two forward paths. A separate sample demonstrates actual generation.
- Would the same speedup hold on a GPU? The experiment does not establish that.
  Kernel overhead, parallelism, memory bandwidth and model size all differ.
