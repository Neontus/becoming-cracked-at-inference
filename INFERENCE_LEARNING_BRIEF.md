# Learning brief: from GPT-2 architecture to inference

This brief defines the next learning step after Session 1. It is an internal
project guide, not a draft public post.

The accompanying [resource guide](RESOURCES.md) records the references used by
this plan, including [Alisa's Book of LLMs](https://alisawuffles.notion.site/alisa-s-book-of-llms).

## Where I am now

I can describe the high-level GPT-2 path:

```text
text
→ token and position IDs
→ embeddings
→ residual stream
→ Transformer blocks
→ final LayerNorm
→ vocabulary logits
→ next token
```

I also have a runnable GPT-2 architecture in
[`src/inference_lab/gpt2.py`](src/inference_lab/gpt2.py). I understand what the
major components are, but I do not yet understand every operation inside them
well enough to predict inference behavior or optimize it.

## Immediate objective

Build a precise mental model of what the model computes during generation,
then establish a correct, measurable naive inference baseline.

The key transition is:

```text
understanding the model's structure
→ understanding one forward pass
→ understanding repeated autoregressive forward passes
→ seeing what work is repeated
→ removing that repetition with a KV cache
```

Training is not the focus yet. I should understand that training uses a forward
pass to compute predictions and a backward pass to compute gradients, but I do
not need to train GPT-2 or derive backpropagation before studying inference.
Pretrained weights let me study the inference path directly.

## What I need to learn next

### 1. The computations inside one Transformer block

For attention, I should be able to explain and trace:

- how the residual stream is projected into queries, keys, and values;
- how the embedding dimension is split across multiple attention heads;
- why query-key dot products produce attention scores;
- how the causal mask prevents a token from reading future positions;
- how softmax turns scores into attention weights;
- how the weighted value vectors are joined and projected back into the
  residual stream.

For the MLP, I should understand why it expands the channel dimension, applies
GELU, and projects back down. I should be able to distinguish attention's
communication across token positions from the MLP's independent transformation
at each position.

The practical deliverable is a shape trace for every important tensor using
`B` for batch size, `T` for sequence length, `C` for embedding size, and `H` for
the number of heads.

### 2. One forward pass versus generation

A forward pass accepts a sequence of token IDs and produces logits for every
position in that sequence. Generation is an outer loop that repeatedly:

1. runs a forward pass;
2. reads the logits at the final position;
3. selects one next token;
4. appends that token to the sequence;
5. runs the model again.

I should implement or carefully inspect a deliberately naive generation loop
and print the input and output shapes at each step. The goal is to see that
naive decoding recomputes representations for tokens the model has already
processed.

I should also learn the two inference phases:

- **Prefill:** process all prompt tokens, usually in parallel.
- **Decode:** generate subsequent tokens one at a time, reusing prior state
  when possible.

### 3. Pretrained weights and correctness

I do not need to write weight-loading glue by hand for learning purposes, but I
do need to understand what loading accomplishes: it copies already-trained
parameter tensors into the matching modules of my architecture.

Before optimizing anything, I should:

- load pretrained GPT-2 weights;
- run in evaluation mode without gradient tracking;
- compare selected logits or generated tokens with a trusted implementation;
- make generation deterministic while testing;
- keep correctness tests passing as the inference path changes.

This establishes that later performance changes preserve model behavior.

### 4. A trustworthy naive baseline

I should measure the unoptimized model before trying to improve it. For every
measurement, record the device, dtype, batch size, prompt length, generated
length, software versions, and timing method.

The first useful metrics are:

- prefill latency and time to first token;
- decode latency or time per output token;
- tokens generated per second;
- peak memory use when the device exposes it.

Measurements should include warmup and device synchronization where required.
The initial goal is not a good number; it is a number I can reproduce and
explain.

### 5. The first inference optimization: KV caching

Once I can point to the repeated work in naive decoding, I should learn why
past keys and values can be reused while the new token still needs a new query,
key, and value.

Then I should implement a simple KV cache and verify:

- cached and uncached decoding choose the same tokens under deterministic
  decoding;
- tensor shapes grow as expected across decode steps;
- the cache's memory grows with layers, batch size, sequence length, heads, and
  head dimension;
- the performance difference changes with prompt and generation length.

This is the first point where architecture knowledge becomes an inference
systems experiment.

### 6. A basic cost model

After the baseline and KV cache work, I should learn to estimate:

- parameter memory from parameter count and dtype;
- KV-cache memory from model dimensions and sequence length;
- approximate floating-point work in linear layers and attention;
- bytes moved versus operations performed;
- why prefill and decode can have different bottlenecks.

These estimates do not need to be perfect. Their purpose is to make a
prediction before profiling and then explain why measurements agree or differ.

## Suggested order for the next few sessions

Do not begin a session by opening every resource below. Start a session note,
write down the questions for that session, use the listed sources in order, and
return to the code after each source. The learning loop is:

```text
predict or explain from memory
→ consult one source
→ trace or implement in this repository
→ test the result
→ explain it again with the source closed
```

### Session 2 — Attention and MLP from tensors

Estimated time: two or three focused hours. Splitting it across two days is
fine.

#### Learn

1. Watch or read [3Blue1Brown's attention lesson](https://www.3blue1brown.com/lessons/attention/)
   for the conceptual picture. Focus on what a query asks for, what a key
   advertises, why their dot product becomes a relevance score, and what the
   value carries.
2. Watch Karpathy's
   [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) from roughly
   `1:02:00` to `1:37:50`. The most important stretch begins around `1:07:11`
   with query, key, and value projections. Multi-head attention and the MLP
   begin around `1:22:01`; residual connections and LayerNorm follow around
   `1:25:01`.
3. Use the modern-Transformer and attention material in
   [Alisa's Book of LLMs](https://alisawuffles.notion.site/alisa-s-book-of-llms)
   only as a second reference for equations or tensor conventions that remain
   unclear.
4. After the code trace, read sections 2.2.2–2.2.3 of
   [*Inference Engineering*](https://www.baseten.co/inference-engineering/book/)
   to place Transformer blocks and attention in the wider inference stack.

#### Trace in the current implementation

Open `CausalSelfAttention.forward` in `src/inference_lab/gpt2.py`. Use the real
GPT-2 values `B = 1`, `T = 4`, `C = 768`, and `H = 12`, and write down these
shapes before running anything:

```text
x                 (B, T, C)
qkv               (B, T, 3C)
q, k, v           (B, T, C)
q, k, v by head   (B, H, T, C/H)
attention scores  (B, H, T, T)
attention output  (B, H, T, C/H)
joined heads      (B, T, C)
```

Then check the prediction against `CausalSelfAttention.forward` by temporarily
printing shapes or by stepping through the function in a small test. Repeat the
same exercise for `MLP.forward`:

```text
(B, T, 768) → (B, T, 3072) → GELU → (B, T, 768)
```

#### Explain

Answer these without the resources open:

- Why is the attention-score matrix `T × T` for every head?
- Why does `softmax` use the final dimension?
- What does the causal mask change, and what does it not change?
- Why are the heads concatenated back to 768 dimensions?
- Why can attention move information between positions while the MLP cannot?
- Why do both sublayers return an update that is added to the residual stream?

#### Produce

- A more detailed hand-drawn Transformer-block diagram.
- A session note containing the complete shape trace and any remaining
  questions.
- No optimization code yet.

Finish when I can draw one Transformer block in more detail without referring
to an existing diagram and can recover every shape from `B`, `T`, `C`, and
`H`.

### Session 3 — From forward pass to naive generation

Estimated time: two focused hours after Session 2 is complete.

#### Learn

1. Revisit the generation portion of
   [Karpathy's GPT-from-scratch video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
   around `42:14` only to see the outer autoregressive loop.
2. Read the input, logits, and `past_key_values` portions of the
   [Hugging Face GPT-2 documentation](https://huggingface.co/docs/transformers/model_doc/gpt2).
   Ignore caching during the first implementation; notice it only so the later
   contrast is clear.
3. Read only the greedy-decoding portion of the
   [Hugging Face generation guide](https://huggingface.co/docs/transformers/llm_tutorial).
4. Read section 2.2, “LLM Inference Mechanics,” of
   [*Inference Engineering*](https://www.baseten.co/inference-engineering/book/)
   after the naive loop runs. Use it to review the whole inference path, not as
   implementation instructions.

#### Build and inspect

Use trusted glue code to load pretrained GPT-2 weights into the matching local
modules. Weight loading itself is not the lesson; the important fact is that
trained tensors are copied into the parameters of the architecture already
implemented here.

Run one prompt through `GPT.forward` under evaluation and inference modes. For
the first pass, record:

```text
input IDs             (B, prompt_T)
all output logits     (B, prompt_T, vocab_size)
last-position logits  (B, vocab_size)
selected token ID     (B, 1)
next input IDs        (B, prompt_T + 1)
```

Implement the most obvious loop first: pass the entire growing sequence back
through the model for each new token. Use greedy `argmax` while testing so the
same prompt always takes the same path.

#### Compare forward and backward explicitly

- The forward pass uses the current weights to compute activations and logits.
- The loss, backward pass, gradients, and optimizer update are training work.
- Inference needs repeated forward passes but none of the gradient or update
  steps.
- `model.eval()` changes the behavior of training-dependent layers;
  `torch.inference_mode()` avoids autograd bookkeeping. They solve different
  problems and should both appear in the inference path.

#### Produce

- A runnable deterministic generation command.
- A correctness comparison against Hugging Face for selected logits or a short
  greedy continuation.
- A table showing how `T` grows across five decode steps.
- A sentence identifying the waste: the naive loop recomputes hidden states,
  keys, and values for tokens it processed on earlier steps.

Finish when I can clearly distinguish a forward pass, a backward pass, prefill,
and decode, and when the local model generates a trusted deterministic result.

### Session 4 — Establish the baseline

Estimated time: two or three focused hours. Do not optimize during this
session.

#### Learn

Read the introduction and basic `Timer` sections of the
[PyTorch benchmark recipe](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html).
Pay particular attention to warmup and accelerator synchronization. Use the
inference and performance sections of Alisa's Book of LLMs to connect latency,
throughput, model memory, and arithmetic work. Then read sections 1.4, 2.4,
and 4.5 of [*Inference Engineering*](https://www.baseten.co/inference-engineering/book/)
for metric definitions, bottleneck analysis, and benchmarking practice.

#### Define the measurements

- **Prefill latency:** time for the forward pass over the initial prompt.
- **Time to first token (TTFT):** tokenization plus prefill plus first-token
  selection, if the benchmark includes the full user-visible path.
- **Decode latency:** time for one subsequent token step.
- **Time per output token (TPOT):** average decode time over generated tokens;
  state whether the first token is excluded.
- **Throughput:** output tokens per second for the defined request or batch.

Keep low-level model timing separate from end-to-end timing so tokenizer and
Python overhead do not silently change the question being answered.

#### Benchmark matrix

Start with batch size 1 and greedy decoding. Use a small matrix such as:

```text
prompt lengths:     8, 32, 128, 512
generated lengths:  1, 8, 32
dtype:              record the actual dtype
device:             record the exact device
```

Before running it, predict which cases should take longer and why. Warm up,
repeat each measurement, report a median or distribution rather than one run,
and record software versions. If using CUDA, synchronize correctly; if using
CPU or MPS, document the timing method and its limitations.

#### Produce

- A small benchmark harness, not a notebook cell with unrecorded state.
- A machine-readable result file in ignored `artifacts/` plus a short committed
  summary if useful.
- A table separating prefill from decode.
- A written explanation of the repeated computation visible in the naive loop.

Finish when I trust the measurement process, even if performance is poor.

### Session 5 — Implement and measure a KV cache

Estimated time: two or three sessions. Correctness comes before speed.

#### Learn

1. Read the [Hugging Face caching explanation](https://huggingface.co/docs/transformers/cache_explanation)
   through the dynamic-cache tensor shapes.
2. Return to the `past_key_values` contract in the
   [GPT-2 documentation](https://huggingface.co/docs/transformers/model_doc/gpt2).
3. Use the KV-cache and inference sections in Alisa's Book of LLMs as the
   derivation reference.
4. Read section 5.3, “Caching,” of
   [*Inference Engineering*](https://www.baseten.co/inference-engineering/book/)
   after the simple cache works, so prefix reuse and cache-aware serving do not
   distract from the first implementation.
5. Do not read paged-attention implementation details yet.

#### Predict the changed shapes

For a decode step after `T_past` cached tokens, derive:

```text
new x                (B, 1, C)
new q, k, v          (B, H, 1, head_dim)
cached k and v       (B, H, T_past, head_dim)
combined k and v     (B, H, T_past + 1, head_dim)
attention scores     (B, H, 1, T_past + 1)
```

The current token still needs new query, key, and value projections. The cache
removes the need to recompute earlier tokens' hidden states, keys, and values;
it does not make attention over the existing context free.

#### Implement in two stages

1. Change attention and block interfaces so each layer can accept past keys and
   values and return updated ones.
2. Split generation into prefill, which populates all layer caches, and decode,
   which passes only the new token plus the cache.

Begin with a simple dynamically growing cache. It may use concatenation and be
imperfect; preallocation and paging are separate optimizations.

#### Verify before timing

- Compare cached and uncached logits within a justified numerical tolerance at
  every decode step.
- Compare the generated greedy token IDs, not just the final decoded string.
- Test sequence length 1, several decode steps, and the context-length limit.
- Check every layer's cache shape.

Only after those tests pass should the Session 4 benchmark run against both
paths.

Finish when I can explain both what the cache saves and what it costs.

## What to do in the very next study block

Do only Session 2, not the entire plan:

1. Create a session note:

   ```bash
   python3 scripts/new_session.py "attention and MLP tensor shapes"
   ```

2. Watch the 3Blue1Brown attention lesson.
3. Watch `1:02:00–1:37:50` of Karpathy's GPT-from-scratch video.
4. Fill out the `B, T, C, H` shape trace before executing the code.
5. Verify the trace in `CausalSelfAttention.forward` and `MLP.forward`.
6. Draw one detailed block and write down the first point that remains unclear.

That is enough for one productive session. Pretrained-weight loading and the
generation loop begin only after this explanation checkpoint is met.

## What to defer

For now, I should not let these topics interrupt the path above:

- training a useful GPT-2 model from scratch;
- optimizer details and full backpropagation derivations;
- CUDA or Triton kernels;
- FlashAttention implementation details;
- quantization;
- continuous batching, paged attention, or production scheduling;
- multi-GPU inference;
- switching architectures solely to cover RoPE, RMSNorm, or grouped-query
  attention.

Those topics become useful after I have a correct generation loop, a KV cache,
and baseline measurements that expose the problems they solve.

## Readiness checkpoint

I am ready to move from model comprehension into serious inference experiments
when I can do all of the following:

- trace the shapes through attention and the MLP;
- explain query, key, and value in operational terms;
- explain a forward pass without confusing it with backpropagation;
- distinguish prefill from decode;
- generate text with pretrained weights through a naive loop;
- measure that loop reproducibly;
- identify the repeated work in naive decoding;
- predict what a KV cache changes in computation and memory.

The next concrete task is Session 2: trace attention and MLP tensors through
the current GPT-2 implementation.
