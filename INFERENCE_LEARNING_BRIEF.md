# Learning brief: from GPT-2 architecture to inference

This brief defines the next learning step after Session 1. It is an internal
project guide, not a draft public post.

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

### Session 2 — Attention and MLP from tensors

- Trace query, key, value, attention-score, attention-output, and MLP shapes.
- Connect every operation to the corresponding line in `gpt2.py`.
- Explain why attention mixes tokens while the MLP does not.

Finish when I can draw one Transformer block in more detail without referring
to an existing diagram.

### Session 3 — From forward pass to naive generation

- Load pretrained weights using trusted glue code.
- Run one prompt through the model.
- Write or inspect the naive autoregressive loop.
- Print sequence lengths and tensor shapes during several decode steps.

Finish when I can clearly distinguish a forward pass, a backward pass, prefill,
and decode.

### Session 4 — Establish the baseline

- Add a small reproducible benchmark harness.
- Measure prefill and decode separately across a few sequence lengths.
- Record the experimental setup and make a prediction before timing.
- Identify exactly which computation the naive loop repeats.

Finish when I trust the measurement process, even if performance is poor.

### Session 5 — Implement and measure a KV cache

- Add the simplest correct cache before considering paged memory layouts.
- Test cached output against the naive path.
- Measure the change in decode time and memory use.
- Explain the result using tensor shapes and the basic cost model.

Finish when I can explain both what the cache saves and what it costs.

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
