---
title: "Session 1: Understanding the Basic GPT-2 Architecture"
summary: "A hand-drawn walkthrough of how GPT-2 turns prompt tokens into next-token logits through embeddings, residual streams, attention, and MLP updates."
date: "2026-09-22"
updated: "2026-09-22"
stage: "Architecture"
order: 1
draft: false
tags:
  - inference
  - gpt-2
  - transformers
codeUrl: "https://github.com/Neontus/becoming-cracked-at-inference/tree/6e21ee0"
---

# Session 1: Understanding the Basic GPT-2 Architecture

After spending some time watching Andrej Karpathy’s video on
[reproducing GPT-2](https://www.youtube.com/watch?v=l8pRSuU81PU), I put what I
learned into a diagram.

![A hand-drawn diagram tracing a prompt through GPT-2 embeddings, Transformer blocks, and next-token selection.](/writing/assets/session-1-gpt2-architecture.png)

Drawing the architecture myself was surprisingly useful, and I strongly
encourage doing the same rather than only looking at an existing diagram. At a
high level, mine shows how a prompt moves through GPT-2 to produce its next
token.

The corresponding code is pinned to
[the GPT-2 architecture implementation commit](https://github.com/Neontus/becoming-cracked-at-inference/tree/6e21ee0),
which includes the model structure and its initial correctness tests.

## A brief walkthrough

- The tokenizer converts the natural-language prompt into token IDs. GPT-2 also
  assigns each token a position ID based on where it occurs in the prompt.
- The token IDs and position IDs are used to look up learned token and position
  embeddings. These vectors are added to create the initial residual stream.
- The residual stream passes through 12 Transformer blocks. Each block preserves
  its overall shape while adding two updates: one from attention and another
  from an MLP.
- Attention allows each token position to retrieve information from itself and
  previous positions. Its output is added to the existing residual stream.
- The MLP then transforms the information at each token position independently.
  Its output is also added to the residual stream.
- After all 12 blocks, GPT-2 applies a final LayerNorm and passes the result
  through the language-model head.
- The LM head converts each 768-dimensional residual representation into 50,257
  logits—one score for every token in GPT-2’s vocabulary.
- During generation, we take the logits at the final prompt position. Softmax
  can convert them into probabilities, and a decoding method such as greedy
  selection or sampling chooses the next token.

## Another way I think about it

If you think in stages like I do, the following abstraction may help.

1. We begin in comfortable territory: natural-language text.
2. The tokenizer converts that text into discrete tokens and their corresponding
   token IDs. We also create position IDs representing where those tokens occur
   in the sequence.
3. These IDs are used as indices into learned embedding tables. The
   token-embedding vector answers something like “what token is this?”, while
   the position-embedding vector adds information about where it occurs.
4. The token and position vectors are added to create the initial residual
   stream. We are now operating in the model’s 768-dimensional internal
   representation space, with one vector for every token position.
5. Each Transformer block updates those vectors without changing their overall
   shape. Attention communicates information between token positions, while the
   MLP nonlinearly transforms the information available at each position.
6. After 12 blocks, the LM head maps each 768-dimensional representation into
   vocabulary space, producing one score for each possible next token.
7. We use the scores at the final position to select a new token and decode it
   back into text.

A compact summary is:

```text
text
→ token IDs and position IDs
→ token and position embeddings
→ residual stream
→ 12 Transformer blocks
→ final LayerNorm
→ LM head
→ vocabulary logits
→ next token
```

## What comes next

I’m now ready to examine individual parts of the architecture more
deeply—particularly query, key, and value projections, multiple attention heads,
and the MLP.

I still have more studying to do before beginning serious inference experiments,
but this session gave me a working mental model of the complete GPT-2
architecture. I also set up a small GPT-2 implementation based on Karpathy’s
walkthrough locally so I can connect each concept to actual code.

In the next session, I plan to study the attention and MLP operations more
carefully as preparation for understanding inference optimizations.
