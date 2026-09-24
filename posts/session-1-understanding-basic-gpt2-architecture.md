---
title: "Session 1: Understanding the Basic GPT-2 Architecture"
summary: "A hand-drawn walkthrough of how GPT-2 turns prompt tokens into next-token logits through embeddings, residual streams, attention, and MLP updates."
date: "2026-09-22"
updated: "2026-09-24"
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
high level, mine shows how a <dfn class="concept-term" tabindex="0" data-definition="The input text or token sequence given to a language model before it generates a continuation." aria-label="Prompt: the input text or token sequence given to a language model before it generates a continuation.">prompt</dfn>
moves through GPT-2 to produce its next token.

The corresponding code is pinned to
[the GPT-2 architecture implementation commit](https://github.com/Neontus/becoming-cracked-at-inference/tree/6e21ee0),
which includes the model structure and its initial correctness tests.

## A brief walkthrough

- The <dfn class="concept-term" tabindex="0" data-definition="A component that splits text into model-readable tokens and maps each token to an integer ID." aria-label="Tokenizer: a component that splits text into model-readable tokens and maps each token to an integer ID.">tokenizer</dfn>
  converts the natural-language prompt into <dfn class="concept-term" tabindex="0" data-definition="Integer indices identifying entries in GPT-2’s 50,257-token vocabulary." aria-label="Token IDs: integer indices identifying entries in GPT-2’s 50,257-token vocabulary.">token IDs</dfn>.
  GPT-2 also assigns each token a <dfn class="concept-term" tabindex="0" data-definition="An integer marking where a token occurs in the sequence, from position 0 up to GPT-2’s context limit." aria-label="Position ID: an integer marking where a token occurs in the sequence, from position 0 up to GPT-2’s context limit.">position ID</dfn>
  based on where it occurs in the prompt.
- The token IDs and position IDs are used to look up learned token and position
  <dfn class="concept-term" tabindex="0" data-definition="Learned dense vectors retrieved from tables using token or position IDs as indices." aria-label="Embeddings: learned dense vectors retrieved from tables using token or position IDs as indices.">embeddings</dfn>.
  These vectors are added to create the initial <dfn class="concept-term" tabindex="0" data-definition="The 768-dimensional vector at each token position that carries information through every Transformer block." aria-label="Residual stream: the 768-dimensional vector at each token position that carries information through every Transformer block.">residual stream</dfn>.
- The residual stream passes through 12 <dfn class="concept-term" tabindex="0" data-definition="Repeated layers that update the residual stream with an attention operation followed by an MLP operation." aria-label="Transformer blocks: repeated layers that update the residual stream with an attention operation followed by an MLP operation.">Transformer blocks</dfn>.
  Each block preserves its overall shape while adding two updates: one from
  <dfn class="concept-term" tabindex="0" data-definition="An operation that lets each token retrieve a weighted mixture of information from itself and allowed earlier tokens." aria-label="Attention: an operation that lets each token retrieve a weighted mixture of information from itself and allowed earlier tokens.">attention</dfn>
  and another from an <dfn class="concept-term" tabindex="0" data-definition="A position-wise feed-forward network that expands each 768-dimensional vector to 3,072 dimensions, applies a nonlinearity, then projects it back to 768." aria-label="MLP: a position-wise feed-forward network that expands each 768-dimensional vector to 3,072 dimensions, applies a nonlinearity, then projects it back to 768.">MLP</dfn>.
- Attention allows each token position to retrieve information from itself and
  previous positions. Its output is added to the existing residual stream.
- The MLP then transforms the information at each token position independently.
  Its output is also added to the residual stream.
- After all 12 blocks, GPT-2 applies a final <dfn class="concept-term" tabindex="0" data-definition="A normalization applied across each token’s 768 features to keep the representation numerically well-behaved." aria-label="LayerNorm: a normalization applied across each token’s 768 features to keep the representation numerically well-behaved.">LayerNorm</dfn>
  and passes the result through the <dfn class="concept-term" tabindex="0" data-definition="The final linear projection that maps a 768-dimensional representation to one score for every vocabulary token." aria-label="Language-model head: the final linear projection that maps a 768-dimensional representation to one score for every vocabulary token.">language-model head</dfn>.
- The LM head converts each 768-dimensional residual representation into 50,257
  <dfn class="concept-term" tabindex="0" data-definition="Unnormalized scores representing the model’s relative preference for every possible next token." aria-label="Logits: unnormalized scores representing the model’s relative preference for every possible next token.">logits</dfn>—one
  score for every token in GPT-2’s vocabulary.
- During generation, we take the logits at the final prompt position.
  <dfn class="concept-term" tabindex="0" data-definition="A function that converts logits into nonnegative probabilities that sum to one." aria-label="Softmax: a function that converts logits into nonnegative probabilities that sum to one.">Softmax</dfn>
  can convert them into probabilities, and a decoding method such as
  <dfn class="concept-term" tabindex="0" data-definition="A decoding method that always chooses the token with the highest score or probability." aria-label="Greedy selection: a decoding method that always chooses the token with the highest score or probability.">greedy selection</dfn>
  or <dfn class="concept-term" tabindex="0" data-definition="A decoding method that randomly draws the next token from the model’s probability distribution." aria-label="Sampling: a decoding method that randomly draws the next token from the model’s probability distribution.">sampling</dfn>
  chooses the next token.

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
