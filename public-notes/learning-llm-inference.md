---
title: "Learning LLM Inference From First Principles"
summary: "A running lab notebook about building, measuring, and understanding the systems behind fast language-model inference."
date: "2026-09-15"
updated: "2026-09-15"
stage: "Orientation"
order: 0
draft: true
tags:
  - inference
  - systems
codeUrl: ""
---

# Learning LLM Inference From First Principles

I am spending the next several months learning how language models actually run:
from transformer computation and KV caches down through GPU kernels, schedulers,
memory management, and multi-GPU communication.

This is not a sequence of course summaries. It is a record of things I derived,
built, measured, misunderstood, and eventually learned to explain.

## The learning loop

For each topic, I will follow the same order:

1. build a mental model and make a prediction;
2. implement the smallest version that exposes the mechanism;
3. test it against a trusted reference;
4. benchmark and profile it;
5. explain the evidence, including surprises and open questions;
6. only then inspect how production systems solve the same problem.

The project begins with an intentionally inefficient autoregressive generation
loop. From there, each bottleneck should motivate the next system: KV caching,
continuous batching, paged memory, optimized attention, and eventually
distributed inference.

## What will appear here

Short field notes will cover individual mechanisms and experiments. Larger
reports will connect milestones across the engine. Every performance claim will
include enough information to understand the workload and reproduce the result,
and every implementation note will point to the corresponding code when it
exists.

The first technical note will come after I have implemented naive generation and
can explain exactly where its repeated work comes from.
