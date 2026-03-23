# MoE CDC Experiment Plan

## Goal

Build an experimental cross-domain MoE training setup on top of the current CDC codebase,
using the existing LLaMA tokenizer and C4 dataset, and iterate on several MoE sync policies.

This worktree is intentionally isolated from the main CDC implementation directory.

## Principles

- No destructive operations.
- First validate that the MoE model itself trains stably before adding CDC logic.
- Separate model-design risk from communication-policy risk.
- Keep the first MoE implementation simple: no EP, no grouped GEMM, no CUDA graph.
- Use validation loss and validation perplexity as the main selection metrics.

## Candidate MoE Configurations

These estimates use the repository's parameter counting formula in
`megatron/training/theoretical_memory_usage.py`.

### Primary Candidate

- Layers: 14
- Hidden size: 1536
- FFN hidden size: 4096
- Attention heads: 24
- Query groups: 6
- MoE layers: every other layer (`--moe-layer-freq 2` => 7 MoE layers)
- Experts: 8
- Router top-k: 2
- MoE FFN hidden size: 2816
- Estimated total params: ~1.02B
- Estimated active params: ~0.48B

Why this one:

- Keeps total parameters near the requested 1B scale.
- Uses 8 experts and top-2 routing, which is more meaningful for expert-selection experiments
  than a 4-expert model.
- Avoids making experts so tiny that expert specialization becomes trivial.

### Backup Candidate

- Layers: 20
- Hidden size: 1536
- FFN hidden size: 4096
- Attention heads: 24
- Query groups: 6
- MoE layers: every other layer (`--moe-layer-freq 2` => 10 MoE layers)
- Experts: 4
- Router top-k: 2
- MoE FFN hidden size: 4096
- Estimated total params: ~1.14B
- Estimated active params: ~0.76B

Why keep it:

- More active capacity than the primary candidate.
- Simpler expert-selection problem if the 8-expert model is unstable or too weak.

## Implementation Variants

### Variant A: Baseline MoE, No CDC

Purpose:

- Confirm the MoE model trains at all on the current tokenizer/data setup.
- Measure memory, speed, validation loss, and validation PPL.

Policy:

- No CDC.
- Standard MoE training only.

### Variant B: Dense-Only CDC

Purpose:

- Build the first MoE + CDC baseline with minimal algorithmic risk.

Policy:

- Dense weights participate in CDC.
- Routed experts remain local-only.
- Router and shared experts are treated as dense/high-priority parameters.

This is the lowest-risk MoE CDC baseline and is conceptually close to partially-local training.

### Variant C: Dense Round-Robin + Expert Round-Robin

Purpose:

- Add expert synchronization while keeping the policy deterministic and easy to debug.

Policy:

- Dense shards use a fixed round-robin calendar.
- Experts also use a fixed round-robin calendar.
- No score-based expert choice yet.

This separates scheduler bugs from score-definition bugs.

### Variant D: Dense Round-Robin + Score-Based Expert Selection

Purpose:

- Implement the intended MoE CDC behavior.

Policy:

- Dense shards use round-robin.
- Expert sync uses a byte-budgeted priority selection.
- First score should be based on expert token counts plus age/staleness.
- Optional second-generation score adds EMA update norm.

## Recommended Rollout Order

1. Baseline MoE no CDC.
2. Dense-only CDC.
3. Expert round-robin CDC.
4. Expert score-based CDC.

Do not jump directly to score-based expert selection before the first three work.

## Metrics to Record

- Training `lm loss`
- Validation loss
- Validation perplexity
- Iteration time
- Peak GPU memory
- CDC communication size / time

Later, after the first MoE CDC version is working:

- Expert token-count skew
- Expert selection frequency
- Fraction of expert bytes synchronized per super-cycle

## 5000-Step Experiment Matrix

### Stage 0: Model Validation

- `baseline_moe_primary`
- `baseline_moe_backup` if the primary configuration is unstable

### Stage 1: CDC Policy Validation

- `moe_dense_only_cdc`
- `moe_dense_rr_expert_rr`

### Stage 2: Score-Based Expert Sync

- `moe_dense_rr_expert_score_tokens`
- `moe_dense_rr_expert_score_tokens_age`
- `moe_dense_rr_expert_score_tokens_delta`

## First Script Defaults

- Train iterations: 5000
- Eval interval: 250
- Eval iters: 50
- Save interval: 1000
- Validation metric for model/policy selection: validation loss first, validation PPL second

## Current Technical Boundaries

- The existing `CDCOptimizer` shard planner is dense-GPT-specific.
- Current streaming/DC code has a single queue; MoE CDC will need at least dense/expert split.
- For the first MoE implementation, keep `--expert-model-parallel-size 1`.
- For the first MoE implementation, keep `--moe-grouped-gemm` disabled to avoid packed-expert
  parameter slicing complexity.

## Immediate Next Steps

1. Add a baseline MoE training script for the primary candidate.
2. Smoke-test memory and launch stability.
3. Add the first dense-only CDC MoE variant.
4. Only after that, modify CDC code for expert-aware scheduling.
