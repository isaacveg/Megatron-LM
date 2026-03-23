# MoE CDC Runbook

## Purpose

This runbook keeps the first MoE CDC iteration disciplined:

- validate the MoE model before adding CDC complexity
- keep the first CDC variant limited to `dense-only`
- evaluate every run with the same tokenizer, dataset split, and validation cadence

## First Runs

1. Baseline MoE, no CDC

```bash
cd /data/yzhu/megatron-lm-moe-exp
bash scripts/pretrain_llama_moe1b_base.sh
```

2. Baseline MoE + DiLoCo full sync

```bash
cd /data/yzhu/megatron-lm-moe-exp
bash scripts/pretrain_llama_moe1b_diloco_baseline.sh
```

3. Dense-only CDC, fixed round-robin dense shards

```bash
cd /data/yzhu/megatron-lm-moe-exp
bash scripts/pretrain_llama_moe1b_dense_cdc.sh
```

4. Hybrid CDC, dense rolling shards plus low-frequency routed-expert refresh

```bash
cd /data/yzhu/megatron-lm-moe-exp
bash scripts/pretrain_llama_moe1b_hybrid_expert_cdc.sh
```

## Recommended Early Sweeps

### Baseline Stability

- `RUN_NAME=moe1b_base_e8_top2_seed43 SEED=43 bash scripts/pretrain_llama_moe1b_base.sh`
- `RUN_NAME=moe1b_base_e8_top2_mb24 TRAIN_ITERS=5000 bash scripts/pretrain_llama_moe1b_base.sh`
- `RUN_NAME=moe1b_diloco_baseline_i100 CDC_SYNC_INTERVAL=100 bash scripts/pretrain_llama_moe1b_diloco_baseline.sh`
- `RUN_NAME=moe1b_diloco_baseline_i50 CDC_SYNC_INTERVAL=50 bash scripts/pretrain_llama_moe1b_diloco_baseline.sh`

### Dense-Only CDC

- `RUN_NAME=moe1b_dense_cdc_i10_s10 CDC_SYNC_INTERVAL=10 CDC_NUM_SHARDS=10 bash scripts/pretrain_llama_moe1b_dense_cdc.sh`
- `RUN_NAME=moe1b_dense_cdc_i20_s10 CDC_SYNC_INTERVAL=20 CDC_NUM_SHARDS=10 bash scripts/pretrain_llama_moe1b_dense_cdc.sh`
- `RUN_NAME=moe1b_dense_cdc_i10_s5 CDC_SYNC_INTERVAL=10 CDC_NUM_SHARDS=5 bash scripts/pretrain_llama_moe1b_dense_cdc.sh`
- `RUN_NAME=moe1b_dense_cdc_alpha03 CDC_STREAMING_ALPHA=0.3 bash scripts/pretrain_llama_moe1b_dense_cdc.sh`
- `RUN_NAME=moe1b_dense_cdc_alpha07 CDC_STREAMING_ALPHA=0.7 bash scripts/pretrain_llama_moe1b_dense_cdc.sh`

### Hybrid Dense + Expert Refresh

- `RUN_NAME=moe1b_hybrid_exp_i20_o5 CDC_MOE_EXPERT_SYNC_INTERVAL=20 CDC_MOE_EXPERT_SYNC_OFFSET=5 bash scripts/pretrain_llama_moe1b_hybrid_expert_cdc.sh`
- `RUN_NAME=moe1b_hybrid_exp_rr CDC_MOE_EXPERT_SELECTION=round_robin bash scripts/pretrain_llama_moe1b_hybrid_expert_cdc.sh`
- `RUN_NAME=moe1b_hybrid_exp_stale800 CDC_MOE_EXPERT_MAX_STALENESS=800 bash scripts/pretrain_llama_moe1b_hybrid_expert_cdc.sh`

## Interpretation Notes

- `CDC_NUM_SHARDS=10` and `CDC_SYNC_INTERVAL=10` means dense CDC completes one full dense sweep every 100 steps.
- `dense-only` excludes routed expert parameters from CDC and keeps them local-only.
- `router` and `shared experts` still stay in CDC because they directly shape expert usage and global behavior.
- `dense-expert-hybrid` keeps the same dense rolling queue and adds a second routed-expert queue.
- The current expert refresh score is based on past synchronized update magnitude plus age, not live router token counts.

## Metrics To Record

- final train `lm loss`
- validation loss
- validation PPL
- average iteration time
- peak memory per GPU
- CDC communication size/time if CDC is enabled

Use `results_template.csv` to keep the comparison table consistent.
