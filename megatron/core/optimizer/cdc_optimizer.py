# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.distributed as dist
from torch.optim import SGD
from copy import deepcopy
from typing import List, Dict, Any, Optional, Tuple
import time
import re
import math

from megatron.core import mpu
from megatron.training.utils import print_rank_0
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.transformer.module import MegatronModule
from megatron.training.global_vars import get_args

class CDCOptimizer(MegatronOptimizer):
    """
    Wrapper optimizer for cross data-center training (DiLoCo, Streaming, DC).
    It wraps an inner MegatronOptimizer(ChainedOptimizer) and adds outer optimization steps.
    Supports standard DiLoCo as well as Streaming and Delay Compensated (DC) variants
    """

    def __init__(
        self,
        inner_optimizer: MegatronOptimizer,
        model_chunks: Optional[List[MegatronModule]] = None
    ):
        self.inner_optimizer = inner_optimizer
        # 要引用模型的chunks以便划分参数；优先显式传入，其次复用内层优化器携带的 model_chunks
        self.model_chunks = model_chunks if model_chunks is not None else getattr(inner_optimizer, "model_chunks", None)
        assert self.model_chunks is not None, "model_chunks must be provided to CDCOptimizer."
        self.config = self.inner_optimizer.config

        # 获取参数
        args = get_args()
        self.tie_embeddings = not args.untie_embeddings_and_output_weights
        # assert args.untie_embeddings_and_output_weights, "CDC does not support tied embeddings and output weights."
        self.cdc_group = mpu.get_cdc_parallel_group()
        self.sync_interval = args.cdc_sync_interval
        self.step_count = 0
        self.algorithm = args.cdc_algorithm
        self.offload_outer_opt = args.cdc_offload_outer_opt
        self.outer_lr = float(args.cdc_outer_lr)
        self.dense_outer_lr_arg = float(getattr(args, "cdc_dense_outer_lr", -1.0))
        self.expert_outer_lr_arg = float(getattr(args, "cdc_moe_expert_outer_lr", -1.0))
        self.dense_outer_lr = self._resolve_component_outer_lr(
            self.dense_outer_lr_arg, "cdc_dense_outer_lr"
        )
        self.expert_outer_lr = self._resolve_component_outer_lr(
            self.expert_outer_lr_arg, "cdc_moe_expert_outer_lr"
        )
        self.num_shards = args.cdc_num_shards
        self.dc_lambda = args.cdc_dc_lambda
        self.streaming_alpha = float(args.cdc_streaming_alpha)
        if self.streaming_alpha < 0.0 or self.streaming_alpha > 1.0:
            raise ValueError(
                f"cdc_streaming_alpha must be in [0, 1], got {self.streaming_alpha}"
            )
        self.dense_alpha_arg = float(getattr(args, "cdc_dense_alpha", -1.0))
        self.router_alpha_arg = float(args.cdc_moe_router_alpha)
        self.expert_alpha_arg = float(getattr(args, "cdc_moe_expert_alpha", -1.0))
        self.dense_alpha = self._resolve_component_alpha(
            self.dense_alpha_arg, "cdc_dense_alpha"
        )
        self.router_alpha = self._resolve_component_alpha(
            self.router_alpha_arg, "cdc_moe_router_alpha"
        )
        self.expert_alpha = self._resolve_component_alpha(
            self.expert_alpha_arg, "cdc_moe_expert_alpha"
        )
        self.delay = args.cdc_delay
        self.dc_N = args.cdc_dc_N
        self.shard_pattern = args.cdc_shard_pattern
        self.moe_param_mode = str(args.cdc_moe_param_mode).lower()
        self.expert_sync_interval = int(args.cdc_moe_expert_sync_interval)
        self.expert_sync_offset = int(args.cdc_moe_expert_sync_offset)
        self.expert_selection = str(args.cdc_moe_expert_selection).lower()
        self.expert_topk = int(args.cdc_moe_expert_topk)
        self.expert_score_mode = str(args.cdc_moe_expert_score_mode).lower()
        self.expert_max_age_slots = int(args.cdc_moe_expert_max_age_slots)
        self.expert_min_age_slots = int(args.cdc_moe_expert_min_age_slots)
        self.expert_max_staleness = int(args.cdc_moe_expert_max_staleness)
        self.blocking_full_sync_steps = int(args.cdc_blocking_full_sync_steps)
        self.verbose = args.cdc_verbose
        self.mixed_precision = args.bf16 or args.fp16
        self._named_model_param_list = self._iter_named_trainable_params_unique(self.model_chunks)
        self._expert_named_model_params = self._collect_routed_expert_named_params(
            self._named_model_param_list
        )
        self._router_named_model_params = self._collect_router_named_params(
            self._named_model_param_list
        )
        self.enable_moe_router_refresh = (
            self.moe_param_mode == 'dense-expert-hybrid'
            and self.algorithm == 'streaming'
            and len(self._router_named_model_params) > 0
        )
        self._tracked_named_model_params = self._filter_named_params_for_cdc(
            self._named_model_param_list
        )
        self.has_moe_expert_trackers = (
            self.moe_param_mode == 'dense-expert-hybrid'
            and self.algorithm == 'streaming'
            and len(self._expert_named_model_params) > 0
            and (
                self.expert_sync_interval > 0
                or self.blocking_full_sync_steps > 0
            )
        )
        self.enable_moe_expert_refresh = (
            self.has_moe_expert_trackers and self.expert_sync_interval > 0
        )
        self.track_expert_token_load = (
            self.enable_moe_expert_refresh
            and (
                (
                    self.expert_selection == 'score'
                    and self.expert_score_mode in {'token_load', 'mixed'}
                )
                or self.verbose
            )
        )
        self._token_load_source_debug_printed = False
        self._token_load_step_debug_printed = False
        tracked_params = self.tracked_model_param_list
        self.model_param_dtype = tracked_params[0].dtype if tracked_params else torch.float32
        # Follow Streaming DiLoCo: keep outer-state math in fp32, but keep communication low
        # precision by default to avoid doubling bandwidth.
        self.outer_state_dtype = torch.float32 if self.mixed_precision else self.model_param_dtype
        self.outer_comm_dtype = (
            self.model_param_dtype if self.mixed_precision else self.outer_state_dtype
        )
        # DC specifics.
        self.dc_type = str(args.cdc_dc_type).lower()
        # Initialized for all algorithms so checkpoint/state construction does not fail.
        self.next_shard_idx = 0
        self.next_expert_group_idx = 0
        self.expert_sync_event_count = 0
        self._cdc_state_loaded = False

        if self.moe_param_mode == 'dense-expert-hybrid' and self.algorithm != 'streaming':
            raise ValueError(
                "cdc_moe_param_mode=dense-expert-hybrid currently only supports "
                "cdc_algorithm=streaming."
            )
        if self.expert_topk < 1:
            raise ValueError(f"cdc_moe_expert_topk must be >= 1, got {self.expert_topk}")
        if self.expert_min_age_slots < 0:
            raise ValueError(
                f"cdc_moe_expert_min_age_slots must be >= 0, got {self.expert_min_age_slots}"
            )
        if self.blocking_full_sync_steps < 0:
            raise ValueError(
                f"cdc_blocking_full_sync_steps must be >= 0, got {self.blocking_full_sync_steps}"
            )
        if self.expert_selection not in {'round_robin', 'score'}:
            raise ValueError(
                f"Unknown cdc_moe_expert_selection: {self.expert_selection}"
            )
        if self.expert_score_mode not in {'update_norm', 'token_load', 'mixed'}:
            raise ValueError(
                f"Unknown cdc_moe_expert_score_mode: {self.expert_score_mode}"
            )
        if self.enable_moe_expert_refresh and int(args.expert_model_parallel_size) != 1:
            raise NotImplementedError(
                "dense-expert-hybrid currently supports expert_model_parallel_size=1 only."
            )
        if self.algorithm == 'diloco' and self.blocking_full_sync_steps > 0:
            raise ValueError(
                "cdc_blocking_full_sync_steps is only supported for streaming/DC CDC runs. "
                "diloco already performs a blocking full sync via cdc_sync_interval."
            )

        self._moe_module_index = (
            self._build_moe_module_index() if self.track_expert_token_load else {}
        )
        if self.verbose and self._moe_module_index:
            preview_keys = sorted(self._moe_module_index.keys())[:8]
            print_rank_0(
                f"[CDC][MoEIndex] modules={len(self._moe_module_index)} preview={preview_keys}"
            )

        # Track per-param-group LR/WD history for DC debiasing.
        self._optimizer_param_id_to_group_idx: Dict[int, int] = {}
        self._lr_cumsums_by_group: Optional[List[float]] = None
        self._wd_log_cumsums_by_group: Optional[List[float]] = None
        self._init_lr_wd_tracking()
        self._build_optimizer_param_group_index()

        if self.verbose:
            print_rank_0(
                f"[CDC] Initialized {self.algorithm} optimizer. Sync interval: "
                f"{self.sync_interval}, Shards: {self.num_shards}, "
                f"outer_lr={self.outer_lr}, "
                f"dense_outer_lr={self._format_component_value(self.dense_outer_lr_arg, self.dense_outer_lr)}, "
                f"expert_outer_lr={self._format_component_value(self.expert_outer_lr_arg, self.expert_outer_lr)}, "
                f"outer_state_dtype={self.outer_state_dtype}, outer_comm_dtype={self.outer_comm_dtype}, "
                f"moe_param_mode={self.moe_param_mode}, "
                f"dense_alpha={self._format_component_alpha(self.dense_alpha_arg, self.dense_alpha)}, "
                f"router_refresh={'every_dense_slot' if self.enable_moe_router_refresh else 'off'}, "
                f"router_alpha={self._format_component_alpha(self.router_alpha_arg, self.router_alpha)}, "
                f"expert_refresh={'on' if self.enable_moe_expert_refresh else 'off'}, "
                f"expert_alpha={self._format_component_alpha(self.expert_alpha_arg, self.expert_alpha)}, "
                f"expert_topk={self.expert_topk}, expert_score_mode={self.expert_score_mode}, "
                f"expert_min_age_slots={self.expert_min_age_slots}, "
                f"blocking_full_sync={'off' if self.blocking_full_sync_steps <= 0 else f'every_{self.blocking_full_sync_steps}_steps'}"
            )
        
        if self.algorithm == 'diloco':
            self._init_diloco_state()
        elif self.algorithm in ['streaming', 'dc']:
            self._init_streaming_state()
        else:
            raise ValueError(f"Unknown DiLoCo algorithm: {self.algorithm}")

    def _resolve_component_alpha(self, alpha_value: float, arg_name: str) -> float:
        """Resolve per-component alpha; negative values inherit cdc_streaming_alpha."""
        if alpha_value < 0.0:
            return self.streaming_alpha
        if alpha_value > 1.0:
            raise ValueError(f"{arg_name} must be <= 1.0, got {alpha_value}")
        return float(alpha_value)

    def _resolve_component_outer_lr(self, lr_value: float, arg_name: str) -> float:
        """Resolve per-component outer LR; negative values inherit cdc_outer_lr."""
        if lr_value < 0.0:
            return self.outer_lr
        return float(lr_value)

    @staticmethod
    def _format_component_value(raw_value: float, effective_value: float) -> str:
        if raw_value < 0.0:
            return f"inherit({effective_value})"
        return str(effective_value)

    _format_component_alpha = _format_component_value

    # ------------------------------------------------------------------
    # LR/WD tracking (for DC weight-decay debias)
    # ------------------------------------------------------------------

    def _build_optimizer_param_group_index(self) -> None:
        """Build mapping from optimizer (main) param id to param_group index."""
        mapping: Dict[int, int] = {}
        for group_idx, group in enumerate(self.param_groups):
            for p in group.get("params", []):
                mapping[id(p)] = group_idx
        self._optimizer_param_id_to_group_idx = mapping

    def _get_param_group_index_for_model_param(self, param: torch.nn.Parameter) -> Optional[int]:
        """Return the param_group index for a model param (handles fp16 main params)."""
        if param is None:
            return None
        main_param = getattr(param, "main_param", param)
        return self._optimizer_param_id_to_group_idx.get(id(main_param))

    def _init_lr_wd_tracking(self) -> None:
        """Initialize cumulative LR and log(weight-decay factors) per param group."""
        num_groups = len(self.param_groups)
        self._lr_cumsums_by_group = [0.0 for _ in range(num_groups)]
        self._wd_log_cumsums_by_group = [0.0 for _ in range(num_groups)]

    def _ensure_lr_wd_tracking(self) -> None:
        if self._lr_cumsums_by_group is None or self._wd_log_cumsums_by_group is None:
            self._init_lr_wd_tracking()
            return

    def _record_lr_wd_for_step(self) -> None:
        """Record LR/WD of the just-applied inner step into cumulative trackers."""
        self._ensure_lr_wd_tracking()
        assert self._lr_cumsums_by_group is not None
        assert self._wd_log_cumsums_by_group is not None

        for group_idx, group in enumerate(self.param_groups):
            lr = float(group.get("lr", 0.0))
            wd = float(group.get("weight_decay", 0.0))

            self._lr_cumsums_by_group[group_idx] += lr

            # For AdamW-style decoupled weight decay, the per-step multiplicative factor is:
            #   rho_step = (1 - lr * wd)
            # We accumulate log(rho_step) to later get products over intervals.
            if lr == 0.0 or wd == 0.0:
                continue
            x = lr * wd
            if x >= 1.0:
                # Pathological; treat as zeroing the parameter.
                self._wd_log_cumsums_by_group[group_idx] = float("-inf")
                continue
            self._wd_log_cumsums_by_group[group_idx] += math.log1p(-x)

    def _snapshot_sent_wd_log_cumsums(self, tracker: Dict[str, Any]) -> Dict[int, float]:
        """Snapshot per-group cumulative log(weight-decay factors) at send time."""
        self._ensure_lr_wd_tracking()
        assert self._wd_log_cumsums_by_group is not None

        sent: Dict[int, float] = {}
        for group_idx in tracker.get("unique_param_group_indices", []):
            if group_idx is None:
                continue
            if group_idx < 0 or group_idx >= len(self._wd_log_cumsums_by_group):
                continue
            sent[int(group_idx)] = float(self._wd_log_cumsums_by_group[group_idx])
        return sent

    def _rho_wd_between_send_and_now(self, tracker: Dict[str, Any], group_idx: Optional[int]) -> float:
        """Compute weight-decay-only multiplicative factor rho over (sent_at_step, current_step]."""
        if group_idx is None:
            return 1.0
        self._ensure_lr_wd_tracking()
        assert self._wd_log_cumsums_by_group is not None

        # 越界检查
        if group_idx < 0 or group_idx >= len(self._wd_log_cumsums_by_group):
            return 1.0

        # 发送的时候的wd log累积值
        sent_logs = tracker.get("sent_wd_log_cumsums")
        if not isinstance(sent_logs, dict):
            return 1.0
        sent_log = sent_logs.get(int(group_idx))
        if sent_log is None:
            return 1.0

        # 当前的wd log累积值
        current_log = float(self._wd_log_cumsums_by_group[group_idx])
        sent_log = float(sent_log)

        if math.isinf(current_log) and current_log < 0:
            if math.isinf(sent_log) and sent_log < 0:
                return 1.0
            return 0.0
        if math.isinf(sent_log) and sent_log < 0:
            return 0.0

        # 计算 rho = exp(current_log - sent_log)
        log_rho = current_log - sent_log
        # Avoid underflow in exp for very negative values.
        rho = 0.0 if log_rho < -745.0 else math.exp(log_rho)
        # Clamp to [0, 1] to avoid numerical drift.
        return float(max(0.0, min(1.0, rho)))

    @property
    def is_stub_optimizer(self):
        return getattr(self.inner_optimizer, 'is_stub_optimizer', False)
    
    @property
    def optimizer(self):
        return self.inner_optimizer.optimizer

    @property
    def state(self):
        return self.inner_optimizer.state

    @state.setter
    def state(self, value):
        self.inner_optimizer.state = value
    
    @property
    def main_param_list(self):
        """返回optimizer的param groups展平后的列表"""
        return [p for g in self.param_groups for p in g['params']] 
    
    @property
    def model_param_list(self):
        """返回遍历model_chunks后保存到一个展平列表的params
        named_parameters()会自动去重
        """
        if hasattr(self, "_named_model_param_list"):
            return [p for _, p in self._named_model_param_list]
        model_param_list = []
        for chunk in self.model_chunks:
            for name, p in chunk.named_parameters():
                if p.requires_grad:
                    model_param_list.append(p)
        return model_param_list

    @property
    def tracked_model_param_list(self):
        if hasattr(self, "_tracked_named_model_params"):
            return [p for _, p in self._tracked_named_model_params]
        return self.model_param_list

    def get_loss_scale(self):
        return self.inner_optimizer.get_loss_scale()

    def get_parameters(self):
        return self.inner_optimizer.get_parameters()

    def get_grad_stats_parallel_group(self):
        return self.inner_optimizer.get_grad_stats_parallel_group()

    @torch.no_grad()
    def get_grad_norm(self):
        return self.inner_optimizer.get_grad_norm()

    def clip_grad_norm(self, clip_grad: float):
        return self.inner_optimizer.clip_grad_norm(clip_grad)

    def count_zeros(self):
        return self.inner_optimizer.count_zeros()

    def reload_model_params(self):
        self.inner_optimizer.reload_model_params()
        # When a model checkpoint is loaded without CDC state, Megatron only asks the optimizer to
        # refresh its inner fp32 main params. Keep CDC outer state aligned with the loaded model.
        if self.step_count == 0 and not self._cdc_state_loaded:
            self._reset_outer_state_from_model()

    def state_dict(self, is_loading: bool = False):
        """Return optimizer state plus CDC metadata."""
        return {
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "cdc_state": self._build_cdc_state(),
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state including CDC metadata (compatible with normal checkpoint)."""
        if "inner_optimizer" not in state_dict: # Normal checkpoint without CDC metadata
            self._cdc_state_loaded = False
            self.inner_optimizer.load_state_dict(state_dict)
            self._reset_outer_state_from_model()
        else:   # CDC checkpoint
            self.inner_optimizer.load_state_dict(state_dict["inner_optimizer"])
            self._load_cdc_state(state_dict.get("cdc_state"))
            self._cdc_state_loaded = True

    def zero_grad(self, set_to_none=True):
        self.inner_optimizer.zero_grad(set_to_none)

    def save_parameter_state(self, filename: str):
        self.inner_optimizer.save_parameter_state(filename)

    def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
        self.inner_optimizer.load_parameter_state(
            filename, update_legacy_format=update_legacy_format
        )

    @property
    def param_groups(self):
        """直接复用ChainedOptimizer的param_groups
        [
            { "params": [param_0, param_1, ...], "lr": ..., "weight_decay": ..., ... },
            { "params": [param_k, ...], "lr": ..., ... },
        ]
        param_x: torch.nn.Parameter, 模型参数张量, .grad, .data
        """
        return self.inner_optimizer.param_groups

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _serialize_diloco_state(self):
        """Return original_snapshot list and outer_optimizer state."""
        # Optimization: If we just synced, original_snapshot == local params.
        # We can skip saving it to avoid redundancy.
        snapshot = None
        if not (self.step_count > 0 and self.step_count % self.sync_interval == 0):
            snapshot = [p.detach().to('cpu', copy=True) for p in self.original_snapshot or []]

        outer_state = None
        if self.outer_optimizer is not None:
            outer_state = self._optimizer_state_to_cpu(self.outer_optimizer.state_dict())

        return {
            "original_snapshot": snapshot,
            "outer_optimizer": outer_state,
        }

    def _serialize_lr_wd_tracking(self) -> Dict[str, Any]:
        self._ensure_lr_wd_tracking()
        return {
            "lr_cumsums_by_group": deepcopy(self._lr_cumsums_by_group),
            "wd_log_cumsums_by_group": deepcopy(self._wd_log_cumsums_by_group),
        }

    def _build_cdc_state(self):
        state = {
            "algorithm": self.algorithm,
            "step_count": self.step_count,
            "next_shard_idx": self.next_shard_idx,
            "next_expert_group_idx": self.next_expert_group_idx,
            "expert_sync_event_count": self.expert_sync_event_count,
            "lr_wd_tracking": self._serialize_lr_wd_tracking(),
        }

        if self.algorithm == 'diloco' and getattr(self, "original_snapshot", None) is not None:
            state["diloco"] = self._serialize_diloco_state()
        elif self.algorithm in ['streaming', 'dc'] and getattr(self, "shard_tracker", None) is not None:
            state["streaming_layout_version"] = 2
            state["shards"] = self._serialize_tracker_dict(self.shard_tracker)
            if getattr(self, "router_tracker", None):
                state["router_shards"] = self._serialize_tracker_dict(self.router_tracker)
            if getattr(self, "expert_shard_tracker", None):
                state["expert_shards"] = self._serialize_tracker_dict(self.expert_shard_tracker)

        return state

    def _serialize_tracker_dict(self, tracker_dict):
        shards = []
        if not tracker_dict:
            return shards

        for shard_idx in sorted(tracker_dict.keys()):
            tracker = tracker_dict[shard_idx]

            # Optimization: staged_params is only needed if a sync is in flight.
            # If next_receive_step is 0 (or <= step_count, meaning completed), it's redundant.
            save_staged = tracker["next_receive_step"] > self.step_count

            shard_entry = {
                "shard_idx": shard_idx,
                "display_name": tracker.get("display_name", str(shard_idx)),
                "params": [self._clone_tensor_to_cpu(p) for p in tracker["params"]],
                "staged_params": [self._clone_tensor_to_cpu(p) for p in tracker["staged_params"]] if save_staged else None,
                "sent_at_step": int(tracker["sent_at_step"]),
                "old_sent_at_step": int(tracker["old_sent_at_step"]),
                "next_receive_step": int(tracker["next_receive_step"]),
                "sent_wd_log_cumsums": deepcopy(tracker.get("sent_wd_log_cumsums")) if save_staged else None,
                "global_num_params": int(tracker["global_num_params"]),
                "last_score": float(tracker["last_score"]),
                "last_token_load": float(tracker.get("last_token_load", 0.0)),
                "token_load_accum": float(tracker.get("token_load_accum", 0.0)),
                "sent_at_expert_event": int(tracker.get("sent_at_expert_event", 0)),
                "moe_module_key": tracker.get("moe_module_key"),
                "local_expert_idx": tracker.get("local_expert_idx"),
            }

            if tracker.get("outer_optimizer") is not None:
                shard_entry["outer_optimizer"] = self._optimizer_state_to_cpu(
                    tracker["outer_optimizer"].state_dict()
                )

            shards.append(shard_entry)

        return shards

    def _optimizer_state_to_cpu(self, optimizer_state):
        if optimizer_state is None:
            return None

        cpu_state = {
            "state": {},
            "param_groups": deepcopy(optimizer_state.get("param_groups", [])),
        }

        for key, value in optimizer_state.get("state", {}).items():
            cpu_entry = {}
            for inner_key, inner_value in value.items():
                if torch.is_tensor(inner_value):
                    cpu_entry[inner_key] = self._clone_tensor_to_cpu(inner_value)
                else:
                    cpu_entry[inner_key] = deepcopy(inner_value)
            cpu_state["state"][key] = cpu_entry

        return cpu_state

    def _load_cdc_state(self, cdc_state):
        """Load CDC metadata from checkpoint state.
        Including: step_count, algorithm, next_shard_idx, diloco/shards;
        Diloco: original_snapshot, outer_optimizer;
        Shards: list, each with params, staged_params, sent_at_step, next_receive_step, outer_optimizer.
        """
        if not cdc_state:
            return
        
        checkpoint_algorithm = cdc_state.get("algorithm", self.algorithm)
        if checkpoint_algorithm != self.algorithm:
            raise ValueError(
                f"Checkpoint algorithm {checkpoint_algorithm} does not match runtime algorithm {self.algorithm}."
            )

        self.step_count = cdc_state.get("step_count", self.step_count)
        self.next_shard_idx = cdc_state.get("next_shard_idx", self.next_shard_idx)
        self.next_expert_group_idx = cdc_state.get(
            "next_expert_group_idx", self.next_expert_group_idx
        )
        self.expert_sync_event_count = cdc_state.get(
            "expert_sync_event_count", self.expert_sync_event_count
        )

        # Restore LR/WD tracking (best-effort; older checkpoints may not have it).
        tracking = cdc_state.get("lr_wd_tracking")
        if isinstance(tracking, dict):
            lr_cumsums = tracking.get("lr_cumsums_by_group")
            wd_log_cumsums = tracking.get("wd_log_cumsums_by_group")
            if isinstance(lr_cumsums, list) and isinstance(wd_log_cumsums, list):
                self._lr_cumsums_by_group = [float(x) for x in lr_cumsums]
                self._wd_log_cumsums_by_group = [float(x) for x in wd_log_cumsums]
        self._ensure_lr_wd_tracking()

        if self.algorithm == 'diloco':
            self._load_diloco_state(cdc_state.get("diloco"))
        elif self.algorithm in ['streaming', 'dc']:
            layout_version = int(cdc_state.get("streaming_layout_version", 1))
            if self.enable_moe_router_refresh and layout_version < 2:
                raise ValueError(
                    "This CDC checkpoint uses the legacy hybrid layout with router parameters "
                    "embedded in dense shards. The current code keeps routers in a dedicated "
                    "tracker, so resume from this CDC optimizer state is not supported."
                )
            self._load_tracker_dict(self.shard_tracker, cdc_state.get("shards"))
            if getattr(self, "router_tracker", None) is not None:
                self._load_tracker_dict(self.router_tracker, cdc_state.get("router_shards"))
            if getattr(self, "expert_shard_tracker", None) is not None:
                self._load_tracker_dict(
                    self.expert_shard_tracker, cdc_state.get("expert_shards")
                )

    def _load_diloco_state(self, diloco_state):
        if not diloco_state:
            return

        snapshot = diloco_state.get("original_snapshot", None)

        # Optimization: If snapshot is None, it means it was identical to local params.
        if snapshot is None:
            for target, local in zip(self.original_snapshot, self.tracked_model_param_list):
                self._copy_tensor_data(target, local)
        else:
            for target, saved in zip(self.original_snapshot, snapshot):
                self._copy_tensor_data(target, saved)

        if self.outer_optimizer is not None:
            self.outer_optimizer.load_state_dict(diloco_state["outer_optimizer"])
            device = (
                self.original_snapshot[0].device
                if getattr(self, "original_snapshot", None)
                else torch.device("cpu")
            )
            self._move_optimizer_state_to_device(self.outer_optimizer, device)

    def _load_tracker_dict(self, tracker_dict, shard_states):
        if not shard_states:
            return

        for shard_state in shard_states:
            shard_idx = shard_state["shard_idx"]
            if shard_idx not in tracker_dict:
                raise ValueError(f"Shard {shard_idx} not initialized but present in checkpoint.")

            tracker = tracker_dict[shard_idx]

            self._copy_tensor_list(tracker["params"], shard_state.get("params", []))

            # Optimization: staged_params might be None if it was redundant.
            # In that case, we leave it as initialized (likely zeros or current local),
            # because it will be overwritten by the next _initiate_sync anyway.
            saved_staged = shard_state.get("staged_params")
            if saved_staged is not None:
                self._copy_tensor_list(tracker["staged_params"], saved_staged)

            tracker["sent_at_step"] = shard_state.get("sent_at_step", tracker["sent_at_step"])
            tracker["old_sent_at_step"] = shard_state.get("old_sent_at_step", tracker["old_sent_at_step"])
            tracker["next_receive_step"] = shard_state.get("next_receive_step", tracker["next_receive_step"])
            tracker["sent_wd_log_cumsums"] = shard_state.get(
                "sent_wd_log_cumsums", tracker.get("sent_wd_log_cumsums")
            )
            tracker["global_num_params"] = shard_state.get("global_num_params", tracker["global_num_params"])
            tracker["last_score"] = shard_state.get("last_score", tracker["last_score"])
            tracker["last_token_load"] = shard_state.get(
                "last_token_load", tracker.get("last_token_load", 0.0)
            )
            tracker["token_load_accum"] = shard_state.get(
                "token_load_accum", tracker.get("token_load_accum", 0.0)
            )
            tracker["sent_at_expert_event"] = shard_state.get(
                "sent_at_expert_event", tracker.get("sent_at_expert_event", 0)
            )
            tracker["display_name"] = shard_state.get(
                "display_name", tracker.get("display_name", str(shard_idx))
            )
            tracker["moe_module_key"] = shard_state.get(
                "moe_module_key", tracker.get("moe_module_key")
            )
            tracker["local_expert_idx"] = shard_state.get(
                "local_expert_idx", tracker.get("local_expert_idx")
            )

            outer_state = shard_state.get("outer_optimizer")
            if tracker.get("outer_optimizer") is not None and outer_state is not None:
                tracker["outer_optimizer"].load_state_dict(outer_state)
                device = (
                    tracker["params"][0].device
                    if tracker.get("params") and len(tracker["params"]) > 0
                    else torch.device("cpu")
                )
                self._move_optimizer_state_to_device(tracker["outer_optimizer"], device)

    def _copy_tensor_list(self, target_list, saved_list):
        if len(target_list) != len(saved_list):
            raise ValueError("Mismatch in tensor list lengths while restoring checkpoint state.")
        for target_tensor, saved_tensor in zip(target_list, saved_list):
            self._copy_tensor_data(target_tensor, saved_tensor)

    @staticmethod
    def _copy_tensor_data(target_tensor, saved_tensor):
        """Copy saved_tensor data into target_tensor, handling device and dtype."""
        with torch.no_grad():
            target_tensor.copy_(
                saved_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            )

    def _apply_global_params_to_local(
        self,
        tracker: Dict[str, Any],
        *,
        force_full_copy: bool = False,
    ) -> None:
        """Apply the tracker global state to local model params with the tracker's alpha."""
        alpha = float(tracker.get("apply_alpha", self.streaming_alpha))
        if force_full_copy:
            alpha = 0.0
        if alpha == 1.0:
            return

        for p_local, p_global in zip(tracker["param_refs"], tracker["params"]):
            if alpha == 0.0:
                self._copy_tensor_data(p_local.data, p_global.data)
                continue

            p_global_data = p_global.data.to(device=p_local.device, dtype=torch.float32)
            blended = (
                p_local.data.to(torch.float32).mul(alpha).add_(
                    p_global_data, alpha=1.0 - alpha
                )
            )
            p_local.data.copy_(blended.to(dtype=p_local.dtype))

    @staticmethod
    def _clone_tensor_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
        """Detach and clone a tensor to CPU for checkpointing."""
        if tensor is None:
            return None
        if not torch.is_tensor(tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(tensor)}")
        with torch.no_grad():
            return tensor.detach().to(device="cpu").clone()

    @staticmethod
    def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        """Move optimizer state tensors to the given device (for restoring CPU-saved state)."""
        if optimizer is None:
            return
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)

    def _clone_param_for_outer_state(self, param: torch.nn.Parameter) -> torch.Tensor:
        target_device = torch.device("cpu") if self.offload_outer_opt else param.device
        return param.detach().to(
            device=target_device, dtype=self.outer_state_dtype, copy=True
        )

    @staticmethod
    def _is_routed_expert_param_name(param_name: str) -> bool:
        return '.experts.' in param_name and '.shared_experts.' not in param_name

    @staticmethod
    def _is_moe_router_param_name(param_name: str) -> bool:
        return '.router.' in param_name

    @staticmethod
    def _routed_expert_group_key(param_name: str) -> Optional[str]:
        match = re.search(r"(.*?\.experts\.local_experts\.\d+)\.", param_name)
        if match is None:
            return None
        return match.group(1)

    def _collect_routed_expert_named_params(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        return [
            (name, param)
            for name, param in named_params
            if self._is_routed_expert_param_name(name)
        ]

    def _collect_router_named_params(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        return [
            (name, param)
            for name, param in named_params
            if self._is_moe_router_param_name(name)
        ]

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        normalized = name
        while normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        return normalized

    def _extract_expert_group_metadata(self, group_name: str) -> Tuple[str, Optional[int], Optional[int]]:
        normalized = self._normalize_module_name(group_name)
        module_match = re.match(r"(.*)\.experts\.local_experts\.(\d+)$", normalized)
        if module_match is None:
            return normalized, None, None

        module_key = module_match.group(1)
        expert_idx = int(module_match.group(2))
        layer_idx = self._parse_layer_index(module_key)
        return module_key, layer_idx, expert_idx

    def _expert_group_sort_key(self, group_name: str) -> Tuple[int, int, str]:
        _, layer_idx, expert_idx = self._extract_expert_group_metadata(group_name)
        return (
            layer_idx if layer_idx is not None else 10**9,
            expert_idx if expert_idx is not None else 10**9,
            self._normalize_module_name(group_name),
        )

    @staticmethod
    def _iter_named_modules_unique(
        model_chunks: List[MegatronModule],
    ) -> List[Tuple[str, torch.nn.Module]]:
        results: List[Tuple[str, torch.nn.Module]] = []
        seen_ids = set()
        for chunk in model_chunks:
            for name, module in chunk.named_modules():
                mid = id(module)
                if mid in seen_ids:
                    continue
                seen_ids.add(mid)
                results.append((name, module))
        return results

    def _build_moe_module_index(self) -> Dict[str, torch.nn.Module]:
        module_index: Dict[str, torch.nn.Module] = {}
        for name, module in self._iter_named_modules_unique(self.model_chunks):
            normalized_name = self._normalize_module_name(name)
            if not normalized_name:
                continue
            if hasattr(module, "token_dispatcher") and hasattr(module, "experts"):
                module_index[normalized_name] = module
        return module_index

    def _lookup_moe_module(self, module_key: str) -> Optional[torch.nn.Module]:
        if not module_key:
            return None
        module = self._moe_module_index.get(module_key)
        if module is not None:
            return module

        suffix_matches = [
            candidate_module
            for candidate_key, candidate_module in self._moe_module_index.items()
            if candidate_key.endswith(module_key) or module_key.endswith(candidate_key)
        ]
        if len(suffix_matches) == 1:
            return suffix_matches[0]
        return None

    @staticmethod
    def _tensor_like_to_float_list(value: Any) -> List[float]:
        if value is None:
            return []
        if torch.is_tensor(value):
            tensor = value.detach().to(dtype=torch.float32)
            if tensor.dim() == 1:
                return [float(v) for v in tensor.cpu().tolist()]
            reduced = tensor.reshape(-1, tensor.shape[-1]).sum(dim=0)
            return [float(v) for v in reduced.cpu().tolist()]
        if isinstance(value, (list, tuple)):
            return [float(v) for v in value]
        if hasattr(value, "tolist"):
            raw = value.tolist()
            if isinstance(raw, list):
                if raw and isinstance(raw[0], list):
                    if not raw[0]:
                        return []
                    cols = len(raw[0])
                    reduced = [0.0 for _ in range(cols)]
                    for row in raw:
                        for idx, item in enumerate(row):
                            reduced[idx] += float(item)
                    return reduced
                return [float(v) for v in raw]
        return []

    def _filter_named_params_for_cdc(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        if self.moe_param_mode == 'all':
            return list(named_params)
        if self.moe_param_mode not in {'dense-only', 'dense-expert-hybrid'}:
            raise ValueError(f"Unknown cdc_moe_param_mode: {self.moe_param_mode}")

        tracked: List[Tuple[str, torch.nn.Parameter]] = []
        excluded_tensors = 0
        excluded_numel = 0
        router_tensors = 0
        router_numel = 0

        for name, param in named_params:
            if self._is_routed_expert_param_name(name):
                excluded_tensors += 1
                excluded_numel += param.numel()
                continue
            if self.enable_moe_router_refresh and self._is_moe_router_param_name(name):
                router_tensors += 1
                router_numel += param.numel()
                continue
            tracked.append((name, param))

        if not tracked:
            raise ValueError(
                f"cdc_moe_param_mode={self.moe_param_mode} excluded every trainable parameter. "
                "Check the model structure and CDC param filter."
            )

        if self.verbose and excluded_tensors > 0:
            tracked_numel = sum(param.numel() for _, param in tracked)
            excluded_note = (
                "kept for expert refresh"
                if self.moe_param_mode == 'dense-expert-hybrid'
                else "kept_local_only"
            )
            print_rank_0(
                f"[CDC] MoE {self.moe_param_mode} mode excludes routed expert params from the "
                f"dense rolling queue: "
                f"{excluded_tensors} tensors, {excluded_numel} params {excluded_note}; "
                f"router_dedicated={router_tensors} tensors/{router_numel} params; "
                f"CDC tracks {len(tracked)} tensors, {tracked_numel} params."
            )

        return tracked

    def _group_routed_expert_named_params(
        self,
    ) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
        grouped: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {}
        for name, param in self._expert_named_model_params:
            group_key = self._routed_expert_group_key(name)
            if group_key is None:
                continue
            grouped.setdefault(group_key, []).append((name, param))
        return grouped

    def _build_tracker(
        self,
        param_refs: List[torch.nn.Parameter],
        *,
        tp_group,
        pp_group,
        display_name: str,
        moe_module_key: Optional[str] = None,
        local_expert_idx: Optional[int] = None,
        comm_dtype: Optional[torch.dtype] = None,
        apply_alpha: Optional[float] = None,
        outer_lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        tracker_outer_lr = self.outer_lr if outer_lr is None else float(outer_lr)
        param_group_indices = [
            self._get_param_group_index_for_model_param(p) for p in param_refs
        ]
        unique_param_group_indices = sorted(
            {i for i in param_group_indices if i is not None}
        )

        tracker = {
            "display_name": display_name,
            "param_refs": param_refs,
            "param_group_indices": param_group_indices,
            "unique_param_group_indices": unique_param_group_indices,
            "sent_wd_log_cumsums": None,
            "params": [],
            "staged_params": [],
            "comm_dtype": comm_dtype if comm_dtype is not None else self.outer_comm_dtype,
            "apply_alpha": self.streaming_alpha if apply_alpha is None else float(apply_alpha),
            "outer_lr": tracker_outer_lr,
            "sent_at_step": 0,
            "old_sent_at_step": 0,
            "next_receive_step": 0,
            "global_num_params": 0,
            "last_score": 0.0,
            "last_token_load": 0.0,
            "token_load_accum": 0.0,
            "sent_at_expert_event": 0,
            "moe_module_key": moe_module_key,
            "local_expert_idx": local_expert_idx,
        }

        for p in param_refs:
            tracker["params"].append(self._clone_param_for_outer_state(p))
            tracker["staged_params"].append(self._clone_param_for_outer_state(p))

        if tracker_outer_lr != 1.0 and len(tracker["params"]) > 0:
            for p in tracker["params"]:
                p.requires_grad_(True)
            tracker["outer_optimizer"] = SGD(
                tracker["params"],
                lr=tracker_outer_lr,
                momentum=0.9,
                nesterov=True,
            )
        else:
            tracker["outer_optimizer"] = None

        local_numel = sum(p.numel() for p in param_refs)
        global_numel = self._all_reduce_scalar_sum(local_numel, group=tp_group)
        global_numel = self._all_reduce_scalar_sum(global_numel, group=pp_group)
        tracker["global_num_params"] = int(global_numel)

        if len(tracker["params"]) > 0:
            self._all_reduce_flattened(
                [t.data for t in tracker["params"]],
                communication_dtype=tracker["comm_dtype"],
            )

        return tracker

    def _reset_tracker_dict_from_model(self, tracker_dict: Optional[Dict[int, Dict[str, Any]]]) -> None:
        if not tracker_dict:
            return

        for tracker in tracker_dict.values():
            for p_global, p_local in zip(tracker["params"], tracker["param_refs"]):
                self._copy_tensor_data(p_global, p_local.data)
            for p_staged, p_local in zip(tracker["staged_params"], tracker["param_refs"]):
                self._copy_tensor_data(p_staged, p_local.data)

            tracker["sent_wd_log_cumsums"] = None
            tracker["sent_at_step"] = 0
            tracker["old_sent_at_step"] = 0
            tracker["next_receive_step"] = 0
            tracker["last_score"] = 0.0
            tracker["last_token_load"] = 0.0
            tracker["token_load_accum"] = 0.0
            tracker["sent_at_expert_event"] = 0

            if tracker["outer_optimizer"] is not None:
                tracker["outer_optimizer"].state.clear()

            if tracker["params"]:
                self._all_reduce_flattened(
                    [t.data for t in tracker["params"]],
                    communication_dtype=tracker.get("comm_dtype", self.outer_comm_dtype),
                )

    def _reset_outer_state_from_model(self) -> None:
        """Reinitialize CDC outer state from the current model parameters."""
        self.step_count = 0
        self.next_shard_idx = 0
        self.next_expert_group_idx = 0
        self.expert_sync_event_count = 0
        self._init_lr_wd_tracking()

        if self.algorithm == 'diloco' and getattr(self, "original_snapshot", None) is not None:
            for target, local in zip(self.original_snapshot, self.tracked_model_param_list):
                self._copy_tensor_data(target, local.data)
            if self.original_snapshot:
                self._all_reduce_flattened(
                    self.original_snapshot, communication_dtype=self.outer_comm_dtype
                )
            if self.outer_optimizer is not None:
                self.outer_optimizer.state.clear()
            return

        if self.algorithm not in ['streaming', 'dc'] or getattr(self, "shard_tracker", None) is None:
            return

        self._reset_tracker_dict_from_model(self.shard_tracker)
        self._reset_tracker_dict_from_model(getattr(self, "router_tracker", None))
        self._reset_tracker_dict_from_model(getattr(self, "expert_shard_tracker", None))

    def _init_diloco_state(self):
        """Initialize state for standard DiLoCo."""
        self.original_snapshot = []  # 上一次同步时的模型参数快照（展平列表）

        for param in self.tracked_model_param_list:
            self.original_snapshot.append(self._clone_param_for_outer_state(param).requires_grad_(True))

        # Initialize outer optimizer (Nesterov SGD)
        if self.outer_lr != 1.0:
            self.outer_optimizer = SGD(
                self.original_snapshot,
                lr=self.outer_lr,
                momentum=0.9,
                nesterov=True,
            )
        else:
            self.outer_optimizer = None

    def _init_streaming_state(self):
        """
        Initialize state for Streaming/DC DiLoCo.
        一个模型的结构：
        Embeddings:
        - embedding.word_embeddings.weight
        - embedding.position_embeddings.weight(RoPE 没有)
        Decoder layers:
        - decoder.layers.<0-N>.
            - self_attention.
                - linear_proj.weight
                - linear_qkv.
                    - weight
                    - bias (disable了)
                    - layer_norm_weight
            - mlp.fc1.
                - layer_norm_weight
                - weight
                - bias (disable了)
            - mlp.fc2.
                - weight
                - bias (disable了)
        - decoder.final_layernorm.weight
        Output Layer:
        - output_layer.weight (tied or untied)
        """
        # Build a deterministic shard plan from parameter names.
        # Note:
        # - CDC group is formed across outer-DP ranks that share the same PP/TP partition.
        # - To get a *global* shard selection across pipeline stages, we aggregate per-shard metadata
        #   (num_params, score) across TP and PP groups using scalar all-reduces.

        if self.num_shards < 1:
            raise ValueError(f"cdc_num_shards must be >= 1, got {self.num_shards}")

        # Number of embedding shards.
        # tied: one shard for embeddings (output weights are tied and will only appear once).
        # untied: two shards: input embeddings, and output embedding/LM head.
        embedding_shards = 1 if self.tie_embeddings else 2
        if self.num_shards < embedding_shards:
            raise ValueError(
                f"cdc_num_shards ({self.num_shards}) must be >= {embedding_shards} "
                f"for tie_embeddings={self.tie_embeddings}."
            )

        # Remaining shards are used for decoder layers.
        decoder_shards = max(self.num_shards - embedding_shards, 0)

        # Determine total number of decoder layers (global) from args, fallback to local discovery.
        args = get_args()
        num_layers = getattr(args, 'num_layers', None)
        if num_layers is None:
            num_layers = getattr(args, 'decoder_num_layers', None)
        if num_layers is None:
            # Fallback: infer from local parameter names.
            local_layer_ids = set()
            for name, _ in self._iter_named_trainable_params_unique(self.model_chunks):
                layer_idx = self._parse_layer_index(name)
                if layer_idx is not None:
                    local_layer_ids.add(layer_idx)
            num_layers = (max(local_layer_ids) + 1) if local_layer_ids else 0

        if decoder_shards == 0 and num_layers > 0:
            raise ValueError(
                f"cdc_num_shards ({self.num_shards}) is insufficient: embeddings use {embedding_shards} shard(s) "
                f"and decoder layers require at least 1 more shard (num_layers={num_layers})."
            )

        if self.shard_pattern not in ['stride', 'sequential']:
            raise ValueError(f"Unknown cdc_shard_pattern: {self.shard_pattern}")

        # Build local shard assignment: shard_idx -> list[param_ref]
        shard_to_param_refs: Dict[int, List[torch.nn.Parameter]] = {i: [] for i in range(self.num_shards)}
        unique_named_params = list(self._tracked_named_model_params)
        for name, param in unique_named_params:
            # 如果tie embedding，只用一个 shard 存output layer和input layer
            # 否则用 0 存 embedding，1 存 output layer
            shard_idx = self._assign_param_to_shard(
                name=name,
                num_layers=num_layers,
                embedding_shards=embedding_shards,
                decoder_shards=decoder_shards,
            )
            shard_to_param_refs[shard_idx].append(param)

        if self.verbose:
            assigned = sum(len(v) for v in shard_to_param_refs.values())
            expected = len(unique_named_params)
            if assigned != expected:
                print_rank_0(f"[CDC] Warning: Assigned {assigned} params to shards, expected {expected}.")

        # Initialize trackers.
        self.shard_tracker = {}
        self.router_tracker = {}
        self.next_shard_idx = 0
        self.expert_shard_tracker = {}
        self.next_expert_group_idx = 0

        tp_group = mpu.get_tensor_model_parallel_group()
        pp_group = mpu.get_pipeline_model_parallel_group()

        for shard_idx in range(self.num_shards):
            param_refs = shard_to_param_refs.get(shard_idx, [])
            tracker = self._build_tracker(
                param_refs,
                tp_group=tp_group,
                pp_group=pp_group,
                display_name=f"dense-shard-{shard_idx}",
                apply_alpha=self.dense_alpha,
                outer_lr=self.dense_outer_lr,
            )

            self.shard_tracker[shard_idx] = tracker

            if self.verbose:
                print_rank_0(
                    f"[CDC] Shard {shard_idx} initialized: local_tensors={len(param_refs)}, "
                    f"global_numel={tracker['global_num_params']}"
                )

        if self.enable_moe_router_refresh:
            router_param_refs = [param for _, param in self._router_named_model_params]
            router_tracker = self._build_tracker(
                router_param_refs,
                tp_group=tp_group,
                pp_group=pp_group,
                display_name="moe-router",
                comm_dtype=torch.float32,
                apply_alpha=self.router_alpha,
                outer_lr=self.dense_outer_lr,
            )
            self.router_tracker[0] = router_tracker

            if self.verbose:
                print_rank_0(
                    f"[CDC] Router tracker initialized: local_tensors={len(router_param_refs)}, "
                    f"global_numel={router_tracker['global_num_params']}"
                )

        if self.has_moe_expert_trackers:
            expert_groups = self._group_routed_expert_named_params()
            if not expert_groups:
                raise ValueError(
                    "cdc_moe_param_mode=dense-expert-hybrid requested expert refresh, "
                    "but no routed expert parameter groups were discovered."
                )

            for expert_idx, expert_group_name in enumerate(
                sorted(expert_groups.keys(), key=self._expert_group_sort_key)
            ):
                expert_param_refs = [param for _, param in expert_groups[expert_group_name]]
                moe_module_key, _, local_expert_idx = self._extract_expert_group_metadata(
                    expert_group_name
                )
                tracker = self._build_tracker(
                    expert_param_refs,
                    tp_group=tp_group,
                    pp_group=pp_group,
                    display_name=expert_group_name,
                    moe_module_key=moe_module_key,
                    local_expert_idx=local_expert_idx,
                    apply_alpha=self.expert_alpha,
                    outer_lr=self.expert_outer_lr,
                )
                self.expert_shard_tracker[expert_idx] = tracker

                if self.verbose:
                    print_rank_0(
                        f"[CDC] Expert group {expert_idx} ({expert_group_name}) initialized: "
                        f"local_tensors={len(expert_param_refs)}, "
                        f"global_numel={tracker['global_num_params']}"
                    )
                    if expert_idx < 8:
                        print_rank_0(
                            f"[CDC][MoETracker] group={expert_group_name} "
                            f"module_key={moe_module_key} local_expert_idx={local_expert_idx}"
                        )

    @staticmethod
    def _tracker_has_any_params(tracker: Dict[str, Any]) -> bool:
        return bool(tracker.get("param_refs")) or bool(tracker.get("params"))

    @staticmethod
    def _parse_layer_index(param_name: str) -> Optional[int]:
        # Support common Megatron naming patterns:
        # - "...layers.0...."
        # - "...decoder.layers.0...."
        m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", param_name)
        if m is None:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _is_embedding_param_name(param_name: str) -> bool:
        # Input embedding-like params. Keep this broad but avoid matching decoder layers.
        if '.layers.' in param_name:
            return False
        keys = (
            'embedding',
            'word_embeddings',
            'position_embeddings',
            'tok_embeddings',
        )
        return any(k in param_name for k in keys)

    def _assign_param_to_shard(
        self,
        *,
        name: str,
        num_layers: int,
        embedding_shards: int,
        decoder_shards: int,
    ) -> int:
        # 1) Embeddings
        if "output_layer" in name or "lm_head" in name:
            # If embeddings/output are tied, treat output layer as the embedding shard.
            # If untied, give output layer its own shard.
            return 0 if self.tie_embeddings else 1

        if self._is_embedding_param_name(name):
            return 0

        # 2) Decoder layers
        layer_idx = self._parse_layer_index(name)
        if layer_idx is not None and num_layers > 0 and decoder_shards > 0:
            decoder_shard_idx = self._layer_to_decoder_shard(layer_idx, num_layers, decoder_shards)
            return embedding_shards + decoder_shard_idx

        # 3) Misc (final norm, output bias, etc.)
        # Keep embeddings isolated: place misc into the last shard if possible.
        return max(self.num_shards - 1, 0)

    def _layer_to_decoder_shard(self, layer_idx: int, num_layers: int, decoder_shards: int) -> int:
        if decoder_shards <= 0:
            return 0
        if self.shard_pattern == 'stride':
            return layer_idx % decoder_shards

        # sequential: contiguous balanced partition
        base = num_layers // decoder_shards
        rem = num_layers % decoder_shards
        # First 'rem' shards get (base+1) layers
        # Determine shard by walking boundaries.
        boundary = 0
        for s in range(decoder_shards):
            size = base + (1 if s < rem else 0)
            next_boundary = boundary + size
            if boundary <= layer_idx < next_boundary:
                return s
            boundary = next_boundary
        return decoder_shards - 1

    @staticmethod
    def _iter_named_trainable_params_unique(
        model_chunks: List[MegatronModule],
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        # named_parameters() already removes duplicates by default, but be extra safe across chunks.
        results: List[Tuple[str, torch.nn.Parameter]] = []
        seen_ids = set()
        for chunk in model_chunks:
            for name, p in chunk.named_parameters():
                if not getattr(p, 'requires_grad', False):
                    continue
                pid = id(p)
                if pid in seen_ids:
                    continue
                seen_ids.add(pid)
                results.append((name, p))
        return results

    def _all_reduce_scalar_sum(self, value: float, group) -> float:
        if group is None:
            return float(value)
        try:
            if dist.get_world_size(group=group) <= 1:
                return float(value)
        except Exception:
            return float(value)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        t = torch.tensor(float(value), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        return float(t.item())

    def _all_reduce_vector_sum(self, values: List[float], group) -> List[float]:
        if not values:
            return []
        if group is None:
            return [float(v) for v in values]
        try:
            if dist.get_world_size(group=group) <= 1:
                return [float(v) for v in values]
        except Exception:
            return [float(v) for v in values]

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        tensor = torch.tensor(values, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return [float(v) for v in tensor.cpu().tolist()]

    def _collect_current_expert_token_loads(self) -> Dict[int, float]:
        if not self.enable_moe_expert_refresh or not getattr(self, "expert_shard_tracker", None):
            return {}

        module_load_cache: Dict[str, List[float]] = {}
        loads: Dict[int, float] = {}

        for tracker_idx, tracker in self.expert_shard_tracker.items():
            module_key = tracker.get("moe_module_key")
            local_expert_idx = tracker.get("local_expert_idx")
            load_value = 0.0

            if module_key is not None and local_expert_idx is not None:
                if module_key not in module_load_cache:
                    module = self._lookup_moe_module(module_key)
                    expert_loads: List[float] = []
                    if module is not None:
                        token_dispatcher = getattr(module, "token_dispatcher", None)
                        load_tensor = None
                        if token_dispatcher is not None:
                            local_map = getattr(token_dispatcher, "local_map", None)
                            if torch.is_tensor(local_map):
                                load_tensor = local_map.sum(dim=0)
                            load_tensor = getattr(
                                token_dispatcher, "num_global_tokens_per_local_expert", None
                            ) if load_tensor is None else load_tensor
                            if load_tensor is None:
                                load_tensor = getattr(
                                    token_dispatcher, "num_global_tokens_per_local_expert_cpu", None
                                )
                            if load_tensor is None:
                                load_tensor = getattr(token_dispatcher, "tokens_per_expert", None)
                            if load_tensor is None and hasattr(
                                token_dispatcher, "get_number_of_tokens_per_expert"
                            ):
                                getter = getattr(token_dispatcher, "get_number_of_tokens_per_expert")
                                if callable(getter):
                                    load_tensor = getter()

                        expert_loads = self._tensor_like_to_float_list(load_tensor)
                        if self.verbose and not self._token_load_source_debug_printed:
                            dispatcher_fields = {
                                "module_key": module_key,
                                "dispatcher_type": type(token_dispatcher).__name__
                                if token_dispatcher is not None
                                else "None",
                                "has_num_global_tokens_per_local_expert": hasattr(
                                    token_dispatcher, "num_global_tokens_per_local_expert"
                                )
                                if token_dispatcher is not None
                                else False,
                                "has_num_global_tokens_per_local_expert_cpu": hasattr(
                                    token_dispatcher, "num_global_tokens_per_local_expert_cpu"
                                )
                                if token_dispatcher is not None
                                else False,
                                "has_tokens_per_expert": hasattr(
                                    token_dispatcher, "tokens_per_expert"
                                )
                                if token_dispatcher is not None
                                else False,
                                "load_len": len(expert_loads),
                                "load_sum": float(sum(expert_loads)) if expert_loads else 0.0,
                                "load_preview": expert_loads[:8],
                            }
                            print_rank_0(f"[CDC][TokenLoadSource] {dispatcher_fields}")
                            self._token_load_source_debug_printed = True
                    module_load_cache[module_key] = expert_loads

                expert_loads = module_load_cache.get(module_key, [])
                if 0 <= int(local_expert_idx) < len(expert_loads):
                    load_value = float(expert_loads[int(local_expert_idx)])

            loads[tracker_idx] = load_value

        return loads

    def _update_moe_expert_token_load_stats(self) -> None:
        if not self.track_expert_token_load:
            return

        current_loads = self._collect_current_expert_token_loads()
        if not current_loads:
            return

        for tracker_idx, load_value in current_loads.items():
            tracker = self.expert_shard_tracker[tracker_idx]
            tracker["last_token_load"] = float(load_value)
            tracker["token_load_accum"] = float(tracker.get("token_load_accum", 0.0) + load_value)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        1. Inner step (Megatron).
        2. Outer step:
           - DiLoCo: sync every sync_interval steps.
           - Streaming/DC: check pending receives every step and initiate sync every sync_interval steps.
        """
        # 1. Inner Step
        update_successful, grad_norm, num_zeros_in_grad = self.inner_optimizer.step()

        if update_successful:
            self.step_count += 1
            self._record_lr_wd_for_step()
            if self.verbose and not self._token_load_step_debug_printed:
                print_rank_0(
                    f"[CDC][TokenLoadStep] step={self.step_count} "
                    f"track_expert_token_load={self.track_expert_token_load} "
                    f"enable_moe_expert_refresh={self.enable_moe_expert_refresh} "
                    f"expert_tracker_count={len(getattr(self, 'expert_shard_tracker', {}))}"
                )
                self._token_load_step_debug_printed = True
            if self.track_expert_token_load:
                self._update_moe_expert_token_load_stats()

            # 2. Outer Step
            if self.algorithm == 'diloco':
                if self.step_count % self.sync_interval == 0:
                    start_time = time.time()
                    self._sync_diloco()
                    duration = time.time() - start_time
                    if self.verbose:
                        print_rank_0(
                            f"[CDC] Step {self.step_count}: Outer sync completed in {duration:.4f}s."
                        )
            elif self.algorithm in ['streaming', 'dc']:
                # Always check pending receives to respect per-step delay semantics.
                self._sync_streaming()

        return update_successful, grad_norm, num_zeros_in_grad

    @torch.no_grad()
    def _sync_diloco(self):
        """Standard DiLoCo synchronization."""
        # Calculate pseudo-gradients: G = Original - Current
        all_grads = []
        for snap_param, model_param in zip(self.original_snapshot, self.tracked_model_param_list):
            if snap_param.grad is None:
                snap_param.grad = torch.zeros_like(snap_param.data)
            model_data = model_param.data
            if model_data.device != snap_param.device or model_data.dtype != snap_param.dtype:
                model_data = model_data.to(device=snap_param.device, dtype=snap_param.dtype)
            snap_param.grad.copy_(snap_param.data - model_data)
            all_grads.append(snap_param.grad)

        # Batch All-Reduce for efficiency
        self._all_reduce_flattened(all_grads, communication_dtype=self.outer_comm_dtype)

        # Outer Optimizer Step
        if self.outer_optimizer:
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad(set_to_none=True)
        else:
            # Simple averaging
            for snap_param in self.original_snapshot:
                snap_param.data.sub_(snap_param.grad)
                snap_param.grad = None

        # Copy back to current model parameters
        for updated_param, curr_param in zip(self.original_snapshot, self.tracked_model_param_list):
            curr_param.copy_(updated_param.to(device=curr_param.device, dtype=curr_param.dtype))

        # Keep optimizer main params in sync with model params for mixed precision.
        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()
        
    def _sync_streaming(self):
        """Unified synchronization step for Streaming and DC."""
        self._complete_due_tracker_syncs(self.shard_tracker, tracker_kind='dense-shard')
        if self.enable_moe_router_refresh:
            self._complete_due_tracker_syncs(self.router_tracker, tracker_kind='router')
        if self.enable_moe_expert_refresh:
            self._complete_due_expert_syncs_batched()
        if self._should_run_blocking_full_sync():
            self._run_blocking_full_sync()
            return

        if self.step_count % self.sync_interval == 0:
            shard_idx = self._select_next_shard()
            self._initiate_tracker_sync(
                self.shard_tracker, shard_idx, tracker_kind='dense-shard'
            )
            if self.enable_moe_router_refresh:
                router_tracker = self.router_tracker.get(0)
                if router_tracker is not None and router_tracker.get("next_receive_step", 0) <= self.step_count:
                    self._initiate_tracker_sync(
                        self.router_tracker, 0, tracker_kind='router'
                    )
                elif self.verbose:
                    print_rank_0(
                        f"[CDC] Step {self.step_count}: Skip router sync because previous "
                        f"router sync is still in flight until step "
                        f"{router_tracker.get('next_receive_step', 0) if router_tracker is not None else 0}."
                    )
            return

        if self._should_sync_expert_group():
            expert_group_indices = self._select_next_expert_groups()
            if expert_group_indices:
                expert_event_index = self.expert_sync_event_count + 1
                batch_immediate_completion = self.delay == 0 and len(expert_group_indices) > 1
                for expert_group_idx in expert_group_indices:
                    self._initiate_tracker_sync(
                        self.expert_shard_tracker,
                        expert_group_idx,
                        tracker_kind='expert-group',
                        expert_event_index=expert_event_index,
                        defer_completion=batch_immediate_completion,
                    )
                self.expert_sync_event_count = expert_event_index
                if batch_immediate_completion:
                    self._complete_tracker_sync_batch(
                        self.expert_shard_tracker,
                        expert_group_indices,
                        tracker_kind='expert-group',
                    )
                    for expert_group_idx in expert_group_indices:
                        self.expert_shard_tracker[expert_group_idx]["next_receive_step"] = 0

    @torch.no_grad()
    def _run_blocking_full_sync(self) -> None:
        dense_indices = [
            idx
            for idx, tracker in self.shard_tracker.items()
            if self._tracker_has_any_params(tracker)
        ]
        router_indices = [
            idx
            for idx, tracker in getattr(self, "router_tracker", {}).items()
            if self._tracker_has_any_params(tracker)
        ]
        expert_indices = [
            idx
            for idx, tracker in getattr(self, "expert_shard_tracker", {}).items()
            if self._tracker_has_any_params(tracker)
        ]

        total_tracker_count = len(dense_indices) + len(router_indices) + len(expert_indices)
        if total_tracker_count == 0:
            return

        start_time = time.time()
        canceled = 0
        canceled += self._cancel_pending_tracker_syncs(self.shard_tracker)
        canceled += self._cancel_pending_tracker_syncs(getattr(self, "router_tracker", None))
        canceled += self._cancel_pending_tracker_syncs(getattr(self, "expert_shard_tracker", None))

        expert_event_index = None
        if expert_indices:
            self.expert_sync_event_count += 1
            expert_event_index = self.expert_sync_event_count

        if self.verbose:
            print_rank_0(
                f"[CDC] Step {self.step_count}: Starting blocking full sync "
                f"(dense={len(dense_indices)}, router={len(router_indices)}, "
                f"expert={len(expert_indices)}, canceled_inflight={canceled})."
            )

        for tracker_idx in dense_indices:
            self._stage_tracker_sync(
                self.shard_tracker,
                tracker_idx,
                tracker_kind='dense-shard',
            )
            self._complete_tracker_sync(
                self.shard_tracker,
                tracker_idx,
                tracker_kind='dense-shard',
                reload_main_params=False,
                use_algorithm_specific_update=False,
                force_full_copy=True,
            )

        for tracker_idx in router_indices:
            self._stage_tracker_sync(
                self.router_tracker,
                tracker_idx,
                tracker_kind='router',
            )
            self._complete_tracker_sync(
                self.router_tracker,
                tracker_idx,
                tracker_kind='router',
                reload_main_params=False,
                use_algorithm_specific_update=False,
                force_full_copy=True,
            )

        for tracker_idx in expert_indices:
            self._stage_tracker_sync(
                self.expert_shard_tracker,
                tracker_idx,
                tracker_kind='expert-group',
                expert_event_index=expert_event_index,
            )
            self._complete_tracker_sync(
                self.expert_shard_tracker,
                tracker_idx,
                tracker_kind='expert-group',
                reload_main_params=False,
                use_algorithm_specific_update=False,
                force_full_copy=True,
            )

        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()

        duration = time.time() - start_time
        if self.verbose:
            print_rank_0(
                f"[CDC] Step {self.step_count}: Blocking full sync completed in "
                f"{duration:.4f}s."
            )

    def _complete_due_expert_syncs_batched(self) -> None:
        if not getattr(self, "expert_shard_tracker", None):
            return

        due_event_to_trackers: Dict[int, List[int]] = {}
        for tracker_idx, tracker in self.expert_shard_tracker.items():
            next_receive_step = int(tracker.get("next_receive_step", 0))
            if next_receive_step <= 0 or self.step_count < next_receive_step:
                continue

            event_id = int(tracker.get("sent_at_expert_event", 0))
            if event_id <= 0:
                event_id = -(tracker_idx + 1)
            due_event_to_trackers.setdefault(event_id, []).append(tracker_idx)

        for event_id in sorted(due_event_to_trackers.keys()):
            tracker_indices = sorted(due_event_to_trackers[event_id])
            if len(tracker_indices) == 1:
                self._complete_tracker_sync(
                    self.expert_shard_tracker,
                    tracker_indices[0],
                    tracker_kind='expert-group',
                    reload_main_params=False,
                )
                if self.mixed_precision:
                    self.inner_optimizer.reload_model_params()
            else:
                self._complete_tracker_sync_batch(
                    self.expert_shard_tracker,
                    tracker_indices,
                    tracker_kind='expert-group',
                )

            for tracker_idx in tracker_indices:
                self.expert_shard_tracker[tracker_idx]["next_receive_step"] = 0

    def _complete_due_tracker_syncs(
        self, tracker_dict: Optional[Dict[int, Dict[str, Any]]], *, tracker_kind: str
    ) -> None:
        if not tracker_dict:
            return

        for tracker_idx, tracker in tracker_dict.items():
            if tracker["next_receive_step"] > 0 and self.step_count >= tracker["next_receive_step"]:
                self._complete_tracker_sync(
                    tracker_dict, tracker_idx, tracker_kind=tracker_kind
                )
                tracker["next_receive_step"] = 0

    def _next_nonempty_tracker_index(
        self, tracker_dict: Dict[int, Dict[str, Any]], start_idx: int
    ) -> Optional[int]:
        if not tracker_dict:
            return None
        tracker_count = len(tracker_dict)
        if tracker_count <= 0:
            return None

        for offset in range(tracker_count):
            idx = (start_idx + offset) % tracker_count
            tracker = tracker_dict[idx]
            if self._tracker_has_any_params(tracker):
                return idx
        return None

    def _should_sync_expert_group(self) -> bool:
        if not self.enable_moe_expert_refresh:
            return False
        if not getattr(self, "expert_shard_tracker", None):
            return False
        if self.expert_sync_interval <= 0:
            return False
        shifted_step = self.step_count - self.expert_sync_offset
        return shifted_step >= 0 and shifted_step % self.expert_sync_interval == 0

    def _should_run_blocking_full_sync(self) -> bool:
        if self.blocking_full_sync_steps <= 0:
            return False
        if self.step_count <= 0:
            return False
        return self.step_count % self.blocking_full_sync_steps == 0

    def _cancel_pending_tracker_syncs(
        self,
        tracker_dict: Optional[Dict[int, Dict[str, Any]]],
    ) -> int:
        if not tracker_dict:
            return 0

        canceled = 0
        for tracker in tracker_dict.values():
            if int(tracker.get("next_receive_step", 0)) > 0:
                tracker["next_receive_step"] = 0
                tracker["sync_start_time"] = None
                canceled += 1
        return canceled

    def _stage_tracker_sync(
        self,
        tracker_dict: Dict[int, Dict[str, Any]],
        tracker_idx: int,
        *,
        tracker_kind: str,
        expert_event_index: Optional[int] = None,
    ) -> None:
        tracker = tracker_dict[tracker_idx]
        tracker_label = tracker.get("display_name", str(tracker_idx))

        total_bytes = 0
        for param in tracker["param_refs"]:
            total_bytes += param.numel() * param.element_size()
        size_mb = total_bytes / (1024 * 1024)

        if self.verbose:
            print_rank_0(
                f"[CDC] Step {self.step_count}: Blocking full sync staging {tracker_kind} "
                f"{tracker_label} (Size: {size_mb:.2f} MB)."
            )

        tracker["sync_start_time"] = time.time()
        for p_local, p_staged in zip(tracker["param_refs"], tracker["staged_params"]):
            self._copy_tensor_data(p_staged, p_local.data)

        tracker["old_sent_at_step"] = tracker["sent_at_step"]
        tracker["sent_at_step"] = self.step_count
        tracker["sent_wd_log_cumsums"] = self._snapshot_sent_wd_log_cumsums(tracker)
        if tracker_kind == 'expert-group':
            if expert_event_index is not None:
                tracker["sent_at_expert_event"] = int(expert_event_index)
            tracker["token_load_accum"] = 0.0
        tracker["next_receive_step"] = 0

    def _select_next_shard(self):
        """Select the next shard to sync based on staleness and gradient norm."""
        # Streaming: Simple Round-Robin
        if self.algorithm == 'streaming':
            idx = self._next_nonempty_tracker_index(self.shard_tracker, self.next_shard_idx)
            if idx is None:
                return 0
            self.next_shard_idx = (idx + 1) % self.num_shards
            return idx

        # DC: Smart Selection
        # Max staleness: dc_N * sync_interval (allow dc_N skips)
        H = self.dc_N * self.sync_interval
        K = self.num_shards

        # 1. Check for stale shards
        for shard_idx in range(K):
            t_p_b = self.shard_tracker[shard_idx]["sent_at_step"]
            I_p = self.step_count - t_p_b
            if I_p >= H:
                return shard_idx

        # 2. Select based on score R (calculated from previous sync)
        scores = {}

        for shard_idx in range(K):
            tracker = self.shard_tracker[shard_idx]
            if tracker["sent_at_step"] == 0:
                return shard_idx # Prioritize never sent

            # Use cached score from last sync
            # R = ||grad||^2 * 1e8 / (I_p * N_params)
            # Note: tracker['last_score'] stores ||grad||^2 (aggregated across TP)
            update_magnitude_sq = tracker["last_score"]

            I_p = max(self.step_count - tracker["sent_at_step"], 1)
            
            last_sync_interval = max(tracker["sent_at_step"] - tracker["old_sent_at_step"], 1)

            denom = tracker["global_num_params"] if tracker["global_num_params"] > 0 else 1
            # current_R = update_magnitude_sq * 1e8 / (I_p * denom)
            # 分数逻辑：norm * I_p / last_sync_interval / N_params，1e8放大数值防止下溢
            # current_R = 1e8 * (update_magnitude_sq * I_p) / ( last_sync_interval * denom )
            # 改用RMS
            current_R = 1e8 * (math.sqrt(update_magnitude_sq/denom)) * (I_p / last_sync_interval)
            scores[shard_idx] = current_R

        # No global agreement needed (deterministic if all ranks have same history)
        # We assume all ranks have same last_score because they all-reduced the gradient.

        # Find max score
        selected_idx = max(scores, key=scores.get)
        return selected_idx

    def _expert_tracker_is_available(self, tracker: Dict[str, Any]) -> bool:
        if not self._tracker_has_any_params(tracker):
            return False
        if tracker.get("next_receive_step", 0) > self.step_count:
            return False
        return True

    @staticmethod
    def _expert_tracker_is_unsent(tracker: Dict[str, Any]) -> bool:
        return int(tracker.get("sent_at_expert_event", 0)) <= 0

    def _expert_tracker_age_slots(
        self, tracker: Dict[str, Any], upcoming_event: int
    ) -> int:
        sent_event = int(tracker.get("sent_at_expert_event", 0))
        if sent_event <= 0:
            return 0
        return int(upcoming_event - sent_event)

    def _expert_tracker_age_steps(self, tracker: Dict[str, Any]) -> int:
        sent_step = int(tracker.get("sent_at_step", 0))
        if sent_step <= 0:
            return 0
        return int(self.step_count - sent_step)

    def _expert_tracker_is_slot_stale(
        self, tracker: Dict[str, Any], upcoming_event: int
    ) -> bool:
        if self.expert_max_age_slots <= 0:
            return False
        if self._expert_tracker_is_unsent(tracker):
            return False
        return self._expert_tracker_age_slots(tracker, upcoming_event) >= self.expert_max_age_slots

    def _expert_tracker_is_step_stale(self, tracker: Dict[str, Any]) -> bool:
        if self.expert_max_staleness <= 0:
            return False
        if self._expert_tracker_is_unsent(tracker):
            return False
        return self._expert_tracker_age_steps(tracker) >= self.expert_max_staleness

    def _expert_tracker_is_min_age_blocked(
        self, tracker: Dict[str, Any], upcoming_event: int
    ) -> bool:
        if self.expert_min_age_slots <= 0:
            return False
        if self._expert_tracker_is_unsent(tracker):
            return False
        return self._expert_tracker_age_slots(tracker, upcoming_event) < self.expert_min_age_slots

    def _collect_round_robin_expert_indices(
        self,
        *,
        limit: int,
        selected: Optional[set] = None,
        require_unsent: bool = False,
        eligible_indices: Optional[set] = None,
    ) -> List[int]:
        if not self.expert_shard_tracker or limit <= 0:
            return []

        tracker_count = len(self.expert_shard_tracker)
        selected = selected or set()
        chosen: List[int] = []

        for offset in range(tracker_count):
            idx = (self.next_expert_group_idx + offset) % tracker_count
            if idx in selected or idx in chosen:
                continue
            if eligible_indices is not None and idx not in eligible_indices:
                continue
            tracker = self.expert_shard_tracker[idx]
            if not self._expert_tracker_is_available(tracker):
                continue
            if require_unsent and tracker.get("sent_at_expert_event", 0) > 0:
                continue
            chosen.append(idx)
            if len(chosen) >= limit:
                break

        if chosen:
            self.next_expert_group_idx = (chosen[-1] + 1) % tracker_count
        return chosen

    def _build_expert_score_map(self, candidate_indices: List[int]) -> Dict[int, float]:
        if not candidate_indices:
            return {}

        update_scores: Dict[int, float] = {}
        token_scores_local: Dict[int, float] = {}
        for idx in candidate_indices:
            tracker = self.expert_shard_tracker[idx]
            denom = tracker["global_num_params"] if tracker["global_num_params"] > 0 else 1
            update_scores[idx] = math.sqrt(max(float(tracker["last_score"]), 0.0) / denom)
            token_scores_local[idx] = max(float(tracker.get("token_load_accum", 0.0)), 0.0)

        if self.expert_score_mode == 'update_norm':
            return update_scores

        ordered_indices = sorted(candidate_indices)
        reduced_token_values = self._all_reduce_vector_sum(
            [token_scores_local[idx] for idx in ordered_indices], group=self.cdc_group
        )
        token_scores = {
            idx: value for idx, value in zip(ordered_indices, reduced_token_values)
        }

        if self.expert_score_mode == 'token_load':
            return token_scores

        update_max = max(update_scores.values(), default=0.0)
        token_max = max(token_scores.values(), default=0.0)
        combined_scores: Dict[int, float] = {}
        for idx in candidate_indices:
            update_component = update_scores[idx] / update_max if update_max > 0.0 else 0.0
            token_component = token_scores[idx] / token_max if token_max > 0.0 else 0.0
            combined_scores[idx] = 0.5 * update_component + 0.5 * token_component
        return combined_scores

    def _collect_expert_selection_debug_state(
        self,
        *,
        all_indices: List[int],
        candidate_indices: List[int],
        upcoming_event: int,
        selection_score_all: Dict[int, float],
        selection_score_used: Dict[int, float],
    ) -> Dict[int, Dict[str, Any]]:
        debug_rows: Dict[int, Dict[str, Any]] = {}

        global_token_accum: Dict[int, float] = {}
        if all_indices:
            ordered_indices = sorted(all_indices)
            local_token_values = [
                max(
                    float(self.expert_shard_tracker[idx].get("token_load_accum", 0.0)),
                    0.0,
                )
                for idx in ordered_indices
            ]
            reduced_token_values = self._all_reduce_vector_sum(
                local_token_values, group=self.cdc_group
            )
            global_token_accum = {
                idx: float(value)
                for idx, value in zip(ordered_indices, reduced_token_values)
            }

        candidate_set = set(candidate_indices)
        for idx in sorted(all_indices):
            tracker = self.expert_shard_tracker[idx]
            sent_event = int(tracker.get("sent_at_expert_event", 0))
            sent_step = int(tracker.get("sent_at_step", 0))
            next_receive_step = int(tracker.get("next_receive_step", 0))
            age_slots = self._expert_tracker_age_slots(tracker, upcoming_event)
            age_steps = self._expert_tracker_age_steps(tracker)
            slot_stale = self._expert_tracker_is_slot_stale(tracker, upcoming_event)
            step_stale = self._expert_tracker_is_step_stale(tracker)
            min_age_blocked = self._expert_tracker_is_min_age_blocked(tracker, upcoming_event)
            denom = tracker["global_num_params"] if tracker["global_num_params"] > 0 else 1
            update_norm = math.sqrt(max(float(tracker["last_score"]), 0.0) / denom)

            debug_rows[idx] = {
                "display_name": tracker.get("display_name", str(idx)),
                "available": self._expert_tracker_is_available(tracker),
                "candidate": idx in candidate_set,
                "in_flight": next_receive_step > self.step_count,
                "unsent": self._expert_tracker_is_unsent(tracker),
                "sent_at_step": sent_step,
                "sent_at_expert_event": sent_event,
                "next_receive_step": next_receive_step,
                "age_slots": age_slots,
                "age_steps": age_steps,
                "slot_stale": slot_stale,
                "step_stale": step_stale,
                "min_age_blocked": min_age_blocked,
                "last_score_norm_sq": float(tracker["last_score"]),
                "update_norm": update_norm,
                "last_token_load": float(tracker.get("last_token_load", 0.0)),
                "token_load_accum_local": float(tracker.get("token_load_accum", 0.0)),
                "token_load_accum_global": float(global_token_accum.get(idx, 0.0)),
                "selection_score_all": selection_score_all.get(idx),
                "selection_score_used": selection_score_used.get(idx),
            }

        return debug_rows

    def _log_expert_selection_debug(
        self,
        *,
        all_indices: List[int],
        candidate_indices: List[int],
        upcoming_event: int,
        selected: List[int],
        selected_reasons: Dict[int, str],
        selection_score_all: Dict[int, float],
        selection_score_used: Dict[int, float],
    ) -> None:
        if not self.verbose or not self.expert_shard_tracker:
            return

        debug_rows = self._collect_expert_selection_debug_state(
            all_indices=all_indices,
            candidate_indices=candidate_indices,
            upcoming_event=upcoming_event,
            selection_score_all=selection_score_all,
            selection_score_used=selection_score_used,
        )

        selected_order = {idx: order for order, idx in enumerate(selected)}
        print_rank_0(
            f"[CDC][ExpertSelect] Step {self.step_count}: "
            f"event={upcoming_event}, mode={self.expert_selection}, "
            f"score_mode={self.expert_score_mode}, topk={self.expert_topk}, "
            f"available_count={len(candidate_indices)}, min_age_slots={self.expert_min_age_slots}, "
            f"selected={selected}"
        )

        for idx in sorted(all_indices):
            row = debug_rows[idx]
            selected_flag = idx in selected_order
            selected_rank = selected_order.get(idx, -1)
            select_reason = selected_reasons.get(idx, "-")
            selection_score_all = row["selection_score_all"]
            selection_score_used = row["selection_score_used"]
            score_all_str = (
                f"{selection_score_all:.6e}"
                if selection_score_all is not None
                else "n/a"
            )
            score_used_str = (
                f"{selection_score_used:.6e}"
                if selection_score_used is not None
                else "n/a"
            )
            print_rank_0(
                "[CDC][ExpertSelect] "
                f"idx={idx} "
                f"name={row['display_name']} "
                f"candidate={int(row['candidate'])} "
                f"available={int(row['available'])} "
                f"in_flight={int(row['in_flight'])} "
                f"unsent={int(row['unsent'])} "
                f"selected={int(selected_flag)} "
                f"selected_rank={selected_rank} "
                f"reason={select_reason} "
                f"sent_step={row['sent_at_step']} "
                f"sent_event={row['sent_at_expert_event']} "
                f"next_recv={row['next_receive_step']} "
                f"age_slots={row['age_slots']} "
                f"age_steps={row['age_steps']} "
                f"slot_stale={int(row['slot_stale'])} "
                f"step_stale={int(row['step_stale'])} "
                f"min_age_blocked={int(row['min_age_blocked'])} "
                f"last_score_norm2={row['last_score_norm_sq']:.6e} "
                f"update_norm={row['update_norm']:.6e} "
                f"last_token_load={row['last_token_load']:.6e} "
                f"token_accum_local={row['token_load_accum_local']:.6e} "
                f"token_accum_global={row['token_load_accum_global']:.6e} "
                f"score_all={score_all_str} "
                f"score_used={score_used_str}"
            )

    def _select_next_expert_groups(self) -> List[int]:
        if not self.expert_shard_tracker:
            return []

        all_indices = sorted(self.expert_shard_tracker.keys())
        candidate_indices = [
            idx
            for idx, tracker in self.expert_shard_tracker.items()
            if self._expert_tracker_is_available(tracker)
        ]
        if not candidate_indices:
            return []

        max_select = min(self.expert_topk, len(candidate_indices))
        selected: List[int] = []
        selected_set = set()
        selected_reasons: Dict[int, str] = {}
        upcoming_event = self.expert_sync_event_count + 1
        selection_score_all = self._build_expert_score_map(candidate_indices)
        selection_score_used: Dict[int, float] = {}

        # Phase 1: mandatory first-send pass. Any never-sent expert has the highest priority.
        unsent = self._collect_round_robin_expert_indices(
            limit=max_select,
            selected=selected_set,
            require_unsent=True,
        )
        for idx in unsent:
            selected.append(idx)
            selected_set.add(idx)
            selected_reasons[idx] = 'mandatory_unsent'

        remaining = max_select - len(selected)
        stale_candidates: List[Tuple[int, int, int, bool, bool]] = []
        if remaining > 0:
            for idx in candidate_indices:
                if idx in selected_set:
                    continue
                tracker = self.expert_shard_tracker[idx]
                age_slots = self._expert_tracker_age_slots(tracker, upcoming_event)
                age_steps = self._expert_tracker_age_steps(tracker)
                slot_stale = self._expert_tracker_is_slot_stale(tracker, upcoming_event)
                step_stale = self._expert_tracker_is_step_stale(tracker)
                if slot_stale or step_stale:
                    stale_candidates.append((age_slots, age_steps, idx, slot_stale, step_stale))

            if stale_candidates:
                stale_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
                for _, _, idx, slot_stale, step_stale in stale_candidates[:remaining]:
                    selected.append(idx)
                    selected_set.add(idx)
                    if slot_stale and step_stale:
                        selected_reasons[idx] = 'stale_slot+step'
                    elif slot_stale:
                        selected_reasons[idx] = 'stale_slot'
                    else:
                        selected_reasons[idx] = 'stale_step'

        remaining = max_select - len(selected)
        if remaining <= 0:
            self._log_expert_selection_debug(
                all_indices=all_indices,
                candidate_indices=candidate_indices,
                upcoming_event=upcoming_event,
                selected=selected,
                selected_reasons=selected_reasons,
                selection_score_all=selection_score_all,
                selection_score_used=selection_score_used,
            )
            return selected

        regular_candidate_set = {
            idx
            for idx in candidate_indices
            if idx not in selected_set
            and not self._expert_tracker_is_min_age_blocked(
                self.expert_shard_tracker[idx], upcoming_event
            )
        }

        if self.expert_selection == 'round_robin':
            rr_indices = self._collect_round_robin_expert_indices(
                limit=remaining,
                selected=selected_set,
                require_unsent=False,
                eligible_indices=regular_candidate_set,
            )
            for idx in rr_indices:
                selected_reasons[idx] = 'round_robin'
            selected.extend(rr_indices)
            self._log_expert_selection_debug(
                all_indices=all_indices,
                candidate_indices=candidate_indices,
                upcoming_event=upcoming_event,
                selected=selected,
                selected_reasons=selected_reasons,
                selection_score_all=selection_score_all,
                selection_score_used=selection_score_used,
            )
            return selected

        score_candidates = sorted(regular_candidate_set)
        selection_score_used = self._build_expert_score_map(score_candidates)
        ranked = sorted(
            score_candidates,
            key=lambda idx: (-selection_score_used.get(idx, 0.0), idx),
        )
        chosen_from_score = ranked[:remaining]
        for rank, idx in enumerate(chosen_from_score):
            selected_reasons[idx] = f"score_rank_{rank}"
        selected.extend(chosen_from_score)
        self._log_expert_selection_debug(
            all_indices=all_indices,
            candidate_indices=candidate_indices,
            upcoming_event=upcoming_event,
            selected=selected,
            selected_reasons=selected_reasons,
            selection_score_all=selection_score_all,
            selection_score_used=selection_score_used,
        )
        return selected

    def _initiate_tracker_sync(
        self,
        tracker_dict: Dict[int, Dict[str, Any]],
        tracker_idx: int,
        *,
        tracker_kind: str,
        expert_event_index: Optional[int] = None,
        defer_completion: bool = False,
    ) -> None:
        """Start the sync process for a tracker entry (snapshot & send)."""
        tracker = tracker_dict[tracker_idx]
        tracker_label = tracker.get("display_name", str(tracker_idx))

        # Calculate payload size for logging.
        total_bytes = 0
        for p in tracker["param_refs"]:
            total_bytes += p.numel() * p.element_size()
        size_mb = total_bytes / (1024 * 1024)

        if self.verbose:
            print_rank_0(
                f"[CDC] Step {self.step_count}: Initiating sync for {tracker_kind} "
                f"{tracker_label} (Size: {size_mb:.2f} MB)."
            )

        tracker["sync_start_time"] = time.time()

        # Snapshot current local params to staged_params
        for p_local, p_staged in zip(tracker["param_refs"], tracker["staged_params"]):
            self._copy_tensor_data(p_staged, p_local.data)

        tracker["old_sent_at_step"] = tracker["sent_at_step"]
        tracker["sent_at_step"] = self.step_count
        tracker["sent_wd_log_cumsums"] = self._snapshot_sent_wd_log_cumsums(tracker)
        if tracker_kind == 'expert-group':
            if expert_event_index is not None:
                tracker["sent_at_expert_event"] = int(expert_event_index)
            tracker["token_load_accum"] = 0.0

        # Schedule receive
        tracker["next_receive_step"] = self.step_count + self.delay

        # If delay is 0, complete immediately (synchronous)
        if self.delay == 0 and not defer_completion:
            self._complete_tracker_sync(
                tracker_dict, tracker_idx, tracker_kind=tracker_kind
            )
            tracker["next_receive_step"] = 0

    def _complete_tracker_sync(
        self,
        tracker_dict: Dict[int, Dict[str, Any]],
        tracker_idx: int,
        *,
        tracker_kind: str,
        reload_main_params: bool = True,
        use_algorithm_specific_update: bool = True,
        force_full_copy: bool = False,
    ) -> None:
        """Complete the sync process for a tracker entry (receive & update)."""
        tracker = tracker_dict[tracker_idx]
        tracker_label = tracker.get("display_name", str(tracker_idx))
        param_refs = tracker["param_refs"]
        global_params = tracker["params"]
        staged_params = tracker["staged_params"]

        # 1. Calculate sync gradients (Global - Staged)
        sync_grads = []
        for p_global, p_staged in zip(global_params, staged_params):
            # Global and Staged 一定在同一个设备上
            g = p_global.data.clone()
            g.sub_(p_staged.data)
            sync_grads.append(g)

        # 2. All-Reduce sync_grads (Across DiLoCo Islands)
        self._all_reduce_flattened(
            sync_grads,
            communication_dtype=tracker.get("comm_dtype", self.outer_comm_dtype),
        )

        # 3. Calculate Score for Next Selection (Norm of Global Pseudo-Gradient)
        total_norm_sq = 0.0
        for g in sync_grads:
            total_norm_sq += float(g.float().pow(2).sum().item())

        # All-Reduce norm across TP group, then across PP group.
        tp_group = mpu.get_tensor_model_parallel_group()
        if tp_group is not None and dist.get_world_size(group=tp_group) > 1:
            norm_tensor = torch.tensor(float(total_norm_sq), device=torch.device('cuda'))
            dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM, group=tp_group)
            total_norm_sq = norm_tensor.item()

        pp_group = mpu.get_pipeline_model_parallel_group()
        if pp_group is not None and dist.get_world_size(group=pp_group) > 1:
            # Use GPU tensor if available (sync_grads may be empty on some partitions; handle that).
            norm_tensor = torch.tensor(float(total_norm_sq), device=torch.device('cuda'))
            dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM, group=pp_group)
            total_norm_sq = norm_tensor.item()

        tracker["last_score"] = total_norm_sq

        if self.verbose:
            duration = time.time() - tracker.get("sync_start_time", time.time())
            print_rank_0(
                f"[CDC] Step {self.step_count}: Completed sync for {tracker_kind} "
                f"{tracker_label} in {duration:.4f}s. Score (Norm^2): {total_norm_sq:.4e}"
            )

        # 4. Outer Optimizer Step (Update Global)
        if tracker["outer_optimizer"]:
            if len(global_params) > 0:
                for p_global, avg_delta in zip(global_params, sync_grads):
                    if p_global.grad is None:
                        p_global.grad = torch.zeros_like(p_global.data)
                    p_global.grad.copy_(avg_delta)
                tracker["outer_optimizer"].step()
                tracker["outer_optimizer"].zero_grad(set_to_none=True)
        else:
            # Simple averaging
            for p_global, avg_delta in zip(global_params, sync_grads):
                p_global.data.sub_(avg_delta)

        # 5. Update Local Params (Algorithm Specific)
        if use_algorithm_specific_update and self.algorithm == 'dc' and tracker_kind == 'dense-shard':
            self.delay_compensation(tracker)

        else:
            # Alpha blending for normal streaming receives; alpha=0 hard-overwrites local.
            self._apply_global_params_to_local(tracker, force_full_copy=force_full_copy)

        # Keep optimizer main params in sync with model params for mixed precision.
        if self.mixed_precision and reload_main_params:
            self.inner_optimizer.reload_model_params()

    def _complete_tracker_sync_batch(
        self,
        tracker_dict: Dict[int, Dict[str, Any]],
        tracker_indices: List[int],
        *,
        tracker_kind: str,
    ) -> None:
        if not tracker_indices:
            return

        tracker_entries: List[Tuple[Dict[str, Any], List[torch.nn.Parameter], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]] = []
        batched_sync_grads: List[torch.Tensor] = []

        for tracker_idx in tracker_indices:
            tracker = tracker_dict[tracker_idx]
            param_refs = tracker["param_refs"]
            global_params = tracker["params"]
            staged_params = tracker["staged_params"]

            sync_grads: List[torch.Tensor] = []
            for p_global, p_staged in zip(global_params, staged_params):
                grad_tensor = p_global.data.clone()
                grad_tensor.sub_(p_staged.data)
                sync_grads.append(grad_tensor)
            tracker_entries.append((tracker, param_refs, global_params, staged_params, sync_grads))
            batched_sync_grads.extend(sync_grads)

        batch_comm_dtype = tracker_entries[0][0].get("comm_dtype", self.outer_comm_dtype)
        self._all_reduce_flattened(
            batched_sync_grads, communication_dtype=batch_comm_dtype
        )

        tp_group = mpu.get_tensor_model_parallel_group()
        pp_group = mpu.get_pipeline_model_parallel_group()

        for tracker, param_refs, global_params, _, sync_grads in tracker_entries:
            tracker_label = tracker.get("display_name", "unknown")

            total_norm_sq = 0.0
            for grad_tensor in sync_grads:
                total_norm_sq += float(grad_tensor.float().pow(2).sum().item())

            if tp_group is not None and dist.get_world_size(group=tp_group) > 1:
                norm_tensor = torch.tensor(float(total_norm_sq), device=torch.device('cuda'))
                dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM, group=tp_group)
                total_norm_sq = norm_tensor.item()

            if pp_group is not None and dist.get_world_size(group=pp_group) > 1:
                norm_tensor = torch.tensor(float(total_norm_sq), device=torch.device('cuda'))
                dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM, group=pp_group)
                total_norm_sq = norm_tensor.item()

            tracker["last_score"] = total_norm_sq

            if self.verbose:
                duration = time.time() - tracker.get("sync_start_time", time.time())
                print_rank_0(
                    f"[CDC] Step {self.step_count}: Completed sync for {tracker_kind} "
                    f"{tracker_label} in {duration:.4f}s. Score (Norm^2): {total_norm_sq:.4e}"
                )

            if tracker["outer_optimizer"]:
                if len(global_params) > 0:
                    for p_global, avg_delta in zip(global_params, sync_grads):
                        if p_global.grad is None:
                            p_global.grad = torch.zeros_like(p_global.data)
                        p_global.grad.copy_(avg_delta)
                    tracker["outer_optimizer"].step()
                    tracker["outer_optimizer"].zero_grad(set_to_none=True)
            else:
                for p_global, avg_delta in zip(global_params, sync_grads):
                    p_global.data.sub_(avg_delta)

            self._apply_global_params_to_local(tracker)

        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()
            
    def delay_compensation(self, tracker):
        """ Update Local Params (Algorithm Specific)"""
        staged_params = tracker["staged_params"]
        param_refs = tracker["param_refs"]
        global_params = tracker["params"]

        # 老版
        if self.dc_type == "legacy":
            g_1 = []
            D = []

            for p_staged, p_local, p_global in zip(staged_params, param_refs, global_params):
                if self.offload_outer_opt:
                    p_local_data = p_local.detach().to("cpu", dtype=torch.float32)
                    p_global_data = p_global.data.to(torch.float32)
                else:
                    p_local_data = p_local.data.to(torch.float32)
                    p_global_data = p_global.data.to(torch.float32)

                # g_1 = Staged - Local
                g1_tensor = p_staged.data.to(torch.float32).sub_(p_local_data)
                g_1.append(g1_tensor)

                # D = Global - Staged
                d_tensor = p_global_data.sub(p_staged.data.to(torch.float32))
                D.append(d_tensor)

            epsilon = 1e-8

            g_1_corrected = []
            for g1, d in zip(g_1, D):
                numerator = self.dc_lambda * torch.norm(g1)
                correction_term = (g1 * g1 * d) / 4e-4
                denominator = torch.norm(correction_term)
                dynamic_lambda = numerator / (denominator + epsilon)

                corrected = g1 + (dynamic_lambda * correction_term)
                g_1_corrected.append(corrected)

            for p_local, p_global, g_corr in zip(param_refs, global_params, g_1_corrected):
                if self.offload_outer_opt:
                    target = p_global.data - g_corr
                    p_local.data.copy_(target.to(p_local.device))
                else:
                    p_local.data.copy_(p_global.data - g_corr)
            return
        
        # 新版本
        # ---- Update-space delay compensation ----
        # Effective staleness in inner steps (may be > self.delay if scheduling shifts)
        tau = int(self.step_count - tracker["sent_at_step"])
        tau = max(tau, 1)
        param_group_indices = tracker["param_group_indices"]

        eps = 1e-8  # 数值稳定性
        args = get_args()
        lam0 = self.dc_lambda  # lambda base
        lam_max = float(getattr(args, "cdc_dc_lambda_max", 10.0))  # 上限
        scope = str(getattr(args, "cdc_dc_lambda_scope", "shard")).lower()  # "shard", "tensor", "local"
        debias_wd = bool(getattr(args, "cdc_dc_debias_wd", False))

        # Per-param-group rho for decoupled weight decay debias (optional).
        rho_by_group: Dict[int, float] = {}
        if debias_wd:
            for group_idx in tracker.get("unique_param_group_indices", []):
                rho_by_group[int(group_idx)] = self._rho_wd_between_send_and_now(
                    tracker, int(group_idx)
                )

        tp_group = mpu.get_tensor_model_parallel_group()
        pp_group = mpu.get_pipeline_model_parallel_group()

        def _reduce_sum(x: float, do_tp: bool, do_pp: bool) -> float:
            y = float(x)
            if do_tp:
                y = self._all_reduce_scalar_sum(y, group=tp_group)
            if do_pp:
                y = self._all_reduce_scalar_sum(y, group=pp_group)
            return float(y)

        # Decide reduction policy
        if scope == "shard":
            do_tp, do_pp = True, True
        elif scope == "tensor":
            do_tp, do_pp = True, False
        elif scope == "local":
            do_tp, do_pp = False, False
        else:
            raise ValueError(f"Unknown cdc_dc_lambda_scope: {scope}")

        # Helper: compute sum of squares in fp32 (device-agnostic)
        def _sqsum_fp32(t: torch.Tensor) -> float:
            return float(t.float().pow(2).sum().item())

        def _dc_terms(
            p_staged: torch.Tensor,
            p_local: torch.nn.Parameter,
            p_global: torch.Tensor,
            rho: float,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute (u, c, z_wd) in fp32 for update-space delay compensation."""
            if self.offload_outer_opt:
                theta1 = p_local.detach().to("cpu", dtype=torch.float32)
                z = p_global.data.to(torch.float32)
            else:
                theta1 = p_local.data.to(torch.float32)
                z = p_global.data.to(torch.float32)

            theta0 = p_staged.data.to(torch.float32)
            z_wd = z.mul(rho) if debias_wd else z
            D = z_wd - theta0

            u_total = (theta0 - theta1).div(tau)
            if debias_wd and rho != 1.0:
                u_wd = theta0.mul((1.0 - rho) / tau)
                u = u_total - u_wd
            else:
                u = u_total

            c = (u * u) * D
            return u, c, z_wd

        # -------------------------
        # Case A: per-shard lambda
        # -------------------------
        if scope in ("shard", "local"):
            # Pass 1: accumulate local shard norms (no extra tensor storage)
            local_u2 = 0.0
            local_c2 = 0.0

            for p_staged, p_local, p_global, group_idx in zip(
                staged_params, param_refs, global_params, param_group_indices
            ):
                rho = rho_by_group.get(int(group_idx), 1.0) if group_idx is not None else 1.0
                u, c, _ = _dc_terms(p_staged, p_local, p_global, rho)
                local_u2 += _sqsum_fp32(u)
                local_c2 += _sqsum_fp32(c)

            # Reduce norms if requested
            u2 = _reduce_sum(local_u2, do_tp=do_tp, do_pp=do_pp)
            c2 = _reduce_sum(local_c2, do_tp=do_tp, do_pp=do_pp)

            norm_u = math.sqrt(max(u2, 0.0))
            norm_c = math.sqrt(max(c2, 0.0))

            lam = lam0 * norm_u / (norm_c + eps)
            lam = float(min(lam, lam_max))

            # Pass 2: apply compensated update
            for p_staged, p_local, p_global, group_idx in zip(
                staged_params, param_refs, global_params, param_group_indices
            ):
                rho = rho_by_group.get(int(group_idx), 1.0) if group_idx is not None else 1.0
                u, c, z_wd = _dc_terms(p_staged, p_local, p_global, rho)
                u_hat = u + lam * c

                target = z_wd.to(torch.float32) - (tau * u_hat)

                p_local.data.copy_(target.to(dtype=p_local.dtype, device=p_local.device))

        # -------------------------
        # Case B: per-tensor lambda
        # -------------------------
        else:  # scope == "tensor"
            for p_staged, p_local, p_global, group_idx in zip(
                staged_params, param_refs, global_params, param_group_indices
            ):
                rho = rho_by_group.get(int(group_idx), 1.0) if group_idx is not None else 1.0
                u, c, z_wd = _dc_terms(p_staged, p_local, p_global, rho)

                # compute per-tensor norms and reduce across TP only
                local_u2 = _sqsum_fp32(u)
                local_c2 = _sqsum_fp32(c)

                u2 = _reduce_sum(local_u2, do_tp=True, do_pp=False)
                c2 = _reduce_sum(local_c2, do_tp=True, do_pp=False)

                norm_u = math.sqrt(max(u2, 0.0))
                norm_c = math.sqrt(max(c2, 0.0))

                lam = lam0 * norm_u / (norm_c + eps)
                lam = float(min(lam, lam_max))

                u_hat = u + lam * c
                target = z_wd.to(torch.float32) - (tau * u_hat)

                p_local.data.copy_(target.to(dtype=p_local.dtype, device=p_local.device))
    
    @torch.no_grad()
    def _all_reduce_flattened(self, tensors, communication_dtype=None):
        """Helper to flatten, all-reduce, and unflatten tensors."""
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        # Empty tensor list is a valid no-op (e.g., shards that are empty on this PP stage).
        if not tensors:
            print_rank_0("[CDC] Warning: _all_reduce_flattened called with empty tensor list. No operation performed.")
            return

        start_time = time.time()
        total_bytes = 0

        # Group by dtype
        groups = {}
        for t in tensors:
            dtype = t.dtype
            if dtype not in groups:
                groups[dtype] = []
            groups[dtype].append(t)

        for dtype, group_tensors in groups.items():
            # Flatten
            flat_tensor = _flatten_dense_tensors(group_tensors)
            original_dtype = flat_tensor.dtype
            comm_dtype = communication_dtype if communication_dtype is not None else original_dtype

            device = flat_tensor.device
            if device.type == 'cpu':
                comm_tensor = flat_tensor.to(device='cuda', dtype=comm_dtype)
                total_bytes += comm_tensor.numel() * comm_tensor.element_size()
                dist.all_reduce(comm_tensor, group=self.cdc_group)
                comm_tensor.div_(dist.get_world_size(group=self.cdc_group))
                flat_tensor.copy_(comm_tensor.to(device=device, dtype=original_dtype))
                del comm_tensor
            else:
                if comm_dtype != original_dtype:
                    comm_tensor = flat_tensor.to(dtype=comm_dtype)
                else:
                    comm_tensor = flat_tensor

                total_bytes += comm_tensor.numel() * comm_tensor.element_size()
                dist.all_reduce(comm_tensor, group=self.cdc_group)
                comm_tensor.div_(dist.get_world_size(group=self.cdc_group))
                if comm_tensor is not flat_tensor:
                    flat_tensor.copy_(comm_tensor.to(dtype=original_dtype))

            # Unflatten and copy back to grads
            for t, synced_t in zip(
                group_tensors, _unflatten_dense_tensors(flat_tensor, group_tensors)
            ):
                t.copy_(synced_t)

        end_time = time.time()
        duration = end_time - start_time

        if self.verbose:
            size_mb = total_bytes / (1024 * 1024)
            bandwidth = size_mb / duration if duration > 0 else 0
            print_rank_0(f"[CDC] Communication: {size_mb:.2f} MB in {duration:.4f}s ({bandwidth:.2f} MB/s)")
    
    @torch.no_grad()
    def prepare_grads(self):
        return self.inner_optimizer.prepare_grads()

    @torch.no_grad()
    def step_with_ready_grads(self):
        return self.inner_optimizer.step_with_ready_grads()

    def sharded_state_dict(self, model_sharded_state_dict, is_loading=False):
        """Include CDC metadata alongside sharded optimizer state."""
        try:
            inner_state = self.inner_optimizer.sharded_state_dict(
                model_sharded_state_dict, is_loading=is_loading
            )
        except TypeError:
            inner_state = self.inner_optimizer.sharded_state_dict(model_sharded_state_dict)

        return {
            "inner_optimizer": inner_state,
            "cdc_state": self._build_cdc_state(),
        }
