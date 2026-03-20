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
        self.sync_interval = getattr(args, 'cdc_sync_interval', getattr(args, 'diloco_sync_interval', 100))
        self.step_count = 0
        self.algorithm = getattr(args, 'cdc_algorithm', getattr(args, 'diloco_algorithm', 'diloco'))
        self.offload_outer_opt = getattr(args, 'cdc_offload_outer_opt', getattr(args, 'diloco_offload_outer_opt', False))
        self.outer_lr = getattr(args, 'cdc_outer_lr', getattr(args, 'diloco_outer_lr', 1.0))
        self.num_shards = getattr(args, 'cdc_num_shards', getattr(args, 'diloco_num_shards', 1))
        self.dc_lambda = getattr(args, 'cdc_dc_lambda', getattr(args, 'diloco_dc_lambda', 2.0))
        self.streaming_alpha = getattr(args, 'cdc_streaming_alpha', getattr(args, 'diloco_streaming_alpha', 0.5))
        self.delay = getattr(args, 'cdc_delay', getattr(args, 'diloco_delay', 0))
        self.dc_N = getattr(args, 'cdc_dc_N', getattr(args, 'diloco_dc_N', 4))
        self.shard_pattern = getattr(args, 'cdc_shard_pattern', 'stride')
        self.verbose = getattr(args, 'cdc_verbose', False)
        self.mixed_precision = getattr(args, 'bf16', False) or getattr(args, 'fp16', False)
        model_params = self.model_param_list
        self.model_param_dtype = model_params[0].dtype if model_params else torch.float32
        # Follow Streaming DiLoCo: keep outer-state math in fp32, but keep communication low
        # precision by default to avoid doubling bandwidth.
        self.outer_state_dtype = torch.float32 if self.mixed_precision else self.model_param_dtype
        self.outer_comm_dtype = (
            self.model_param_dtype if self.mixed_precision else self.outer_state_dtype
        )
        # DC specifics.
        self.dc_type = str(getattr(args, "cdc_dc_type", "update")).lower()
        # Initialized for all algorithms so checkpoint/state construction does not fail.
        self.next_shard_idx = 0
        self._cdc_state_loaded = False

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
                f"outer_state_dtype={self.outer_state_dtype}, outer_comm_dtype={self.outer_comm_dtype}"
            )
        
        if self.algorithm == 'diloco':
            self._init_diloco_state()
        elif self.algorithm in ['streaming', 'dc']:
            self._init_streaming_state()
        else:
            raise ValueError(f"Unknown DiLoCo algorithm: {self.algorithm}")

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
        model_param_list = []
        for chunk in self.model_chunks:
            for name, p in chunk.named_parameters():
                if p.requires_grad:
                    model_param_list.append(p)
        return model_param_list

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
            "lr_wd_tracking": self._serialize_lr_wd_tracking(),
        }

        if self.algorithm == 'diloco' and getattr(self, "original_snapshot", None) is not None:
            state["diloco"] = self._serialize_diloco_state()
        elif self.algorithm in ['streaming', 'dc'] and getattr(self, "shard_tracker", None) is not None:
            state["shards"] = self._serialize_shard_trackers()

        return state

    def _serialize_shard_trackers(self):
        shards = []
        if not self.shard_tracker:
            return shards

        for shard_idx in sorted(self.shard_tracker.keys()):
            tracker = self.shard_tracker[shard_idx]

            # Optimization: staged_params is only needed if a sync is in flight.
            # If next_receive_step is 0 (or <= step_count, meaning completed), it's redundant.
            save_staged = tracker["next_receive_step"] > self.step_count

            shard_entry = {
                "shard_idx": shard_idx,
                "params": [self._clone_tensor_to_cpu(p) for p in tracker["params"]],
                "staged_params": [self._clone_tensor_to_cpu(p) for p in tracker["staged_params"]] if save_staged else None,
                "sent_at_step": int(tracker["sent_at_step"]),
                "old_sent_at_step": int(tracker["old_sent_at_step"]),
                "next_receive_step": int(tracker["next_receive_step"]),
                "sent_wd_log_cumsums": deepcopy(tracker.get("sent_wd_log_cumsums")) if save_staged else None,
                "global_num_params": int(tracker["global_num_params"]),
                "last_score": float(tracker["last_score"]),
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
            self._load_shard_trackers(cdc_state.get("shards"))

    def _load_diloco_state(self, diloco_state):
        if not diloco_state:
            return

        snapshot = diloco_state.get("original_snapshot", None)

        # Optimization: If snapshot is None, it means it was identical to local params.
        if snapshot is None:
            for target, local in zip(self.original_snapshot, self.model_param_list):
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

    def _load_shard_trackers(self, shard_states):
        if not shard_states:
            return

        if self.shard_tracker is None:
            self._init_streaming_state()

        for shard_state in shard_states:
            shard_idx = shard_state["shard_idx"]
            if shard_idx not in self.shard_tracker:
                raise ValueError(f"Shard {shard_idx} not initialized but present in checkpoint.")

            tracker = self.shard_tracker[shard_idx]

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

    def _reset_outer_state_from_model(self) -> None:
        """Reinitialize CDC outer state from the current model parameters."""
        self.step_count = 0
        self.next_shard_idx = 0
        self._init_lr_wd_tracking()

        if self.algorithm == 'diloco' and getattr(self, "original_snapshot", None) is not None:
            for target, local in zip(self.original_snapshot, self.model_param_list):
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

        for tracker in self.shard_tracker.values():
            for p_global, p_local in zip(tracker["params"], tracker["param_refs"]):
                self._copy_tensor_data(p_global, p_local.data)
            for p_staged, p_local in zip(tracker["staged_params"], tracker["param_refs"]):
                self._copy_tensor_data(p_staged, p_local.data)

            tracker["sent_wd_log_cumsums"] = None
            tracker["sent_at_step"] = 0
            tracker["old_sent_at_step"] = 0
            tracker["next_receive_step"] = 0
            tracker["last_score"] = 0.0

            if tracker["outer_optimizer"] is not None:
                tracker["outer_optimizer"].state.clear()

            if tracker["params"]:
                self._all_reduce_flattened(
                    [t.data for t in tracker["params"]], communication_dtype=self.outer_comm_dtype
                )

    def _init_diloco_state(self):
        """Initialize state for standard DiLoCo."""
        self.original_snapshot = []  # 上一次同步时的模型参数快照（展平列表）

        for param in self.model_param_list:
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
        unique_named_params = self._iter_named_trainable_params_unique(self.model_chunks)
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
        self.next_shard_idx = 0

        tp_group = mpu.get_tensor_model_parallel_group()
        pp_group = mpu.get_pipeline_model_parallel_group()

        for shard_idx in range(self.num_shards):
            param_refs = shard_to_param_refs.get(shard_idx, [])

            param_group_indices = [
                self._get_param_group_index_for_model_param(p) for p in param_refs
            ]
            unique_param_group_indices = sorted(
                {i for i in param_group_indices if i is not None}
            )

            tracker = {
                "param_refs": param_refs,
                "param_group_indices": param_group_indices,
                "unique_param_group_indices": unique_param_group_indices,
                # Populated at send time to enable accurate WD debias even with lr-warmup-fraction.
                "sent_wd_log_cumsums": None,
                "params": [],
                "staged_params": [],
                "sent_at_step": 0,
                "old_sent_at_step": 0,
                "next_receive_step": 0,
                "global_num_params": 0,
                "last_score": 0.0,
            }

            # Clone for global / staged buffers.
            for p in param_refs:
                tracker["params"].append(self._clone_param_for_outer_state(p))
                tracker["staged_params"].append(self._clone_param_for_outer_state(p))

            # Outer optimizer on the per-shard global params (optional).
            if self.outer_lr != 1.0 and len(tracker["params"]) > 0:
                for p in tracker["params"]:
                    p.requires_grad_(True)
                tracker["outer_optimizer"] = SGD(
                    tracker["params"],
                    lr=self.outer_lr,
                    momentum=0.9,
                    nesterov=True,
                )
            else:
                tracker["outer_optimizer"] = None

            # Compute global shard param count across TP and PP for consistent selection.
            local_numel = sum(p.numel() for p in param_refs)
            global_numel = self._all_reduce_scalar_sum(local_numel, group=tp_group)
            global_numel = self._all_reduce_scalar_sum(global_numel, group=pp_group)
            tracker["global_num_params"] = int(global_numel)

            # Initialize global params to the cross-DC average (within this PP/TP partition).
            # This is crucial: if global == local on every island, the first sync would have zero delta.
            if len(tracker["params"]) > 0:
                # Average the cloned tensors across CDC group in-place.
                self._all_reduce_flattened(
                    [t.data for t in tracker["params"]], communication_dtype=self.outer_comm_dtype
                )

            self.shard_tracker[shard_idx] = tracker

            if self.verbose:
                print_rank_0(
                    f"[CDC] Shard {shard_idx} initialized: local_tensors={len(param_refs)}, "
                    f"global_numel={tracker['global_num_params']}"
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
        for snap_param, model_param in zip(self.original_snapshot, self.model_param_list):
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
        for updated_param, curr_param in zip(self.original_snapshot, self.model_param_list):
            curr_param.copy_(updated_param.to(device=curr_param.device, dtype=curr_param.dtype))

        # Keep optimizer main params in sync with model params for mixed precision.
        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()
        
    def _sync_streaming(self):
        """Unified synchronization step for Streaming and DC."""
        # Check for pending receives
        for shard_idx, tracker in self.shard_tracker.items():
            if tracker["next_receive_step"] > 0 and self.step_count >= tracker["next_receive_step"]:
                self._complete_sync(shard_idx)
                tracker["next_receive_step"] = 0 # Reset

        # Check for new sends
        # Unified logic: Every sync_interval steps, initiate a sync.
        # Selection logic (Round-Robin vs Smart) is handled in _select_next_shard.

        if self.step_count % self.sync_interval == 0:
            shard_idx = self._select_next_shard()
            self._initiate_sync(shard_idx)

    def _select_next_shard(self):
        """Select the next shard to sync based on staleness and gradient norm."""
        # Streaming: Simple Round-Robin
        if self.algorithm == 'streaming':
            idx = self.next_shard_idx
            self.next_shard_idx = (self.next_shard_idx + 1) % self.num_shards
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

    def _initiate_sync(self, shard_idx):
        """Start the sync process for a shard (Snapshot & Send)."""
        tracker = self.shard_tracker[shard_idx]

        # Calculate shard size for logging
        total_bytes = 0
        for p in tracker["param_refs"]:
            total_bytes += p.numel() * p.element_size()
        size_mb = total_bytes / (1024 * 1024)

        if self.verbose:
            print_rank_0(f"[CDC] Step {self.step_count}: Initiating sync for shard {shard_idx} (Size: {size_mb:.2f} MB).")

        tracker["sync_start_time"] = time.time()

        # Snapshot current local params to staged_params
        for p_local, p_staged in zip(tracker["param_refs"], tracker["staged_params"]):
            self._copy_tensor_data(p_staged, p_local.data)

        tracker["old_sent_at_step"] = tracker["sent_at_step"]
        tracker["sent_at_step"] = self.step_count
        tracker["sent_wd_log_cumsums"] = self._snapshot_sent_wd_log_cumsums(tracker)

        # Schedule receive
        tracker["next_receive_step"] = self.step_count + self.delay

        # If delay is 0, complete immediately (synchronous)
        if self.delay == 0:
            self._complete_sync(shard_idx)
            tracker["next_receive_step"] = 0

    def _complete_sync(self, shard_idx):
        """Complete the sync process (Receive & Update)."""
        tracker = self.shard_tracker[shard_idx]
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
        self._all_reduce_flattened(sync_grads, communication_dtype=self.outer_comm_dtype)

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
            print_rank_0(f"[CDC] Step {self.step_count}: Completed sync for shard {shard_idx} in {duration:.4f}s. Score (Norm^2): {total_norm_sq:.4e}")

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
        if self.algorithm == 'dc':
            self.delay_compensation(tracker)

        elif self.algorithm == 'streaming':
            # Alpha Blending
            for p_local, p_global in zip(param_refs, global_params):
                p_global_data = p_global.data.to(device=p_local.device, dtype=torch.float32)
                blended = (
                    p_local.data.to(torch.float32).mul(self.streaming_alpha).add_(
                        p_global_data, alpha=1.0 - self.streaming_alpha
                    )
                )
                p_local.data.copy_(blended.to(dtype=p_local.dtype))

        # Keep optimizer main params in sync with model params for mixed precision.
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
