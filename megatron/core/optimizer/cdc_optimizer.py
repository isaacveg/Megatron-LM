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
        assert int(args.cdc_parallel_size) > 1, (
            "CDCOptimizer requires cdc_parallel_size > 1. "
            "Use the base optimizer when CDC parallelism is disabled."
        )
        cdc_world_size = mpu.get_cdc_parallel_world_size()
        assert cdc_world_size > 1, (
            "CDCOptimizer requires an initialized CDC parallel group with world size > 1."
        )
        self.cdc_group = mpu.get_cdc_parallel_group()
        assert self.cdc_group is not None, "CDCOptimizer requires a real CDC process group."
        self.sync_interval = args.cdc_sync_interval
        self.step_count = 0
        self.algorithm = args.cdc_algorithm
        self.offload_outer_opt = args.cdc_offload_outer_opt
        self.outer_lr = float(args.cdc_outer_lr)
        self.dense_outer_lr_arg = float(args.cdc_dense_outer_lr)
        self.expert_outer_lr_arg = float(args.cdc_moe_expert_outer_lr)
        self.dense_outer_lr = (
            self.outer_lr if self.dense_outer_lr_arg < 0.0 else self.dense_outer_lr_arg
        )
        self.expert_outer_lr = (
            self.outer_lr if self.expert_outer_lr_arg < 0.0 else self.expert_outer_lr_arg
        )
        self.num_shards = args.cdc_num_shards
        self.dc_lambda = args.cdc_dc_lambda
        self.streaming_alpha = float(args.cdc_streaming_alpha)
        self.dense_alpha_arg = float(args.cdc_dense_alpha)
        self.router_alpha_arg = float(args.cdc_moe_router_alpha)
        self.expert_alpha_arg = float(args.cdc_moe_expert_alpha)
        self.router_sync_mode = str(args.cdc_moe_router_sync_mode).lower()
        self.dense_alpha = (
            self.streaming_alpha if self.dense_alpha_arg < 0.0 else self.dense_alpha_arg
        )
        self.router_alpha = (
            self.streaming_alpha if self.router_alpha_arg < 0.0 else self.router_alpha_arg
        )
        self.expert_alpha = (
            self.streaming_alpha if self.expert_alpha_arg < 0.0 else self.expert_alpha_arg
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
        self.expert_layerwise_selection = bool(args.cdc_moe_expert_layerwise_selection)
        self.expert_max_age_slots = int(args.cdc_moe_expert_max_age_slots)
        self.expert_min_age_slots = int(args.cdc_moe_expert_min_age_slots)
        self.blocking_full_sync_steps = int(args.cdc_blocking_full_sync_steps)
        self.verbose = args.cdc_verbose
        self.mixed_precision = args.bf16 or args.fp16
        self._named_model_param_list = self._iter_named_trainable_params_unique(self.model_chunks)
        (
            self._layer_prefix_to_global_idx,
            self._local_moe_module_to_global_key,
            self._global_moe_module_index,
        ) = self._build_global_identity_indices()
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
        self._token_load_module_to_tracker_indices: Dict[str, List[int]] = {}
        self._token_load_hooked_expert_module_ids = set()
        self._token_load_hook_handles = []
        tracked_params = self.tracked_model_param_list
        self.model_param_dtype = tracked_params[0].dtype if tracked_params else torch.float32
        # Follow Streaming DiLoCo: keep outer-state math in fp32, but keep communication low
        # precision by default to avoid doubling bandwidth.
        self.outer_state_dtype = torch.float32 if self.mixed_precision else self.model_param_dtype
        self.outer_comm_dtype = (
            self.model_param_dtype if self.mixed_precision else self.outer_state_dtype
        )
        # Initialized for all algorithms so checkpoint/state construction does not fail.
        self.next_shard_idx = 0
        self.next_expert_group_idx = 0
        self.next_expert_layer_idx = 0
        self.expert_sync_event_count = 0
        self._cdc_state_loaded = False
        self._expert_layer_to_tracker_indices: Dict[int, List[int]] = {}
        self._expert_layer_order: List[int] = []

        if self.verbose and self.track_expert_token_load and self._global_moe_module_index:
            preview_keys = sorted(self._global_moe_module_index.keys())[:8]
            print_rank_0(
                f"[CDC][MoEIndex] modules={len(self._global_moe_module_index)} "
                f"preview={preview_keys}"
            )

        if self.verbose:
            print_rank_0(
                f"[CDC] Initialized {self.algorithm} optimizer. Sync interval: "
                f"{self.sync_interval}, Shards: {self.num_shards}, "
                f"outer_lr={self.outer_lr}, "
                f"dense_outer_lr={'inherit(' + str(self.dense_outer_lr) + ')' if self.dense_outer_lr_arg < 0.0 else self.dense_outer_lr}, "
                f"expert_outer_lr={'inherit(' + str(self.expert_outer_lr) + ')' if self.expert_outer_lr_arg < 0.0 else self.expert_outer_lr}, "
                f"outer_state_dtype={self.outer_state_dtype}, outer_comm_dtype={self.outer_comm_dtype}, "
                f"moe_param_mode={self.moe_param_mode}, "
                f"dense_alpha={'inherit(' + str(self.dense_alpha) + ')' if self.dense_alpha_arg < 0.0 else self.dense_alpha}, "
                f"router_refresh={self.router_sync_mode if self.enable_moe_router_refresh else 'off'}, "
                f"router_alpha={'inherit(' + str(self.router_alpha) + ')' if self.router_alpha_arg < 0.0 else self.router_alpha}, "
                f"expert_refresh={'on' if self.enable_moe_expert_refresh else 'off'}, "
                f"expert_alpha={'inherit(' + str(self.expert_alpha) + ')' if self.expert_alpha_arg < 0.0 else self.expert_alpha}, "
                f"expert_topk={self.expert_topk}, expert_score_mode={self.expert_score_mode}, "
                f"expert_layerwise_selection={self.expert_layerwise_selection}, "
                f"expert_min_age_slots={self.expert_min_age_slots}, "
                f"blocking_full_sync={'off' if self.blocking_full_sync_steps <= 0 else f'every_{self.blocking_full_sync_steps}_steps'}"
            )
        
        if self.algorithm == 'diloco':
            self._init_diloco_state()
        elif self.algorithm in ['streaming', 'dc']:
            self._init_streaming_state()
        else:
            raise ValueError(f"Unknown DiLoCo algorithm: {self.algorithm}")

        if self.track_expert_token_load:
            self._register_token_load_expert_hooks()

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
        return [p for _, p in self._named_model_param_list]

    @property
    def tracked_model_param_list(self):
        return [p for _, p in self._tracked_named_model_params]

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

    def _build_cdc_state(self):
        state = {
            "algorithm": self.algorithm,
            "step_count": self.step_count,
            "next_shard_idx": self.next_shard_idx,
            "next_expert_group_idx": self.next_expert_group_idx,
            "next_expert_layer_idx": self.next_expert_layer_idx,
            "expert_sync_event_count": self.expert_sync_event_count,
        }

        if self.algorithm == 'diloco' and getattr(self, "original_snapshot", None) is not None:
            state["diloco"] = self._serialize_diloco_state()
        elif self.algorithm in ['streaming', 'dc'] and getattr(self, "shard_tracker", None) is not None:
            state["streaming_layout_version"] = 3
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
                "global_num_params": int(tracker["global_num_params"]),
                "last_score": float(tracker["last_score"]),
                "last_token_load": float(tracker.get("last_token_load", 0.0)),
                "token_load_accum": float(tracker.get("token_load_accum", 0.0)),
                "sent_at_expert_event": int(tracker.get("sent_at_expert_event", 0)),
                "moe_module_key": tracker.get("moe_module_key"),
                "moe_layer_idx": tracker.get("moe_layer_idx"),
                "local_expert_idx": tracker.get("local_expert_idx"),
                "global_expert_idx": tracker.get("global_expert_idx"),
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
        self.next_expert_layer_idx = cdc_state.get(
            "next_expert_layer_idx", self.next_expert_layer_idx
        )
        self.expert_sync_event_count = cdc_state.get(
            "expert_sync_event_count", self.expert_sync_event_count
        )

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
            args = get_args()
            if layout_version < 3 and (
                int(getattr(args, "pipeline_model_parallel_size", 1)) > 1
                or int(getattr(args, "expert_model_parallel_size", 1)) > 1
            ):
                raise ValueError(
                    "This CDC checkpoint uses a layout without global layer/expert identities. "
                    "Resume under PP>1 or EP>1 is not supported because local layer/expert "
                    "indices are ambiguous."
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
            tracker["moe_layer_idx"] = shard_state.get(
                "moe_layer_idx", tracker.get("moe_layer_idx")
            )
            tracker["local_expert_idx"] = shard_state.get(
                "local_expert_idx", tracker.get("local_expert_idx")
            )
            tracker["global_expert_idx"] = shard_state.get(
                "global_expert_idx", tracker.get("global_expert_idx")
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
    def _layer_prefix_from_name(name: str) -> Optional[str]:
        normalized = CDCOptimizer._normalize_module_name(name)
        match = re.match(r"(.*?\.layers\.\d+)(?:\.|$)", normalized)
        if match is None:
            return None
        return match.group(1)

    @staticmethod
    def _local_expert_prefix_from_name(param_name: str) -> Optional[Tuple[str, int]]:
        normalized = CDCOptimizer._normalize_module_name(param_name)
        match = re.search(r"(.*)\.experts\.local_experts\.(\d+)(?:\.|$)", normalized)
        if match is None:
            return None
        return match.group(1), int(match.group(2))

    def _build_global_identity_indices(
        self,
    ) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, torch.nn.Module]]:
        layer_to_global_idx: Dict[str, int] = {}
        local_moe_to_global: Dict[str, str] = {}
        global_moe_index: Dict[str, torch.nn.Module] = {}

        for name, module in self._iter_named_modules_unique(self.model_chunks):
            normalized_name = self._normalize_module_name(name)
            layer_number = getattr(module, "layer_number", None)
            if layer_number is not None:
                prefix = self._layer_prefix_from_name(normalized_name)
                if prefix is not None:
                    global_idx = int(layer_number) - 1
                    previous = layer_to_global_idx.get(prefix)
                    if previous is not None and previous != global_idx:
                        raise ValueError(
                            f"Layer prefix {prefix!r} maps to both global layer "
                            f"{previous} and {global_idx}."
                        )
                    layer_to_global_idx[prefix] = global_idx

            if not normalized_name or not (
                hasattr(module, "token_dispatcher") and hasattr(module, "experts")
            ):
                continue

            global_key = self._globalize_moe_module_name(normalized_name, module)
            previous_module = global_moe_index.get(global_key)
            if previous_module is not None and previous_module is not module:
                raise ValueError(f"Duplicate global MoE module key detected: {global_key!r}")
            local_moe_to_global[normalized_name] = global_key
            global_moe_index[global_key] = module

        return layer_to_global_idx, local_moe_to_global, global_moe_index

    def _global_layer_index_for_param(self, param_name: str) -> Optional[int]:
        prefix = self._layer_prefix_from_name(param_name)
        if prefix is None:
            return None
        if prefix in self._layer_prefix_to_global_idx:
            return self._layer_prefix_to_global_idx[prefix]

        if int(getattr(get_args(), "pipeline_model_parallel_size", 1)) > 1:
            raise ValueError(
                f"Cannot map parameter {param_name!r} to a global layer id under PP. "
                "Expected the owning layer module to expose layer_number."
            )

        local_idx = self._parse_layer_index(param_name)
        if local_idx is None:
            return None
        return local_idx

    def _globalize_moe_module_name(self, local_name: str, module: torch.nn.Module) -> str:
        normalized_name = self._normalize_module_name(local_name)
        layer_number = getattr(module, "layer_number", None)
        if layer_number is None:
            if int(getattr(get_args(), "pipeline_model_parallel_size", 1)) > 1:
                raise ValueError(
                    f"MoE module {normalized_name!r} is missing layer_number; "
                    "cannot build a globally unique CDC expert key under PP."
                )
            return normalized_name

        match = re.match(r"(.*?\.layers\.)(\d+)(\..*)?$", normalized_name)
        if match is None:
            raise ValueError(
                f"MoE module {normalized_name!r} does not contain a '.layers.<idx>' segment."
            )
        prefix, _, suffix = match.groups()
        return f"{prefix}{int(layer_number) - 1}{suffix or ''}"

    def _routed_expert_param_metadata(self, param_name: str) -> Optional[Dict[str, Any]]:
        expert_prefix = self._local_expert_prefix_from_name(param_name)
        if expert_prefix is None:
            return None

        local_module_key, local_expert_idx = expert_prefix
        local_module_key = self._normalize_module_name(local_module_key)
        global_module_key = self._local_moe_module_to_global_key.get(local_module_key)
        if global_module_key is None:
            raise ValueError(
                f"Could not map routed expert parameter {param_name!r} to a MoE module."
            )

        module = self._global_moe_module_index[global_module_key]
        local_expert_indices = getattr(module, "local_expert_indices", None)
        if local_expert_indices is None:
            raise ValueError(
                f"MoE module {global_module_key!r} is missing local_expert_indices; "
                "cannot build a globally unique CDC expert key under EP."
            )
        if local_expert_idx < 0 or local_expert_idx >= len(local_expert_indices):
            raise ValueError(
                f"Local expert index {local_expert_idx} from {param_name!r} is outside "
                f"module {global_module_key!r} local expert layout {local_expert_indices}."
            )

        global_expert_idx = int(local_expert_indices[local_expert_idx])
        layer_idx = self._parse_layer_index(global_module_key)
        group_key = f"{global_module_key}.experts.global_experts.{global_expert_idx}"
        return {
            "group_key": group_key,
            "moe_module_key": global_module_key,
            "moe_layer_idx": layer_idx,
            "local_expert_idx": local_expert_idx,
            "global_expert_idx": global_expert_idx,
        }

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

    def _expert_group_sort_key(self, group_name: str) -> Tuple[int, int, str]:
        normalized = self._normalize_module_name(group_name)
        match = re.match(r"(.*)\.experts\.(?:local_experts|global_experts)\.(\d+)$", normalized)
        layer_idx = self._parse_layer_index(match.group(1)) if match is not None else None
        expert_idx = int(match.group(2)) if match is not None else None
        return (
            layer_idx if layer_idx is not None else 10**9,
            expert_idx if expert_idx is not None else 10**9,
            normalized,
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

    @staticmethod
    def _tokens_per_expert_to_float_list(value: Any) -> List[float]:
        if torch.is_tensor(value):
            if value.dim() != 1:
                raise ValueError(
                    f"Expected tokens_per_expert to be a 1-D tensor, got shape {tuple(value.shape)}."
                )
            tensor = value.detach().to(dtype=torch.float32)
            return [float(v) for v in tensor.cpu().tolist()]
        if isinstance(value, (list, tuple)):
            if any(isinstance(item, (list, tuple)) for item in value):
                raise ValueError("Expected tokens_per_expert to be a flat list or tuple.")
            return [float(v) for v in value]
        raise TypeError(f"Expected tokens_per_expert tensor/list/tuple, got {type(value).__name__}.")

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
    ) -> Dict[str, Dict[str, Any]]:
        grouped: Dict[str, Dict[str, Any]] = {}
        for name, param in self._expert_named_model_params:
            metadata = self._routed_expert_param_metadata(name)
            if metadata is None:
                continue
            group_key = metadata["group_key"]
            group = grouped.get(group_key)
            if group is None:
                group = grouped[group_key] = {
                    "named_params": [],
                    "moe_module_key": metadata["moe_module_key"],
                    "moe_layer_idx": metadata["moe_layer_idx"],
                    "local_expert_idx": metadata["local_expert_idx"],
                    "global_expert_idx": metadata["global_expert_idx"],
                }
            group["named_params"].append((name, param))
        return grouped

    def _build_tracker(
        self,
        param_refs: List[torch.nn.Parameter],
        *,
        score_reduce_groups: Optional[List[Any]] = None,
        display_name: str,
        moe_module_key: Optional[str] = None,
        moe_layer_idx: Optional[int] = None,
        local_expert_idx: Optional[int] = None,
        global_expert_idx: Optional[int] = None,
        comm_dtype: Optional[torch.dtype] = None,
        apply_alpha: Optional[float] = None,
        outer_lr: Optional[float] = None,
    ) -> Dict[str, Any]:
        tracker_outer_lr = self.outer_lr if outer_lr is None else float(outer_lr)
        tracker = {
            "display_name": display_name,
            "param_refs": param_refs,
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
            "token_load_step_accum": 0.0,
            "sent_at_expert_event": 0,
            "moe_module_key": moe_module_key,
            "moe_layer_idx": moe_layer_idx,
            "local_expert_idx": local_expert_idx,
            "global_expert_idx": global_expert_idx,
            "score_reduce_groups": list(score_reduce_groups or []),
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
        global_numel = self._all_reduce_scalar_sum_across_groups(
            local_numel, tracker["score_reduce_groups"]
        )
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

            tracker["sent_at_step"] = 0
            tracker["old_sent_at_step"] = 0
            tracker["next_receive_step"] = 0
            tracker["last_score"] = 0.0
            tracker["last_token_load"] = 0.0
            tracker["token_load_accum"] = 0.0
            tracker["token_load_step_accum"] = 0.0
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
        self.next_expert_layer_idx = 0
        self.expert_sync_event_count = 0

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
                layer_idx = self._global_layer_index_for_param(name)
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
        self.next_expert_layer_idx = 0
        self._expert_layer_to_tracker_indices = {}
        self._expert_layer_order = []

        tp_group = mpu.get_tensor_model_parallel_group()
        pp_group = mpu.get_pipeline_model_parallel_group()
        dense_score_reduce_groups = [tp_group, pp_group]
        expert_score_reduce_groups = [mpu.get_expert_tensor_parallel_group()]

        for shard_idx in range(self.num_shards):
            param_refs = shard_to_param_refs.get(shard_idx, [])
            tracker = self._build_tracker(
                param_refs,
                score_reduce_groups=dense_score_reduce_groups,
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
                score_reduce_groups=dense_score_reduce_groups,
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
                expert_group = expert_groups[expert_group_name]
                expert_param_refs = [param for _, param in expert_group["named_params"]]
                moe_module_key = expert_group["moe_module_key"]
                moe_layer_idx = expert_group["moe_layer_idx"]
                local_expert_idx = expert_group["local_expert_idx"]
                global_expert_idx = expert_group["global_expert_idx"]
                tracker = self._build_tracker(
                    expert_param_refs,
                    score_reduce_groups=expert_score_reduce_groups,
                    display_name=expert_group_name,
                    moe_module_key=moe_module_key,
                    moe_layer_idx=moe_layer_idx,
                    local_expert_idx=local_expert_idx,
                    global_expert_idx=global_expert_idx,
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
                            f"module_key={moe_module_key} layer_idx={moe_layer_idx} "
                            f"local_expert_idx={local_expert_idx} "
                            f"global_expert_idx={global_expert_idx}"
                        )
            self._rebuild_expert_layer_index()
            if self.expert_layerwise_selection and not self._expert_layer_order:
                raise ValueError(
                    "cdc_moe_expert_layerwise_selection requires layer ids in routed expert names."
                )
            if self.verbose and self.expert_layerwise_selection:
                layer_summary = {
                    layer_idx: len(indices)
                    for layer_idx, indices in self._expert_layer_to_tracker_indices.items()
                }
                print_rank_0(
                    f"[CDC][ExpertLayerwise] layers={self._expert_layer_order} "
                    f"groups_per_layer={layer_summary}"
                )

    @staticmethod
    def _tracker_has_any_params(tracker: Dict[str, Any]) -> bool:
        return bool(tracker.get("param_refs")) or bool(tracker.get("params"))

    def _rebuild_expert_layer_index(self) -> None:
        self._expert_layer_to_tracker_indices = {}
        if not getattr(self, "expert_shard_tracker", None):
            self._expert_layer_order = []
            return

        for idx, tracker in sorted(self.expert_shard_tracker.items()):
            layer_idx = tracker.get("moe_layer_idx")
            if layer_idx is None:
                continue
            self._expert_layer_to_tracker_indices.setdefault(int(layer_idx), []).append(idx)

        self._expert_layer_order = sorted(self._expert_layer_to_tracker_indices.keys())

    @staticmethod
    def _parse_layer_index(param_name: str) -> Optional[int]:
        # Support common Megatron naming patterns:
        # - "...layers.0...."
        # - "...decoder.layers.0...."
        m = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", param_name)
        if m is None:
            return None
        return int(m.group(1))

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
        layer_idx = self._global_layer_index_for_param(name)
        if layer_idx is not None and num_layers > 0 and decoder_shards > 0:
            decoder_shard_idx = self._layer_to_decoder_shard(layer_idx, num_layers, decoder_shards)
            return embedding_shards + decoder_shard_idx

        # 3) Misc (final norm, output bias, etc.)
        # Keep embeddings isolated: place misc into the last shard if possible.
        return self.num_shards - 1

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

    @staticmethod
    def _canonical_group(group):
        if isinstance(group, list):
            return group[0] if group else None
        return group

    def _all_reduce_scalar_sum_across_groups(self, value: float, groups: List[Any]) -> float:
        reduced = float(value)
        for group in groups:
            reduced = self._all_reduce_scalar_sum(reduced, group=group)
        return float(reduced)

    def _all_reduce_scalar_sum(self, value: float, group) -> float:
        group = self._canonical_group(group)
        if group is None:
            return float(value)
        if dist.get_world_size(group=group) <= 1:
            return float(value)

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        t = torch.tensor(float(value), device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
        return float(t.item())

    def _all_reduce_vector_sum(self, values: List[float], group) -> List[float]:
        if not values:
            return []
        group = self._canonical_group(group)
        if group is None:
            return [float(v) for v in values]
        if dist.get_world_size(group=group) <= 1:
            return [float(v) for v in values]

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        tensor = torch.tensor(values, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return [float(v) for v in tensor.cpu().tolist()]

    def _reduce_expert_token_values(
        self, ordered_indices: List[int], local_values: List[float]
    ) -> List[float]:
        if not ordered_indices:
            return []

        reduced_values = [float(v) for v in local_values]
        score_reduce_groups = self.expert_shard_tracker[ordered_indices[0]].get(
            "score_reduce_groups", []
        )

        # Expert TP ranks process the same global expert identity. Average across ExpertTP
        # to keep selection/logging consistent without multiplying the routed token count.
        for group in score_reduce_groups:
            group = self._canonical_group(group)
            world_size = 1 if group is None else dist.get_world_size(group=group)
            reduced_values = self._all_reduce_vector_sum(reduced_values, group=group)
            if world_size > 1:
                reduced_values = [value / world_size for value in reduced_values]

        return self._all_reduce_vector_sum(reduced_values, group=self.cdc_group)

    def _register_token_load_expert_hooks(self) -> None:
        if not getattr(self, "expert_shard_tracker", None):
            return

        module_to_tracker_indices: Dict[str, List[int]] = {}
        for tracker_idx, tracker in self.expert_shard_tracker.items():
            module_key = tracker.get("moe_module_key")
            local_expert_idx = tracker.get("local_expert_idx")
            if module_key is None or local_expert_idx is None:
                continue
            module_to_tracker_indices.setdefault(module_key, []).append(tracker_idx)

        self._token_load_module_to_tracker_indices = module_to_tracker_indices
        hooked_count = 0

        for module_key in sorted(module_to_tracker_indices.keys()):
            module = self._global_moe_module_index.get(module_key)
            experts_module = getattr(module, "experts", None) if module is not None else None
            if experts_module is None or not hasattr(experts_module, "register_forward_pre_hook"):
                raise ValueError(
                    f"Could not register CDC token-load hook for MoE module {module_key!r}."
                )

            experts_module_id = id(experts_module)
            if experts_module_id in self._token_load_hooked_expert_module_ids:
                continue

            def token_load_pre_hook(
                hook_module,
                inputs,
                _module_key=module_key,
            ):
                if torch.is_grad_enabled():
                    self._accumulate_token_load_from_expert_inputs(
                        _module_key, hook_module, inputs
                    )

            handle = experts_module.register_forward_pre_hook(token_load_pre_hook)
            self._token_load_hooked_expert_module_ids.add(experts_module_id)
            self._token_load_hook_handles.append(handle)
            hooked_count += 1

        if self.verbose:
            print_rank_0(
                f"[CDC][TokenLoadSource] source=experts_forward_pre_hook "
                f"hooked_modules={hooked_count} tracked_modules={len(module_to_tracker_indices)}"
            )

    def _accumulate_token_load_from_expert_inputs(
        self, module_key: str, experts_module: Any, inputs: Any
    ) -> None:
        if not self.track_expert_token_load:
            return
        if not isinstance(inputs, (tuple, list)) or len(inputs) < 2:
            return

        expert_loads = self._tokens_per_expert_to_float_list(inputs[1])
        if not expert_loads:
            return

        for tracker_idx in self._token_load_module_to_tracker_indices.get(module_key, []):
            tracker = self.expert_shard_tracker[tracker_idx]
            local_expert_idx = tracker.get("local_expert_idx")
            if local_expert_idx is None:
                continue
            local_expert_idx = int(local_expert_idx)
            if 0 <= local_expert_idx < len(expert_loads):
                load_value = float(expert_loads[local_expert_idx])
                tracker["token_load_step_accum"] = float(
                    tracker.get("token_load_step_accum", 0.0) + load_value
                )

        if self.verbose and not self._token_load_source_debug_printed:
            print_rank_0(
                "[CDC][TokenLoadSource] "
                f"module_key={module_key} "
                f"experts_type={type(experts_module).__name__} "
                f"load_len={len(expert_loads)} "
                f"load_sum={float(sum(expert_loads)):.6e} "
                f"load_preview={expert_loads[:8]}"
            )
            self._token_load_source_debug_printed = True

    def _commit_moe_expert_token_load_stats(self) -> None:
        if not self.track_expert_token_load:
            return

        step_load_sum = 0.0
        accum_load_sum = 0.0
        for tracker in self.expert_shard_tracker.values():
            step_load = float(tracker.get("token_load_step_accum", 0.0))
            tracker["last_token_load"] = step_load
            tracker["token_load_accum"] = float(
                tracker.get("token_load_accum", 0.0) + step_load
            )
            tracker["token_load_step_accum"] = 0.0
            step_load_sum += step_load
            accum_load_sum += float(tracker.get("token_load_accum", 0.0))

        if self.verbose and not self._token_load_step_debug_printed:
            print_rank_0(
                f"[CDC][TokenLoadStep] step={self.step_count} "
                f"source=experts_forward_pre_hook "
                f"tracked_experts={len(getattr(self, 'expert_shard_tracker', {}))} "
                f"step_load_sum_local={step_load_sum:.6e} "
                f"accum_load_sum_local={accum_load_sum:.6e}"
            )
            self._token_load_step_debug_printed = True

    def _discard_pending_moe_expert_token_load_stats(self) -> None:
        if not self.track_expert_token_load:
            return
        for tracker in self.expert_shard_tracker.values():
            tracker["token_load_step_accum"] = 0.0

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
            if self.track_expert_token_load:
                self._commit_moe_expert_token_load_stats()

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
        elif self.track_expert_token_load:
            self._discard_pending_moe_expert_token_load_stats()

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
            if self.router_sync_mode == 'dense':
                self._try_initiate_router_sync(sync_reason='dense')
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
                if self.router_sync_mode == 'expert':
                    self._try_initiate_router_sync(sync_reason='expert')
                if batch_immediate_completion:
                    self._complete_tracker_sync(
                        self.expert_shard_tracker,
                        tracker_indices=expert_group_indices,
                        tracker_kind='expert-group',
                    )
                    for expert_group_idx in expert_group_indices:
                        self.expert_shard_tracker[expert_group_idx]["next_receive_step"] = 0

    def _try_initiate_router_sync(self, *, sync_reason: str) -> None:
        if not self.enable_moe_router_refresh:
            return

        router_tracker = self.router_tracker.get(0)
        if router_tracker is not None and router_tracker.get("next_receive_step", 0) <= self.step_count:
            self._initiate_tracker_sync(
                self.router_tracker, 0, tracker_kind='router'
            )
        elif self.verbose:
            print_rank_0(
                f"[CDC] Step {self.step_count}: Skip router sync on {sync_reason} slot "
                f"because previous router sync is still in flight until step "
                f"{router_tracker.get('next_receive_step', 0) if router_tracker is not None else 0}."
            )

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
            self._complete_tracker_sync(
                self.expert_shard_tracker,
                tracker_indices=tracker_indices,
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
        if tracker_kind == 'expert-group':
            if expert_event_index is not None:
                tracker["sent_at_expert_event"] = int(expert_event_index)
            tracker["token_load_accum"] = 0.0
            tracker["token_load_step_accum"] = 0.0
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

    def _expert_tracker_is_slot_stale(
        self, tracker: Dict[str, Any], upcoming_event: int
    ) -> bool:
        if self.expert_max_age_slots <= 0:
            return False
        if self._expert_tracker_is_unsent(tracker):
            return False
        return self._expert_tracker_age_slots(tracker, upcoming_event) >= self.expert_max_age_slots

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
        reduced_token_values = self._reduce_expert_token_values(
            ordered_indices, [token_scores_local[idx] for idx in ordered_indices]
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

    def _select_layerwise_expert_candidates(
        self, candidate_indices: List[int]
    ) -> Tuple[List[int], Optional[int]]:
        if not self.expert_layerwise_selection:
            return candidate_indices, None
        if not self._expert_layer_order:
            return candidate_indices, None

        candidate_set = set(candidate_indices)
        layer_count = len(self._expert_layer_order)
        start_pos = self.next_expert_layer_idx % layer_count
        for offset in range(layer_count):
            layer_pos = (start_pos + offset) % layer_count
            layer_idx = self._expert_layer_order[layer_pos]
            layer_candidates = [
                idx
                for idx in self._expert_layer_to_tracker_indices.get(layer_idx, [])
                if idx in candidate_set
            ]
            if layer_candidates:
                self.next_expert_layer_idx = (layer_pos + 1) % layer_count
                return layer_candidates, layer_idx

        return [], None

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
            reduced_token_values = self._reduce_expert_token_values(ordered_indices, local_token_values)
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
            slot_stale = self._expert_tracker_is_slot_stale(tracker, upcoming_event)
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
                "global_expert_idx": tracker.get("global_expert_idx"),
                "next_receive_step": next_receive_step,
                "age_slots": age_slots,
                "slot_stale": slot_stale,
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
        layer_scope: Optional[int] = None,
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
            f"layerwise={int(self.expert_layerwise_selection)}, layer_scope={layer_scope}, "
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
                f"global_expert={row['global_expert_idx']} "
                f"next_recv={row['next_receive_step']} "
                f"age_slots={row['age_slots']} "
                f"slot_stale={int(row['slot_stale'])} "
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
        global_candidate_indices = [
            idx
            for idx, tracker in self.expert_shard_tracker.items()
            if self._expert_tracker_is_available(tracker)
        ]
        candidate_indices, layer_scope = self._select_layerwise_expert_candidates(
            global_candidate_indices
        )
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
            eligible_indices=set(candidate_indices),
        )
        for idx in unsent:
            selected.append(idx)
            selected_set.add(idx)
            selected_reasons[idx] = 'mandatory_unsent'

        remaining = max_select - len(selected)
        stale_candidates: List[Tuple[int, int]] = []
        if remaining > 0:
            for idx in candidate_indices:
                if idx in selected_set:
                    continue
                tracker = self.expert_shard_tracker[idx]
                age_slots = self._expert_tracker_age_slots(tracker, upcoming_event)
                slot_stale = self._expert_tracker_is_slot_stale(tracker, upcoming_event)
                if slot_stale:
                    stale_candidates.append((age_slots, idx))

            if stale_candidates:
                stale_candidates.sort(key=lambda item: (-item[0], item[1]))
                for _, idx in stale_candidates[:remaining]:
                    selected.append(idx)
                    selected_set.add(idx)
                    selected_reasons[idx] = 'stale_slot'

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
                layer_scope=layer_scope,
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
                layer_scope=layer_scope,
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
            layer_scope=layer_scope,
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
        if tracker_kind == 'expert-group':
            if expert_event_index is not None:
                tracker["sent_at_expert_event"] = int(expert_event_index)
            tracker["token_load_accum"] = 0.0
            tracker["token_load_step_accum"] = 0.0

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
        tracker_idx: Optional[int] = None,
        *,
        tracker_indices: Optional[List[int]] = None,
        tracker_kind: str,
        reload_main_params: bool = True,
        use_algorithm_specific_update: bool = True,
        force_full_copy: bool = False,
    ) -> None:
        """Complete one or more tracker syncs (receive & update)."""
        if tracker_indices is None:
            if tracker_idx is None:
                return
            tracker_indices = [tracker_idx]
        if not tracker_indices:
            return

        tracker_entries: List[Tuple[int, Dict[str, Any], List[torch.Tensor], List[torch.Tensor]]] = []
        batched_sync_grads: List[torch.Tensor] = []

        for tracker_idx in tracker_indices:
            tracker = tracker_dict[tracker_idx]
            global_params = tracker["params"]
            staged_params = tracker["staged_params"]

            sync_grads: List[torch.Tensor] = []
            for p_global, p_staged in zip(global_params, staged_params):
                grad_tensor = p_global.data.clone()
                grad_tensor.sub_(p_staged.data)
                sync_grads.append(grad_tensor)
            tracker_entries.append((tracker_idx, tracker, global_params, sync_grads))
            batched_sync_grads.extend(sync_grads)

        batch_comm_dtype = tracker_entries[0][1].get("comm_dtype", self.outer_comm_dtype)
        self._all_reduce_flattened(
            batched_sync_grads, communication_dtype=batch_comm_dtype
        )

        for entry_tracker_idx, tracker, global_params, sync_grads in tracker_entries:
            tracker_label = tracker.get("display_name", str(entry_tracker_idx))

            total_norm_sq = 0.0
            for grad_tensor in sync_grads:
                total_norm_sq += float(grad_tensor.float().pow(2).sum().item())

            total_norm_sq = self._all_reduce_scalar_sum_across_groups(
                total_norm_sq, tracker.get("score_reduce_groups", [])
            )

            tracker["last_score"] = total_norm_sq

            if self.verbose:
                duration = time.time() - tracker.get("sync_start_time", time.time())
                print_rank_0(
                    f"[CDC] Step {self.step_count}: Completed sync for {tracker_kind} "
                    f"{tracker_label} in {duration:.4f}s. Score (Norm^2): {total_norm_sq:.4e}"
                )

            if tracker["outer_optimizer"] is not None:
                for p_global, avg_delta in zip(global_params, sync_grads):
                    if p_global.grad is None:
                        p_global.grad = torch.zeros_like(p_global.data)
                    p_global.grad.copy_(avg_delta)
                tracker["outer_optimizer"].step()
                tracker["outer_optimizer"].zero_grad(set_to_none=True)
            else:
                for p_global, avg_delta in zip(global_params, sync_grads):
                    p_global.data.sub_(avg_delta)

            if use_algorithm_specific_update and self.algorithm == 'dc' and tracker_kind == 'dense-shard':
                self.delay_compensation(tracker)
            else:
                self._apply_global_params_to_local(tracker, force_full_copy=force_full_copy)

        if self.mixed_precision and reload_main_params:
            self.inner_optimizer.reload_model_params()
            
    def delay_compensation(self, tracker):
        """ Update Local Params (Algorithm Specific)"""
        staged_params = tracker["staged_params"]
        param_refs = tracker["param_refs"]
        global_params = tracker["params"]

        # ---- Update-space delay compensation ----
        # Effective staleness in inner steps (may be > self.delay if scheduling shifts)
        tau = int(self.step_count - tracker["sent_at_step"])
        tau = max(tau, 1)

        eps = 1e-8  # 数值稳定性
        args = get_args()
        lam0 = self.dc_lambda  # lambda base
        lam_max = float(args.cdc_dc_lambda_max)  # 上限

        score_reduce_groups = tracker.get("score_reduce_groups", [])

        def _reduce_shard_sum(x: float) -> float:
            return self._all_reduce_scalar_sum_across_groups(float(x), score_reduce_groups)

        # Helper: compute sum of squares in fp32 (device-agnostic)
        def _sqsum_fp32(t: torch.Tensor) -> float:
            return float(t.float().pow(2).sum().item())

        def _dc_terms(
            p_staged: torch.Tensor,
            p_local: torch.nn.Parameter,
            p_global: torch.Tensor,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Compute update-space delay compensation terms in fp32."""
            if self.offload_outer_opt:
                theta1 = p_local.detach().to("cpu", dtype=torch.float32)
                z = p_global.data.to(torch.float32)
            else:
                theta1 = p_local.data.to(torch.float32)
                z = p_global.data.to(torch.float32)

            theta0 = p_staged.data.to(torch.float32)
            D = z - theta0
            u = (theta0 - theta1).div(tau)
            c = (u * u) * D
            return u, c, z

        # Pass 1: accumulate per-shard norms across this tracker.
        local_u2 = 0.0
        local_c2 = 0.0

        for p_staged, p_local, p_global in zip(staged_params, param_refs, global_params):
            u, c, _ = _dc_terms(p_staged, p_local, p_global)
            local_u2 += _sqsum_fp32(u)
            local_c2 += _sqsum_fp32(c)

        u2 = _reduce_shard_sum(local_u2)
        c2 = _reduce_shard_sum(local_c2)

        norm_u = math.sqrt(max(u2, 0.0))
        norm_c = math.sqrt(max(c2, 0.0))

        lam = lam0 * norm_u / (norm_c + eps)
        lam = float(min(lam, lam_max))

        # Pass 2: apply compensated update with one lambda for the shard.
        for p_staged, p_local, p_global in zip(staged_params, param_refs, global_params):
            u, c, z = _dc_terms(p_staged, p_local, p_global)
            u_hat = u + lam * c

            target = z.to(torch.float32) - (tau * u_hat)

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
