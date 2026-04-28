"""CDC optimizer v2.

This file is intentionally self-contained.  The existing ``cdc_optimizer.py`` is
the battle-tested implementation and should remain untouched.  V2 keeps the same
high-level behavior for the currently useful paths, while removing the
lr/weight-decay tracking and avoiding legacy checkpoint fallbacks.

Supported paths:
- DiLoCo full-model sync.
- Streaming/DC dense shard sync.
- Streaming dense-expert-hybrid sync with:
  - dense shard round-robin sync,
  - dedicated router sync every dense slot,
  - optional router hard overwrite via ``--cdc-moe-router-alpha 0.0``,
  - expert top-k selection with mandatory first-send, max age, min age,
    update-norm/token-load/mixed scores,
  - optional blocking full sync.

Important constraints:
- dense-expert-hybrid currently requires expert_model_parallel_size == 1.
- PP is handled by using global layer ids for dense shard placement and expert
  slot keys; ranks that do not own a slot hold an empty tracker.
- TP is handled by reducing scalar metadata across TP and by relying on CDC
  groups that match the same TP/PP partition for tensor all-reduce shapes.
"""

import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.optim import SGD

from megatron.core import mpu
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.transformer.module import MegatronModule
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0


@dataclass
class CDCTracker:
    """One globally meaningful CDC sync slot.

    ``slot_id`` and ``display_name`` must be identical across all ranks in the
    relevant model-parallel group.  Some PP ranks may own zero tensors for a
    slot; that is expected and is represented by empty ``param_refs``.
    """

    slot_id: int
    display_name: str
    kind: str
    param_refs: List[torch.nn.Parameter]
    params: List[torch.Tensor]
    staged_params: List[torch.Tensor]
    global_num_params: int
    comm_dtype: torch.dtype
    outer_optimizer: Optional[torch.optim.Optimizer] = None
    sent_at_step: int = 0
    old_sent_at_step: int = 0
    next_receive_step: int = 0
    last_score: float = 0.0
    last_token_load: float = 0.0
    token_load_accum: float = 0.0
    sent_at_expert_event: int = 0
    sync_start_time: float = 0.0
    moe_module_key: Optional[str] = None
    local_expert_idx: Optional[int] = None

    def has_local_params(self) -> bool:
        return bool(self.param_refs)

    def has_global_params(self) -> bool:
        return self.global_num_params > 0

    def is_in_flight(self, step_count: int) -> bool:
        return self.next_receive_step > step_count


class CDCOptimizerV2(MegatronOptimizer):
    """Single-file CDC optimizer rewrite.

    The class is deliberately explicit instead of plugin-like.  The complexity
    is kept in named sections: delegated optimizer interface, layout discovery,
    branch initialization, scheduling, sync lifecycle, checkpointing, and
    communication helpers.
    """

    STREAMING_LAYOUT_VERSION = 1

    def __init__(
        self,
        inner_optimizer: MegatronOptimizer,
        model_chunks: Optional[List[MegatronModule]] = None,
    ):
        self.inner_optimizer = inner_optimizer
        self.model_chunks = (
            model_chunks if model_chunks is not None else getattr(inner_optimizer, "model_chunks", None)
        )
        assert self.model_chunks is not None, "model_chunks must be provided to CDCOptimizerV2."
        self.config = self.inner_optimizer.config

        args = get_args()
        self.args = args

        self.cdc_group = mpu.get_cdc_parallel_group()
        self.tp_group = mpu.get_tensor_model_parallel_group()
        self.pp_group = mpu.get_pipeline_model_parallel_group()
        self.model_parallel_group = mpu.get_model_parallel_group()

        self.algorithm = str(args.cdc_algorithm).lower()
        self.sync_interval = int(args.cdc_sync_interval)
        self.outer_lr = float(args.cdc_outer_lr)
        self.num_shards = int(args.cdc_num_shards)
        self.delay = int(args.cdc_delay)
        self.streaming_alpha = float(args.cdc_streaming_alpha)
        self.router_alpha = float(args.cdc_moe_router_alpha)
        self.shard_pattern = str(args.cdc_shard_pattern).lower()
        self.offload_outer_opt = bool(args.cdc_offload_outer_opt)
        self.verbose = bool(args.cdc_verbose)
        self.blocking_full_sync_steps = int(args.cdc_blocking_full_sync_steps)

        self.dc_lambda = float(args.cdc_dc_lambda)
        self.dc_lambda_max = float(args.cdc_dc_lambda_max)
        self.dc_lambda_scope = str(args.cdc_dc_lambda_scope).lower()
        self.dc_type = str(args.cdc_dc_type).lower()
        self.dc_N = int(args.cdc_dc_N)

        self.tie_embeddings = not bool(args.untie_embeddings_and_output_weights)
        self.mixed_precision = bool(args.bf16 or args.fp16)

        self.moe_param_mode = str(args.cdc_moe_param_mode).lower()
        self.expert_sync_interval = int(args.cdc_moe_expert_sync_interval)
        self.expert_sync_offset = int(args.cdc_moe_expert_sync_offset)
        self.expert_selection = str(args.cdc_moe_expert_selection).lower()
        self.expert_topk = int(args.cdc_moe_expert_topk)
        self.expert_score_mode = str(args.cdc_moe_expert_score_mode).lower()
        self.expert_max_age_slots = int(args.cdc_moe_expert_max_age_slots)
        self.expert_min_age_slots = int(args.cdc_moe_expert_min_age_slots)
        self.expert_max_staleness = int(args.cdc_moe_expert_max_staleness)

        self.step_count = 0
        self._cdc_state_loaded = False
        self._token_load_source_debug_printed = False

        self._validate_configuration()

        self._named_model_param_list = self._iter_named_trainable_params_unique(self.model_chunks)
        self._layer_prefix_to_global_idx = self._build_layer_prefix_to_global_idx()
        self._moe_module_index = self._build_global_moe_module_index()
        self._router_named_model_params = self._collect_router_named_params(self._named_model_param_list)

        self.enable_moe_expert_refresh = (
            self.moe_param_mode == "dense-expert-hybrid" and self.expert_sync_interval > 0
        )
        self.enable_moe_router_refresh = (
            self.moe_param_mode == "dense-expert-hybrid"
            and self.algorithm == "streaming"
            and bool(self._router_named_model_params)
        )
        self.track_expert_token_load = (
            self.enable_moe_expert_refresh
            and self.expert_selection == "score"
            and self.expert_score_mode in {"token_load", "mixed"}
        )

        self._tracked_named_model_params = self._filter_named_params_for_cdc(
            self._named_model_param_list
        )
        tracked_params = self.tracked_model_param_list
        self.model_param_dtype = tracked_params[0].dtype if tracked_params else torch.float32
        self.outer_state_dtype = torch.float32 if self.mixed_precision else self.model_param_dtype
        self.outer_comm_dtype = self.model_param_dtype if self.mixed_precision else self.outer_state_dtype

        self.original_snapshot: List[torch.Tensor] = []
        self.outer_optimizer: Optional[torch.optim.Optimizer] = None
        self.dense_trackers: Dict[int, CDCTracker] = {}
        self.router_trackers: Dict[int, CDCTracker] = {}
        self.expert_trackers: Dict[int, CDCTracker] = {}
        self.next_dense_slot = 0
        self.next_expert_slot = 0
        self.expert_sync_event_count = 0

        if self.algorithm == "diloco":
            self._init_diloco_state()
        else:
            self._init_streaming_state()

        if self.verbose:
            print_rank_0(
                "[CDC-V2] Initialized "
                f"algorithm={self.algorithm}, sync_interval={self.sync_interval}, "
                f"num_shards={self.num_shards}, delay={self.delay}, "
                f"outer_state_dtype={self.outer_state_dtype}, outer_comm_dtype={self.outer_comm_dtype}, "
                f"moe_param_mode={self.moe_param_mode}, "
                f"router_refresh={'every_dense_slot' if self.enable_moe_router_refresh else 'off'}, "
                f"router_alpha={'inherit' if self.router_alpha < 0.0 else self.router_alpha}, "
                f"expert_refresh={'on' if self.enable_moe_expert_refresh else 'off'}, "
                f"expert_topk={self.expert_topk}, expert_score_mode={self.expert_score_mode}, "
                f"expert_min_age_slots={self.expert_min_age_slots}, "
                f"blocking_full_sync={'off' if self.blocking_full_sync_steps <= 0 else f'every_{self.blocking_full_sync_steps}_steps'}"
            )

    # ------------------------------------------------------------------
    # Delegated optimizer interface
    # ------------------------------------------------------------------

    @property
    def is_stub_optimizer(self):
        return getattr(self.inner_optimizer, "is_stub_optimizer", False)

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
    def param_groups(self):
        return self.inner_optimizer.param_groups

    @property
    def model_param_list(self) -> List[torch.nn.Parameter]:
        return [param for _, param in self._named_model_param_list]

    @property
    def tracked_model_param_list(self) -> List[torch.nn.Parameter]:
        return [param for _, param in self._tracked_named_model_params]

    def zero_grad(self, set_to_none: bool = True):
        return self.inner_optimizer.zero_grad(set_to_none=set_to_none)

    def get_loss_scale(self):
        return self.inner_optimizer.get_loss_scale()

    def get_parameters(self):
        return self.inner_optimizer.get_parameters()

    def get_grad_stats_parallel_group(self):
        return self.inner_optimizer.get_grad_stats_parallel_group()

    def get_grad_norm(self):
        return self.inner_optimizer.get_grad_norm()

    def clip_grad_norm(self, clip_grad: float):
        return self.inner_optimizer.clip_grad_norm(clip_grad)

    def count_zeros(self):
        return self.inner_optimizer.count_zeros()

    def save_parameter_state(self, filename: str):
        return self.inner_optimizer.save_parameter_state(filename)

    def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
        return self.inner_optimizer.load_parameter_state(
            filename, update_legacy_format=update_legacy_format
        )

    def prepare_grads(self):
        return self.inner_optimizer.prepare_grads()

    def step_with_ready_grads(self):
        raise NotImplementedError(
            "CDCOptimizerV2 only supports optimizer.step(); "
            "step_with_ready_grads would bypass CDC receive scheduling."
        )

    def reload_model_params(self):
        if self.step_count == 0 and not self._cdc_state_loaded:
            self._reset_outer_state_from_model()
        return self.inner_optimizer.reload_model_params()

    def state_dict(self, is_loading: bool = False):
        return {
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "cdc_state": self._build_cdc_state(),
        }

    def load_state_dict(self, state_dict):
        if "inner_optimizer" not in state_dict:
            self.inner_optimizer.load_state_dict(state_dict)
            self._cdc_state_loaded = False
            return
        self.inner_optimizer.load_state_dict(state_dict["inner_optimizer"])
        self._load_cdc_state(state_dict.get("cdc_state"))
        self._cdc_state_loaded = True

    def sharded_state_dict(self, model_sharded_state_dict, is_loading: bool = False):
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

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        if self.algorithm not in {"diloco", "streaming", "dc"}:
            raise ValueError(f"Unknown cdc_algorithm: {self.algorithm}")
        if self.sync_interval <= 0:
            raise ValueError(f"cdc_sync_interval must be > 0, got {self.sync_interval}")
        if self.num_shards <= 0:
            raise ValueError(f"cdc_num_shards must be > 0, got {self.num_shards}")
        if self.delay < 0:
            raise ValueError(f"cdc_delay must be >= 0, got {self.delay}")
        if not 0.0 <= self.streaming_alpha <= 1.0:
            raise ValueError(f"cdc_streaming_alpha must be in [0, 1], got {self.streaming_alpha}")
        if self.router_alpha > 1.0:
            raise ValueError(f"cdc_moe_router_alpha must be <= 1.0, got {self.router_alpha}")
        if self.blocking_full_sync_steps < 0:
            raise ValueError(
                f"cdc_blocking_full_sync_steps must be >= 0, got {self.blocking_full_sync_steps}"
            )
        if self.algorithm == "diloco" and self.blocking_full_sync_steps > 0:
            raise ValueError("DiLoCo already performs blocking full sync via cdc_sync_interval.")
        if self.shard_pattern not in {"sequential", "stride"}:
            raise ValueError(f"Unknown cdc_shard_pattern: {self.shard_pattern}")
        if self.moe_param_mode not in {"all", "dense-only", "dense-expert-hybrid"}:
            raise ValueError(f"Unknown cdc_moe_param_mode: {self.moe_param_mode}")
        if self.expert_selection not in {"round_robin", "score"}:
            raise ValueError(f"Unknown cdc_moe_expert_selection: {self.expert_selection}")
        if self.expert_score_mode not in {"update_norm", "token_load", "mixed"}:
            raise ValueError(f"Unknown cdc_moe_expert_score_mode: {self.expert_score_mode}")
        if self.expert_topk < 1:
            raise ValueError(f"cdc_moe_expert_topk must be >= 1, got {self.expert_topk}")
        if self.expert_sync_interval < 0:
            raise ValueError(
                f"cdc_moe_expert_sync_interval must be >= 0, got {self.expert_sync_interval}"
            )
        if self.expert_sync_offset < 0:
            raise ValueError(
                f"cdc_moe_expert_sync_offset must be >= 0, got {self.expert_sync_offset}"
            )
        if self.expert_max_age_slots < 0:
            raise ValueError(
                f"cdc_moe_expert_max_age_slots must be >= 0, got {self.expert_max_age_slots}"
            )
        if self.expert_min_age_slots < 0:
            raise ValueError(
                f"cdc_moe_expert_min_age_slots must be >= 0, got {self.expert_min_age_slots}"
            )
        if self.expert_max_staleness < 0:
            raise ValueError(
                f"cdc_moe_expert_max_staleness must be >= 0, got {self.expert_max_staleness}"
            )
        if self.dc_type not in {"update", "legacy"}:
            raise ValueError(f"Unknown cdc_dc_type: {self.dc_type}")
        if self.dc_lambda_scope not in {"shard", "tensor", "local"}:
            raise ValueError(f"Unknown cdc_dc_lambda_scope: {self.dc_lambda_scope}")
        if self.moe_param_mode == "dense-expert-hybrid" and self.algorithm != "streaming":
            raise ValueError("dense-expert-hybrid is supported only with cdc_algorithm=streaming in V2.")
        if self.moe_param_mode == "dense-expert-hybrid":
            ep_size = int(getattr(self.args, "expert_model_parallel_size", 1))
            if ep_size != 1:
                raise ValueError(
                    "CDCOptimizerV2 dense-expert-hybrid requires expert_model_parallel_size=1. "
                    "EP needs an explicit global expert-id mapping and is not silently approximated."
                )

    # ------------------------------------------------------------------
    # Parameter and layout discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        normalized = name
        while normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        return normalized

    @staticmethod
    def _iter_named_trainable_params_unique(
        model_chunks: Sequence[MegatronModule],
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        results: List[Tuple[str, torch.nn.Parameter]] = []
        seen_ids = set()
        for chunk in model_chunks:
            for name, param in chunk.named_parameters():
                if not getattr(param, "requires_grad", False):
                    continue
                param_id = id(param)
                if param_id in seen_ids:
                    continue
                seen_ids.add(param_id)
                results.append((name, param))
        return results

    @staticmethod
    def _iter_named_modules_unique(
        model_chunks: Sequence[MegatronModule],
    ) -> List[Tuple[str, torch.nn.Module]]:
        results: List[Tuple[str, torch.nn.Module]] = []
        seen_ids = set()
        for chunk in model_chunks:
            for name, module in chunk.named_modules():
                module_id = id(module)
                if module_id in seen_ids:
                    continue
                seen_ids.add(module_id)
                results.append((name, module))
        return results

    @staticmethod
    def _is_routed_expert_param_name(param_name: str) -> bool:
        return ".experts." in param_name and ".shared_experts." not in param_name

    @staticmethod
    def _is_moe_router_param_name(param_name: str) -> bool:
        return ".router." in param_name

    @staticmethod
    def _is_moe_module(module: torch.nn.Module) -> bool:
        return hasattr(module, "token_dispatcher") and hasattr(module, "experts")

    @staticmethod
    def _layer_prefix_from_name(name: str) -> Optional[str]:
        normalized = CDCOptimizerV2._normalize_module_name(name)
        match = re.match(r"(.*?\.layers\.\d+)(?:\.|$)", normalized)
        if match is None:
            return None
        return match.group(1)

    @staticmethod
    def _parse_layer_index(name: str) -> Optional[int]:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match is None:
            return None
        return int(match.group(1))

    def _build_layer_prefix_to_global_idx(self) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        for name, module in self._iter_named_modules_unique(self.model_chunks):
            layer_number = getattr(module, "layer_number", None)
            if layer_number is None:
                continue
            prefix = self._layer_prefix_from_name(name)
            if prefix is None:
                continue
            global_idx = int(layer_number) - 1
            previous = mapping.get(prefix)
            if previous is not None and previous != global_idx:
                raise ValueError(
                    f"Layer prefix {prefix!r} maps to both global layer {previous} and {global_idx}."
                )
            mapping[prefix] = global_idx
        return mapping

    def _global_layer_index_for_param(self, param_name: str) -> Optional[int]:
        prefix = self._layer_prefix_from_name(param_name)
        if prefix is None:
            return None
        if prefix in self._layer_prefix_to_global_idx:
            return self._layer_prefix_to_global_idx[prefix]

        pp_world_size = self._safe_world_size(self.pp_group)
        if pp_world_size > 1:
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
            raise ValueError(
                f"MoE module {normalized_name!r} is missing layer_number; "
                "cannot build a globally unique expert key."
            )
        match = re.match(r"(.*?\.layers\.)(\d+)(\..*)?$", normalized_name)
        if match is None:
            raise ValueError(
                f"MoE module {normalized_name!r} does not contain a '.layers.<idx>' segment."
            )
        prefix, _, suffix = match.groups()
        return f"{prefix}{int(layer_number) - 1}{suffix or ''}"

    def _iter_global_moe_modules(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        for local_name, module in self._iter_named_modules_unique(self.model_chunks):
            if not self._is_moe_module(module):
                continue
            yield self._globalize_moe_module_name(local_name, module), module

    def _build_global_moe_module_index(self) -> Dict[str, torch.nn.Module]:
        module_index: Dict[str, torch.nn.Module] = {}
        for global_key, module in self._iter_global_moe_modules():
            if global_key in module_index:
                raise ValueError(f"Duplicate global MoE module key detected: {global_key!r}")
            module_index[global_key] = module
        return module_index

    def _collect_router_named_params(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        return [
            (name, param)
            for name, param in named_params
            if self._is_moe_router_param_name(name)
        ]

    def _filter_named_params_for_cdc(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        if self.moe_param_mode == "all":
            return list(named_params)

        tracked: List[Tuple[str, torch.nn.Parameter]] = []
        excluded_expert_tensors = 0
        excluded_expert_numel = 0
        dedicated_router_tensors = 0
        dedicated_router_numel = 0

        for name, param in named_params:
            if self._is_routed_expert_param_name(name):
                excluded_expert_tensors += 1
                excluded_expert_numel += param.numel()
                continue
            if self.enable_moe_router_refresh and self._is_moe_router_param_name(name):
                dedicated_router_tensors += 1
                dedicated_router_numel += param.numel()
                continue
            tracked.append((name, param))

        if not tracked:
            raise ValueError(
                f"cdc_moe_param_mode={self.moe_param_mode} excluded every trainable parameter."
            )

        if self.verbose and excluded_expert_tensors > 0:
            tracked_numel = sum(param.numel() for _, param in tracked)
            print_rank_0(
                f"[CDC-V2] MoE {self.moe_param_mode} excludes routed experts from dense queue: "
                f"{excluded_expert_tensors} tensors/{excluded_expert_numel} params; "
                f"router_dedicated={dedicated_router_tensors} tensors/{dedicated_router_numel} params; "
                f"dense_tracked={len(tracked)} tensors/{tracked_numel} params."
            )
        return tracked

    def _build_local_expert_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        grouped: Dict[str, List[torch.nn.Parameter]] = {}
        for module_key, module in self._iter_global_moe_modules():
            experts_module = getattr(module, "experts", None)
            local_experts = getattr(experts_module, "local_experts", None)
            if local_experts is None:
                trainable_params = [
                    param
                    for param in experts_module.parameters(recurse=True)
                    if getattr(param, "requires_grad", False)
                ]
                if trainable_params:
                    raise NotImplementedError(
                        "CDCOptimizerV2 dense-expert-hybrid expects local_experts.* layout. "
                        "Grouped expert weights need an explicit ExpertKey mapping first."
                    )
                continue

            for local_expert_idx, expert in enumerate(local_experts):
                params = [
                    param
                    for param in expert.parameters()
                    if getattr(param, "requires_grad", False)
                ]
                if not params:
                    continue
                expert_key = f"{module_key}.experts.local_experts.{local_expert_idx}"
                if expert_key in grouped:
                    raise ValueError(f"Duplicate global expert key detected: {expert_key!r}")
                grouped[expert_key] = params
        return grouped

    def _extract_expert_group_metadata(
        self, group_name: str
    ) -> Tuple[str, Optional[int], Optional[int]]:
        normalized = self._normalize_module_name(group_name)
        match = re.match(r"(.*)\.experts\.local_experts\.(\d+)$", normalized)
        if match is None:
            return normalized, None, None
        module_key = match.group(1)
        local_expert_idx = int(match.group(2))
        layer_idx = self._parse_layer_index(module_key)
        return module_key, layer_idx, local_expert_idx

    def _expert_group_sort_key(self, group_name: str) -> Tuple[int, int, str]:
        _, layer_idx, expert_idx = self._extract_expert_group_metadata(group_name)
        return (
            layer_idx if layer_idx is not None else 10**9,
            expert_idx if expert_idx is not None else 10**9,
            self._normalize_module_name(group_name),
        )

    def _lookup_moe_module(self, module_key: str) -> Optional[torch.nn.Module]:
        return self._moe_module_index.get(module_key)

    @staticmethod
    def _is_embedding_param_name(param_name: str) -> bool:
        if ".layers." in param_name:
            return False
        keys = ("embedding", "word_embeddings", "position_embeddings", "tok_embeddings")
        return any(key in param_name for key in keys)

    def _layer_to_decoder_shard(
        self, layer_idx: int, num_layers: int, decoder_shards: int
    ) -> int:
        if decoder_shards <= 0:
            return 0
        if self.shard_pattern == "stride":
            return layer_idx % decoder_shards
        base = num_layers // decoder_shards
        remainder = num_layers % decoder_shards
        boundary = 0
        for shard_idx in range(decoder_shards):
            shard_size = base + (1 if shard_idx < remainder else 0)
            next_boundary = boundary + shard_size
            if boundary <= layer_idx < next_boundary:
                return shard_idx
            boundary = next_boundary
        return decoder_shards - 1

    def _assign_param_to_dense_shard(
        self,
        *,
        name: str,
        num_layers: int,
        embedding_shards: int,
        decoder_shards: int,
    ) -> int:
        if "output_layer" in name or "lm_head" in name:
            return 0 if self.tie_embeddings else 1
        if self._is_embedding_param_name(name):
            return 0

        layer_idx = self._global_layer_index_for_param(name)
        if layer_idx is not None and num_layers > 0 and decoder_shards > 0:
            return embedding_shards + self._layer_to_decoder_shard(
                layer_idx, num_layers, decoder_shards
            )
        return max(self.num_shards - 1, 0)

    # ------------------------------------------------------------------
    # State initialization
    # ------------------------------------------------------------------

    def _init_diloco_state(self) -> None:
        self.original_snapshot = [
            self._clone_param_for_outer_state(param).requires_grad_(self.outer_lr != 1.0)
            for param in self.tracked_model_param_list
        ]
        if self.outer_lr != 1.0 and self.original_snapshot:
            self.outer_optimizer = SGD(
                self.original_snapshot, lr=self.outer_lr, momentum=0.9, nesterov=True
            )
        self._all_reduce_flattened(self.original_snapshot, communication_dtype=self.outer_comm_dtype)

    def _init_streaming_state(self) -> None:
        num_layers = int(getattr(self.args, "num_layers", 0) or getattr(self.args, "decoder_num_layers", 0))
        if num_layers <= 0:
            local_layers = {
                self._global_layer_index_for_param(name)
                for name, _ in self._named_model_param_list
                if self._global_layer_index_for_param(name) is not None
            }
            num_layers = max(local_layers) + 1 if local_layers else 0

        embedding_shards = 1 if self.tie_embeddings else 2
        if self.num_shards < embedding_shards:
            raise ValueError(
                f"cdc_num_shards ({self.num_shards}) must be >= {embedding_shards} "
                f"for tie_embeddings={self.tie_embeddings}."
            )
        decoder_shards = max(self.num_shards - embedding_shards, 0)
        if decoder_shards == 0 and num_layers > 0:
            raise ValueError(
                f"cdc_num_shards ({self.num_shards}) is insufficient for num_layers={num_layers}."
            )

        dense_slot_to_params: Dict[int, List[torch.nn.Parameter]] = {
            slot_id: [] for slot_id in range(self.num_shards)
        }
        for name, param in self._tracked_named_model_params:
            slot_id = self._assign_param_to_dense_shard(
                name=name,
                num_layers=num_layers,
                embedding_shards=embedding_shards,
                decoder_shards=decoder_shards,
            )
            dense_slot_to_params[slot_id].append(param)

        for slot_id in range(self.num_shards):
            tracker = self._build_tracker(
                slot_id=slot_id,
                display_name=f"dense-shard-{slot_id}",
                kind="dense-shard",
                param_refs=dense_slot_to_params[slot_id],
                comm_dtype=self.outer_comm_dtype,
            )
            self.dense_trackers[slot_id] = tracker
            if self.verbose:
                print_rank_0(
                    f"[CDC-V2] Dense shard initialized: slot={slot_id}, "
                    f"local_tensors={len(tracker.param_refs)}, global_numel={tracker.global_num_params}"
                )

        if self.enable_moe_router_refresh:
            router_params = [param for _, param in self._router_named_model_params]
            tracker = self._build_tracker(
                slot_id=0,
                display_name="moe-router",
                kind="router",
                param_refs=router_params,
                comm_dtype=torch.float32,
            )
            self.router_trackers[0] = tracker
            if self.verbose:
                print_rank_0(
                    f"[CDC-V2] Router tracker initialized: "
                    f"local_tensors={len(tracker.param_refs)}, global_numel={tracker.global_num_params}, "
                    "comm_dtype=torch.float32"
                )

        if self.enable_moe_expert_refresh:
            local_expert_groups = self._build_local_expert_groups()
            global_expert_keys = self._gather_unique_strings_across_group(
                sorted(local_expert_groups.keys(), key=self._expert_group_sort_key),
                self.model_parallel_group,
            )
            if not global_expert_keys:
                raise ValueError("dense-expert-hybrid requested expert refresh, but no experts were found.")
            for slot_id, expert_key in enumerate(global_expert_keys):
                module_key, _, local_expert_idx = self._extract_expert_group_metadata(expert_key)
                tracker = self._build_tracker(
                    slot_id=slot_id,
                    display_name=expert_key,
                    kind="expert-group",
                    param_refs=local_expert_groups.get(expert_key, []),
                    comm_dtype=self.outer_comm_dtype,
                    moe_module_key=module_key,
                    local_expert_idx=local_expert_idx,
                )
                self.expert_trackers[slot_id] = tracker
                if self.verbose:
                    print_rank_0(
                        f"[CDC-V2] Expert group initialized: slot={slot_id}, name={expert_key}, "
                        f"local_tensors={len(tracker.param_refs)}, global_numel={tracker.global_num_params}"
                    )

    def _build_tracker(
        self,
        *,
        slot_id: int,
        display_name: str,
        kind: str,
        param_refs: List[torch.nn.Parameter],
        comm_dtype: torch.dtype,
        moe_module_key: Optional[str] = None,
        local_expert_idx: Optional[int] = None,
    ) -> CDCTracker:
        params = [self._clone_param_for_outer_state(param) for param in param_refs]
        staged = [self._clone_param_for_outer_state(param) for param in param_refs]

        outer_optimizer = None
        if self.outer_lr != 1.0 and params:
            for tensor in params:
                tensor.requires_grad_(True)
            outer_optimizer = SGD(params, lr=self.outer_lr, momentum=0.9, nesterov=True)

        local_numel = sum(param.numel() for param in param_refs)
        global_numel = self._all_reduce_scalar_sum(local_numel, self.tp_group)
        global_numel = self._all_reduce_scalar_sum(global_numel, self.pp_group)

        tracker = CDCTracker(
            slot_id=slot_id,
            display_name=display_name,
            kind=kind,
            param_refs=param_refs,
            params=params,
            staged_params=staged,
            global_num_params=int(global_numel),
            comm_dtype=comm_dtype,
            outer_optimizer=outer_optimizer,
            moe_module_key=moe_module_key,
            local_expert_idx=local_expert_idx,
        )
        self._all_reduce_flattened(tracker.params, communication_dtype=tracker.comm_dtype)
        return tracker

    # ------------------------------------------------------------------
    # Step and algorithm branches
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self):
        update_successful, grad_norm, num_zeros_in_grad = self.inner_optimizer.step()
        update_successful = self._sync_update_success(update_successful)
        if not update_successful:
            return update_successful, grad_norm, num_zeros_in_grad

        self.step_count += 1
        if self.track_expert_token_load:
            self._update_moe_expert_token_load_stats()

        if self.algorithm == "diloco":
            if self.step_count % self.sync_interval == 0:
                self._run_diloco_sync()
        else:
            self._run_streaming_like_step()

        return update_successful, grad_norm, num_zeros_in_grad

    def _sync_update_success(self, success: bool) -> bool:
        value = 1 if success else 0
        value = int(self._all_reduce_scalar_min(value, self.model_parallel_group))
        return bool(value)

    def _run_diloco_sync(self) -> None:
        if self.verbose:
            print_rank_0(f"[CDC-V2] Step {self.step_count}: Starting DiLoCo sync.")
        start_time = time.time()

        sync_grads: List[torch.Tensor] = []
        for snapshot, local_param in zip(self.original_snapshot, self.tracked_model_param_list):
            delta = snapshot.data.clone()
            delta.sub_(local_param.data.to(device=snapshot.device, dtype=snapshot.dtype))
            sync_grads.append(delta)

        self._all_reduce_flattened(sync_grads, communication_dtype=self.outer_comm_dtype)

        if self.outer_optimizer is not None:
            for snapshot, grad in zip(self.original_snapshot, sync_grads):
                if snapshot.grad is None:
                    snapshot.grad = torch.zeros_like(snapshot.data)
                snapshot.grad.copy_(grad)
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad(set_to_none=True)
        else:
            for snapshot, avg_delta in zip(self.original_snapshot, sync_grads):
                snapshot.data.sub_(avg_delta)

        for snapshot, local_param in zip(self.original_snapshot, self.tracked_model_param_list):
            self._copy_tensor_data(local_param.data, snapshot.data)

        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()

        if self.verbose:
            duration = time.time() - start_time
            print_rank_0(f"[CDC-V2] Step {self.step_count}: DiLoCo sync completed in {duration:.4f}s.")

    def _run_streaming_like_step(self) -> None:
        self._complete_due_trackers(self.dense_trackers)
        self._complete_due_trackers(self.router_trackers)
        self._complete_due_expert_trackers_batched()

        if self._should_run_blocking_full_sync():
            self._run_blocking_full_sync()
            return

        if self.step_count % self.sync_interval == 0:
            dense_slot = self._select_next_dense_slot()
            dense_tracker = self.dense_trackers[dense_slot]
            if dense_tracker.has_global_params() and not dense_tracker.is_in_flight(self.step_count):
                self._initiate_tracker_sync(dense_tracker)
            if self.enable_moe_router_refresh:
                router = self.router_trackers[0]
                if router.has_global_params() and not router.is_in_flight(self.step_count):
                    self._initiate_tracker_sync(router)
                elif self.verbose and router.is_in_flight(self.step_count):
                    print_rank_0(
                        f"[CDC-V2] Step {self.step_count}: Skip router sync; "
                        f"previous receive due at step {router.next_receive_step}."
                    )

        if self._should_open_expert_slot():
            selected = self._select_next_expert_slots()
            if selected:
                self.expert_sync_event_count += 1
                event_id = self.expert_sync_event_count
                for slot_id in selected:
                    self._initiate_tracker_sync(
                        self.expert_trackers[slot_id], expert_event_id=event_id, defer_completion=True
                    )
                if self.delay == 0:
                    self._complete_tracker_batch(
                        [self.expert_trackers[slot_id] for slot_id in selected],
                        tracker_kind="expert-group",
                    )
                    for slot_id in selected:
                        self.expert_trackers[slot_id].next_receive_step = 0

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _select_next_dense_slot(self) -> int:
        if self.algorithm == "streaming":
            slot_id = self._next_nonempty_tracker_slot(self.dense_trackers, self.next_dense_slot)
            if slot_id is None:
                raise RuntimeError("No non-empty dense shard exists globally.")
            self.next_dense_slot = (slot_id + 1) % len(self.dense_trackers)
            return slot_id

        horizon = self.dc_N * self.sync_interval
        best_slot: Optional[int] = None
        best_score = float("-inf")
        for slot_id in sorted(self.dense_trackers):
            tracker = self.dense_trackers[slot_id]
            if not tracker.has_global_params() or tracker.is_in_flight(self.step_count):
                continue
            if tracker.sent_at_step == 0:
                return slot_id
            if horizon > 0 and (self.step_count - tracker.sent_at_step) >= horizon:
                return slot_id

            age_steps = max(self.step_count - tracker.sent_at_step, 1)
            last_interval = max(tracker.sent_at_step - tracker.old_sent_at_step, 1)
            rms = math.sqrt(max(tracker.last_score, 0.0) / max(tracker.global_num_params, 1))
            score = 1e8 * rms * (age_steps / last_interval)
            if score > best_score:
                best_score = score
                best_slot = slot_id

        if best_slot is None:
            raise RuntimeError("No selectable dense shard found.")
        return best_slot

    def _next_nonempty_tracker_slot(
        self, trackers: Dict[int, CDCTracker], start_slot: int
    ) -> Optional[int]:
        if not trackers:
            return None
        count = len(trackers)
        for offset in range(count):
            slot_id = (start_slot + offset) % count
            tracker = trackers[slot_id]
            if tracker.has_global_params() and not tracker.is_in_flight(self.step_count):
                return slot_id
        return None

    def _should_open_expert_slot(self) -> bool:
        if not self.enable_moe_expert_refresh or not self.expert_trackers:
            return False
        shifted_step = self.step_count - self.expert_sync_offset
        return shifted_step >= 0 and shifted_step % self.expert_sync_interval == 0

    def _expert_is_unsent(self, tracker: CDCTracker) -> bool:
        return tracker.sent_at_expert_event <= 0

    def _expert_age_slots(self, tracker: CDCTracker, upcoming_event: int) -> int:
        if tracker.sent_at_expert_event <= 0:
            return 0
        return upcoming_event - tracker.sent_at_expert_event

    def _expert_age_steps(self, tracker: CDCTracker) -> int:
        if tracker.sent_at_step <= 0:
            return 0
        return self.step_count - tracker.sent_at_step

    def _expert_is_slot_stale(self, tracker: CDCTracker, upcoming_event: int) -> bool:
        if self.expert_max_age_slots <= 0 or self._expert_is_unsent(tracker):
            return False
        return self._expert_age_slots(tracker, upcoming_event) >= self.expert_max_age_slots

    def _expert_is_step_stale(self, tracker: CDCTracker) -> bool:
        if self.expert_max_staleness <= 0 or self._expert_is_unsent(tracker):
            return False
        return self._expert_age_steps(tracker) >= self.expert_max_staleness

    def _expert_is_min_age_blocked(self, tracker: CDCTracker, upcoming_event: int) -> bool:
        if self.expert_min_age_slots <= 0 or self._expert_is_unsent(tracker):
            return False
        return self._expert_age_slots(tracker, upcoming_event) < self.expert_min_age_slots

    def _expert_is_available(self, tracker: CDCTracker) -> bool:
        return tracker.has_global_params() and not tracker.is_in_flight(self.step_count)

    def _collect_round_robin_expert_slots(
        self,
        *,
        limit: int,
        selected: Optional[set] = None,
        require_unsent: bool,
        eligible_slots: Optional[set] = None,
    ) -> List[int]:
        if not self.expert_trackers or limit <= 0:
            return []
        selected = selected or set()
        count = len(self.expert_trackers)
        chosen: List[int] = []
        for offset in range(count):
            slot_id = (self.next_expert_slot + offset) % count
            if slot_id in selected or slot_id in chosen:
                continue
            if eligible_slots is not None and slot_id not in eligible_slots:
                continue
            tracker = self.expert_trackers[slot_id]
            if not self._expert_is_available(tracker):
                continue
            if require_unsent and not self._expert_is_unsent(tracker):
                continue
            chosen.append(slot_id)
            if len(chosen) >= limit:
                break
        if chosen:
            self.next_expert_slot = (chosen[-1] + 1) % count
        return chosen

    def _select_next_expert_slots(self) -> List[int]:
        all_slots = sorted(self.expert_trackers)
        candidate_slots = [
            slot_id
            for slot_id in all_slots
            if self._expert_is_available(self.expert_trackers[slot_id])
        ]
        if not candidate_slots:
            return []

        max_select = min(self.expert_topk, len(candidate_slots))
        selected: List[int] = []
        selected_set = set()
        selected_reasons: Dict[int, str] = {}
        upcoming_event = self.expert_sync_event_count + 1
        selection_score_all = self._build_expert_score_map(candidate_slots)
        selection_score_used: Dict[int, float] = {}

        unsent = self._collect_round_robin_expert_slots(
            limit=max_select, selected=selected_set, require_unsent=True
        )
        for slot_id in unsent:
            selected.append(slot_id)
            selected_set.add(slot_id)
            selected_reasons[slot_id] = "mandatory_unsent"

        remaining = max_select - len(selected)
        if remaining > 0:
            stale_candidates: List[Tuple[int, int, int, bool, bool]] = []
            for slot_id in candidate_slots:
                if slot_id in selected_set:
                    continue
                tracker = self.expert_trackers[slot_id]
                age_slots = self._expert_age_slots(tracker, upcoming_event)
                age_steps = self._expert_age_steps(tracker)
                slot_stale = self._expert_is_slot_stale(tracker, upcoming_event)
                step_stale = self._expert_is_step_stale(tracker)
                if slot_stale or step_stale:
                    stale_candidates.append((age_slots, age_steps, slot_id, slot_stale, step_stale))
            stale_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
            for _, _, slot_id, slot_stale, step_stale in stale_candidates[:remaining]:
                selected.append(slot_id)
                selected_set.add(slot_id)
                if slot_stale and step_stale:
                    selected_reasons[slot_id] = "stale_slot+step"
                elif slot_stale:
                    selected_reasons[slot_id] = "stale_slot"
                else:
                    selected_reasons[slot_id] = "stale_step"

        remaining = max_select - len(selected)
        if remaining > 0:
            regular_candidates = {
                slot_id
                for slot_id in candidate_slots
                if slot_id not in selected_set
                and not self._expert_is_min_age_blocked(self.expert_trackers[slot_id], upcoming_event)
            }
            if self.expert_selection == "round_robin":
                rr_slots = self._collect_round_robin_expert_slots(
                    limit=remaining,
                    selected=selected_set,
                    require_unsent=False,
                    eligible_slots=regular_candidates,
                )
                for slot_id in rr_slots:
                    selected_reasons[slot_id] = "round_robin"
                selected.extend(rr_slots)
            else:
                score_candidates = sorted(regular_candidates)
                selection_score_used = self._build_expert_score_map(score_candidates)
                ranked = sorted(
                    score_candidates,
                    key=lambda slot_id: (-selection_score_used.get(slot_id, 0.0), slot_id),
                )
                for rank, slot_id in enumerate(ranked[:remaining]):
                    selected_reasons[slot_id] = f"score_rank_{rank}"
                selected.extend(ranked[:remaining])

        self._log_expert_selection_debug(
            all_slots=all_slots,
            candidate_slots=candidate_slots,
            upcoming_event=upcoming_event,
            selected=selected,
            selected_reasons=selected_reasons,
            selection_score_all=selection_score_all,
            selection_score_used=selection_score_used,
        )
        return selected

    def _build_expert_score_map(self, candidate_slots: List[int]) -> Dict[int, float]:
        if not candidate_slots:
            return {}

        update_scores: Dict[int, float] = {}
        token_scores_local: Dict[int, float] = {}
        for slot_id in candidate_slots:
            tracker = self.expert_trackers[slot_id]
            denom = max(tracker.global_num_params, 1)
            update_scores[slot_id] = math.sqrt(max(tracker.last_score, 0.0) / denom)
            token_scores_local[slot_id] = max(float(tracker.token_load_accum), 0.0)

        if self.expert_score_mode == "update_norm":
            return update_scores

        ordered_slots = sorted(candidate_slots)
        token_values = [token_scores_local[slot_id] for slot_id in ordered_slots]
        token_values = self._reduce_expert_slot_vector(token_values)
        token_scores = {
            slot_id: value for slot_id, value in zip(ordered_slots, token_values)
        }

        if self.expert_score_mode == "token_load":
            return token_scores

        update_max = max(update_scores.values(), default=0.0)
        token_max = max(token_scores.values(), default=0.0)
        mixed_scores: Dict[int, float] = {}
        for slot_id in candidate_slots:
            update_component = update_scores[slot_id] / update_max if update_max > 0.0 else 0.0
            token_component = token_scores[slot_id] / token_max if token_max > 0.0 else 0.0
            mixed_scores[slot_id] = 0.5 * update_component + 0.5 * token_component
        return mixed_scores

    def _should_run_blocking_full_sync(self) -> bool:
        if self.blocking_full_sync_steps <= 0 or self.step_count <= 0:
            return False
        return self.step_count % self.blocking_full_sync_steps == 0

    # ------------------------------------------------------------------
    # Token-load stats and selection logging
    # ------------------------------------------------------------------

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
                    cols = len(raw[0])
                    reduced = [0.0 for _ in range(cols)]
                    for row in raw:
                        for idx, item in enumerate(row):
                            reduced[idx] += float(item)
                    return reduced
                return [float(v) for v in raw]
        return []

    def _collect_current_expert_token_loads(self) -> Dict[int, float]:
        if not self.track_expert_token_load:
            return {}

        module_load_cache: Dict[str, List[float]] = {}
        loads: Dict[int, float] = {}
        for slot_id, tracker in self.expert_trackers.items():
            load_value = 0.0
            if tracker.moe_module_key is not None and tracker.local_expert_idx is not None:
                if tracker.moe_module_key not in module_load_cache:
                    module = self._lookup_moe_module(tracker.moe_module_key)
                    expert_loads: List[float] = []
                    token_dispatcher = None
                    if module is not None:
                        token_dispatcher = getattr(module, "token_dispatcher", None)
                        load_tensor = None
                        if token_dispatcher is not None:
                            local_map = getattr(token_dispatcher, "local_map", None)
                            if torch.is_tensor(local_map):
                                load_tensor = local_map.sum(dim=0)
                            if load_tensor is None:
                                load_tensor = getattr(
                                    token_dispatcher, "num_global_tokens_per_local_expert", None
                                )
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
                        print_rank_0(
                            "[CDC-V2][TokenLoadSource] "
                            f"module_key={tracker.moe_module_key}, "
                            f"dispatcher={type(token_dispatcher).__name__ if token_dispatcher is not None else 'None'}, "
                            f"load_len={len(expert_loads)}, load_sum={sum(expert_loads) if expert_loads else 0.0:.1f}, "
                            f"load_preview={expert_loads[:8]}"
                        )
                        self._token_load_source_debug_printed = True

                    module_load_cache[tracker.moe_module_key] = expert_loads

                expert_loads = module_load_cache.get(tracker.moe_module_key, [])
                expert_idx = int(tracker.local_expert_idx)
                if 0 <= expert_idx < len(expert_loads):
                    load_value = float(expert_loads[expert_idx])

            loads[slot_id] = load_value
        return loads

    def _update_moe_expert_token_load_stats(self) -> None:
        current_loads = self._collect_current_expert_token_loads()
        for slot_id, load_value in current_loads.items():
            tracker = self.expert_trackers[slot_id]
            tracker.last_token_load = float(load_value)
            tracker.token_load_accum += float(load_value)

    def _collect_expert_selection_debug_state(
        self,
        *,
        all_slots: List[int],
        candidate_slots: List[int],
        upcoming_event: int,
        selection_score_all: Dict[int, float],
        selection_score_used: Dict[int, float],
    ) -> Dict[int, Dict[str, Any]]:
        global_token_accum: Dict[int, float] = {}
        if all_slots:
            ordered_slots = sorted(all_slots)
            local_values = [
                max(float(self.expert_trackers[slot_id].token_load_accum), 0.0)
                for slot_id in ordered_slots
            ]
            reduced_values = self._reduce_expert_slot_vector(local_values)
            global_token_accum = {
                slot_id: float(value)
                for slot_id, value in zip(ordered_slots, reduced_values)
            }

        candidate_set = set(candidate_slots)
        rows: Dict[int, Dict[str, Any]] = {}
        for slot_id in sorted(all_slots):
            tracker = self.expert_trackers[slot_id]
            denom = max(tracker.global_num_params, 1)
            update_norm = math.sqrt(max(float(tracker.last_score), 0.0) / denom)
            rows[slot_id] = {
                "display_name": tracker.display_name,
                "candidate": slot_id in candidate_set,
                "available": self._expert_is_available(tracker),
                "in_flight": tracker.is_in_flight(self.step_count),
                "unsent": self._expert_is_unsent(tracker),
                "sent_at_step": tracker.sent_at_step,
                "sent_at_expert_event": tracker.sent_at_expert_event,
                "next_receive_step": tracker.next_receive_step,
                "age_slots": self._expert_age_slots(tracker, upcoming_event),
                "age_steps": self._expert_age_steps(tracker),
                "slot_stale": self._expert_is_slot_stale(tracker, upcoming_event),
                "step_stale": self._expert_is_step_stale(tracker),
                "min_age_blocked": self._expert_is_min_age_blocked(tracker, upcoming_event),
                "last_score_norm_sq": float(tracker.last_score),
                "update_norm": update_norm,
                "last_token_load": float(tracker.last_token_load),
                "token_load_accum_local": float(tracker.token_load_accum),
                "token_load_accum_global": float(global_token_accum.get(slot_id, 0.0)),
                "selection_score_all": selection_score_all.get(slot_id),
                "selection_score_used": selection_score_used.get(slot_id),
            }
        return rows

    def _log_expert_selection_debug(
        self,
        *,
        all_slots: List[int],
        candidate_slots: List[int],
        upcoming_event: int,
        selected: List[int],
        selected_reasons: Dict[int, str],
        selection_score_all: Dict[int, float],
        selection_score_used: Dict[int, float],
    ) -> None:
        if not self.verbose or not self.expert_trackers:
            return

        rows = self._collect_expert_selection_debug_state(
            all_slots=all_slots,
            candidate_slots=candidate_slots,
            upcoming_event=upcoming_event,
            selection_score_all=selection_score_all,
            selection_score_used=selection_score_used,
        )
        selected_order = {slot_id: rank for rank, slot_id in enumerate(selected)}
        print_rank_0(
            f"[CDC-V2][ExpertSelect] Step {self.step_count}: "
            f"event={upcoming_event}, mode={self.expert_selection}, "
            f"score_mode={self.expert_score_mode}, topk={self.expert_topk}, "
            f"available_count={len(candidate_slots)}, min_age_slots={self.expert_min_age_slots}, "
            f"selected={selected}"
        )
        for slot_id in sorted(all_slots):
            row = rows[slot_id]
            score_all = row["selection_score_all"]
            score_used = row["selection_score_used"]
            score_all_str = f"{score_all:.6e}" if score_all is not None else "n/a"
            score_used_str = f"{score_used:.6e}" if score_used is not None else "n/a"
            selected_rank = selected_order.get(slot_id, -1)
            print_rank_0(
                "[CDC-V2][ExpertSelect] "
                f"idx={slot_id} "
                f"name={row['display_name']} "
                f"candidate={int(row['candidate'])} "
                f"available={int(row['available'])} "
                f"in_flight={int(row['in_flight'])} "
                f"unsent={int(row['unsent'])} "
                f"selected={int(slot_id in selected_order)} "
                f"selected_rank={selected_rank} "
                f"reason={selected_reasons.get(slot_id, '-')} "
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

    # ------------------------------------------------------------------
    # Sync lifecycle
    # ------------------------------------------------------------------

    def _initiate_tracker_sync(
        self,
        tracker: CDCTracker,
        *,
        expert_event_id: Optional[int] = None,
        defer_completion: bool = False,
    ) -> None:
        if self.verbose:
            print_rank_0(
                f"[CDC-V2] Step {self.step_count}: Initiating sync for {tracker.kind} "
                f"{tracker.display_name} (Size: {self._tracker_payload_size_mb(tracker):.2f} MB)."
            )

        tracker.sync_start_time = time.time()
        for local_param, staged_param in zip(tracker.param_refs, tracker.staged_params):
            self._copy_tensor_data(staged_param, local_param.data)

        tracker.old_sent_at_step = tracker.sent_at_step
        tracker.sent_at_step = self.step_count
        tracker.next_receive_step = self.step_count + self.delay
        if tracker.kind == "expert-group":
            if expert_event_id is not None:
                tracker.sent_at_expert_event = int(expert_event_id)
            tracker.token_load_accum = 0.0

        if self.delay == 0 and not defer_completion:
            self._complete_tracker_sync(tracker)
            tracker.next_receive_step = 0

    def _complete_due_trackers(self, trackers: Dict[int, CDCTracker]) -> None:
        for tracker in trackers.values():
            if tracker.next_receive_step > 0 and self.step_count >= tracker.next_receive_step:
                self._complete_tracker_sync(tracker)
                tracker.next_receive_step = 0

    def _complete_due_expert_trackers_batched(self) -> None:
        if not self.expert_trackers:
            return
        event_to_trackers: Dict[int, List[CDCTracker]] = {}
        for tracker in self.expert_trackers.values():
            if tracker.next_receive_step <= 0 or self.step_count < tracker.next_receive_step:
                continue
            event_id = tracker.sent_at_expert_event
            if event_id <= 0:
                event_id = -(tracker.slot_id + 1)
            event_to_trackers.setdefault(event_id, []).append(tracker)

        for event_id in sorted(event_to_trackers):
            trackers = sorted(event_to_trackers[event_id], key=lambda item: item.slot_id)
            if len(trackers) == 1:
                self._complete_tracker_sync(trackers[0], reload_main_params=False)
                if self.mixed_precision:
                    self.inner_optimizer.reload_model_params()
            else:
                self._complete_tracker_batch(trackers, tracker_kind="expert-group")
            for tracker in trackers:
                tracker.next_receive_step = 0

    def _complete_tracker_sync(
        self,
        tracker: CDCTracker,
        *,
        reload_main_params: bool = True,
        use_algorithm_specific_update: bool = True,
        force_full_copy: bool = False,
    ) -> None:
        sync_grads = self._build_sync_grads(tracker.params, tracker.staged_params)
        self._all_reduce_flattened(sync_grads, communication_dtype=tracker.comm_dtype)
        tracker.last_score = self._compute_global_score(sync_grads)

        if self.verbose:
            duration = time.time() - tracker.sync_start_time
            print_rank_0(
                f"[CDC-V2] Step {self.step_count}: Completed sync for {tracker.kind} "
                f"{tracker.display_name} in {duration:.4f}s. Score (Norm^2): {tracker.last_score:.4e}"
            )

        self._apply_outer_update(tracker, sync_grads)

        if (
            use_algorithm_specific_update
            and self.algorithm == "dc"
            and tracker.kind == "dense-shard"
        ):
            self._apply_dc_update(tracker)
        else:
            self._apply_streaming_receive(tracker, force_full_copy=force_full_copy)

        if self.mixed_precision and reload_main_params:
            self.inner_optimizer.reload_model_params()

    def _complete_tracker_batch(
        self, trackers: List[CDCTracker], *, tracker_kind: str
    ) -> None:
        if not trackers:
            return

        batched_sync_grads: List[torch.Tensor] = []
        per_tracker_grads: List[List[torch.Tensor]] = []
        for tracker in trackers:
            sync_grads = self._build_sync_grads(tracker.params, tracker.staged_params)
            per_tracker_grads.append(sync_grads)
            batched_sync_grads.extend(sync_grads)

        comm_dtype = trackers[0].comm_dtype
        if any(tracker.comm_dtype != comm_dtype for tracker in trackers):
            raise ValueError("Batched tracker sync requires identical communication dtype.")
        self._all_reduce_flattened(batched_sync_grads, communication_dtype=comm_dtype)

        for tracker, sync_grads in zip(trackers, per_tracker_grads):
            tracker.last_score = self._compute_global_score(sync_grads)
            if self.verbose:
                duration = time.time() - tracker.sync_start_time
                print_rank_0(
                    f"[CDC-V2] Step {self.step_count}: Completed sync for {tracker_kind} "
                    f"{tracker.display_name} in {duration:.4f}s. Score (Norm^2): {tracker.last_score:.4e}"
                )
            self._apply_outer_update(tracker, sync_grads)
            self._apply_streaming_receive(tracker, force_full_copy=False)

        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()

    @staticmethod
    def _build_sync_grads(
        global_params: List[torch.Tensor], staged_params: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        sync_grads: List[torch.Tensor] = []
        for global_param, staged_param in zip(global_params, staged_params):
            grad = global_param.data.clone()
            grad.sub_(staged_param.data)
            sync_grads.append(grad)
        return sync_grads

    def _compute_global_score(self, sync_grads: List[torch.Tensor]) -> float:
        local_norm_sq = 0.0
        for grad in sync_grads:
            local_norm_sq += float(grad.float().pow(2).sum().item())
        local_norm_sq = self._all_reduce_scalar_sum(local_norm_sq, self.tp_group)
        local_norm_sq = self._all_reduce_scalar_sum(local_norm_sq, self.pp_group)
        return float(local_norm_sq)

    def _apply_outer_update(self, tracker: CDCTracker, sync_grads: List[torch.Tensor]) -> None:
        if tracker.outer_optimizer is not None:
            for global_param, grad in zip(tracker.params, sync_grads):
                if global_param.grad is None:
                    global_param.grad = torch.zeros_like(global_param.data)
                global_param.grad.copy_(grad)
            tracker.outer_optimizer.step()
            tracker.outer_optimizer.zero_grad(set_to_none=True)
            return
        for global_param, avg_delta in zip(tracker.params, sync_grads):
            global_param.data.sub_(avg_delta)

    def _apply_streaming_receive(self, tracker: CDCTracker, *, force_full_copy: bool) -> None:
        if force_full_copy:
            for local_param, global_param in zip(tracker.param_refs, tracker.params):
                self._copy_tensor_data(local_param.data, global_param.data)
            return

        alpha = (
            self.router_alpha
            if tracker.kind == "router" and self.router_alpha >= 0.0
            else self.streaming_alpha
        )
        for local_param, global_param in zip(tracker.param_refs, tracker.params):
            global_data = global_param.data.to(device=local_param.device, dtype=torch.float32)
            blended = local_param.data.to(torch.float32).mul(alpha).add_(
                global_data, alpha=1.0 - alpha
            )
            local_param.data.copy_(blended.to(dtype=local_param.dtype))

    def _apply_dc_update(self, tracker: CDCTracker) -> None:
        tau = max(self.step_count - tracker.sent_at_step, 1)
        eps = 1e-8

        if self.dc_type == "legacy":
            corrected: List[torch.Tensor] = []
            for staged_param, local_param, global_param in zip(
                tracker.staged_params, tracker.param_refs, tracker.params
            ):
                local_data = (
                    local_param.detach().to("cpu", dtype=torch.float32)
                    if self.offload_outer_opt
                    else local_param.data.to(torch.float32)
                )
                staged_data = staged_param.data.to(torch.float32)
                global_data = global_param.data.to(torch.float32)
                g1 = staged_data - local_data
                d = global_data - staged_data
                numerator = self.dc_lambda * torch.norm(g1)
                correction = (g1 * g1 * d) / 4e-4
                denominator = torch.norm(correction)
                dynamic_lambda = numerator / (denominator + eps)
                corrected.append(g1 + dynamic_lambda * correction)
            for local_param, global_param, correction in zip(
                tracker.param_refs, tracker.params, corrected
            ):
                target = global_param.data.to(torch.float32) - correction
                local_param.data.copy_(target.to(dtype=local_param.dtype, device=local_param.device))
            return

        def _sqsum(tensor: torch.Tensor) -> float:
            return float(tensor.float().pow(2).sum().item())

        def _terms(
            staged_param: torch.Tensor, local_param: torch.nn.Parameter, global_param: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            local_theta = (
                local_param.detach().to("cpu", dtype=torch.float32)
                if self.offload_outer_opt
                else local_param.data.to(torch.float32)
            )
            staged_theta = staged_param.data.to(torch.float32)
            global_theta = global_param.data.to(torch.float32)
            d = global_theta - staged_theta
            u = (staged_theta - local_theta) / tau
            c = (u * u) * d
            return u, c, global_theta

        if self.dc_lambda_scope == "shard":
            reduce_tp, reduce_pp = True, True
        elif self.dc_lambda_scope == "tensor":
            reduce_tp, reduce_pp = True, False
        else:
            reduce_tp, reduce_pp = False, False

        if self.dc_lambda_scope in {"shard", "local"}:
            local_u2 = 0.0
            local_c2 = 0.0
            terms = []
            for staged_param, local_param, global_param in zip(
                tracker.staged_params, tracker.param_refs, tracker.params
            ):
                u, c, global_theta = _terms(staged_param, local_param, global_param)
                terms.append((u, c, global_theta, local_param))
                local_u2 += _sqsum(u)
                local_c2 += _sqsum(c)
            u2 = self._reduce_norm_scalar(local_u2, reduce_tp=reduce_tp, reduce_pp=reduce_pp)
            c2 = self._reduce_norm_scalar(local_c2, reduce_tp=reduce_tp, reduce_pp=reduce_pp)
            lam = self.dc_lambda * math.sqrt(max(u2, 0.0)) / (math.sqrt(max(c2, 0.0)) + eps)
            lam = float(min(lam, self.dc_lambda_max))
            for u, c, global_theta, local_param in terms:
                compensated = global_theta - tau * (u + lam * c)
                local_param.data.copy_(compensated.to(dtype=local_param.dtype, device=local_param.device))
            return

        for staged_param, local_param, global_param in zip(
            tracker.staged_params, tracker.param_refs, tracker.params
        ):
            u, c, global_theta = _terms(staged_param, local_param, global_param)
            u2 = self._reduce_norm_scalar(_sqsum(u), reduce_tp=True, reduce_pp=False)
            c2 = self._reduce_norm_scalar(_sqsum(c), reduce_tp=True, reduce_pp=False)
            lam = self.dc_lambda * math.sqrt(max(u2, 0.0)) / (math.sqrt(max(c2, 0.0)) + eps)
            lam = float(min(lam, self.dc_lambda_max))
            compensated = global_theta - tau * (u + lam * c)
            local_param.data.copy_(compensated.to(dtype=local_param.dtype, device=local_param.device))

    def _reduce_norm_scalar(self, value: float, *, reduce_tp: bool, reduce_pp: bool) -> float:
        total = float(value)
        if reduce_tp:
            total = self._all_reduce_scalar_sum(total, self.tp_group)
        if reduce_pp:
            total = self._all_reduce_scalar_sum(total, self.pp_group)
        return float(total)

    @torch.no_grad()
    def _run_blocking_full_sync(self) -> None:
        all_trackers = [
            tracker
            for tracker_map in (self.dense_trackers, self.router_trackers, self.expert_trackers)
            for tracker in tracker_map.values()
            if tracker.has_global_params()
        ]
        if not all_trackers:
            return

        canceled = self._cancel_pending_syncs()
        if self.expert_trackers:
            self.expert_sync_event_count += 1
            event_id = self.expert_sync_event_count
        else:
            event_id = None

        if self.verbose:
            kind_counts = {
                "dense": len([t for t in self.dense_trackers.values() if t.has_global_params()]),
                "router": len([t for t in self.router_trackers.values() if t.has_global_params()]),
                "expert": len([t for t in self.expert_trackers.values() if t.has_global_params()]),
            }
            print_rank_0(
                f"[CDC-V2] Step {self.step_count}: Starting blocking full sync "
                f"(dense={kind_counts['dense']}, router={kind_counts['router']}, "
                f"expert={kind_counts['expert']}, canceled_inflight={canceled})."
            )
        start_time = time.time()

        kind_order = {"dense-shard": 0, "router": 1, "expert-group": 2}
        for tracker in sorted(all_trackers, key=lambda item: (kind_order[item.kind], item.slot_id)):
            self._stage_tracker_for_blocking_sync(tracker, expert_event_id=event_id)
            self._complete_tracker_sync(
                tracker,
                reload_main_params=False,
                use_algorithm_specific_update=False,
                force_full_copy=True,
            )

        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()
        if self.verbose:
            print_rank_0(
                f"[CDC-V2] Step {self.step_count}: Blocking full sync completed in "
                f"{time.time() - start_time:.4f}s."
            )

    def _stage_tracker_for_blocking_sync(
        self, tracker: CDCTracker, *, expert_event_id: Optional[int]
    ) -> None:
        tracker.sync_start_time = time.time()
        for local_param, staged_param in zip(tracker.param_refs, tracker.staged_params):
            self._copy_tensor_data(staged_param, local_param.data)
        tracker.old_sent_at_step = tracker.sent_at_step
        tracker.sent_at_step = self.step_count
        tracker.next_receive_step = 0
        if tracker.kind == "expert-group":
            if expert_event_id is not None:
                tracker.sent_at_expert_event = int(expert_event_id)
            tracker.token_load_accum = 0.0

    def _cancel_pending_syncs(self) -> int:
        canceled = 0
        for tracker_map in (self.dense_trackers, self.router_trackers, self.expert_trackers):
            for tracker in tracker_map.values():
                if tracker.next_receive_step > 0:
                    tracker.next_receive_step = 0
                    tracker.sync_start_time = 0.0
                    canceled += 1
        return canceled

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _build_cdc_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "version": 1,
            "algorithm": self.algorithm,
            "step_count": self.step_count,
        }
        if self.algorithm == "diloco":
            state["diloco"] = {
                "snapshot": [self._clone_tensor_to_cpu(tensor) for tensor in self.original_snapshot],
                "outer_optimizer": (
                    self._optimizer_state_to_cpu(self.outer_optimizer.state_dict())
                    if self.outer_optimizer is not None
                    else None
                ),
            }
        else:
            state["streaming_layout_version"] = self.STREAMING_LAYOUT_VERSION
            state["streaming"] = {
                "next_dense_slot": self.next_dense_slot,
                "next_expert_slot": self.next_expert_slot,
                "expert_sync_event_count": self.expert_sync_event_count,
                "dense_trackers": self._serialize_tracker_map(self.dense_trackers),
                "router_trackers": self._serialize_tracker_map(self.router_trackers),
                "expert_trackers": self._serialize_tracker_map(self.expert_trackers),
            }
        return state

    def _serialize_tracker_map(self, trackers: Dict[int, CDCTracker]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for slot_id in sorted(trackers):
            tracker = trackers[slot_id]
            save_staged = tracker.next_receive_step > self.step_count
            entry: Dict[str, Any] = {
                "slot_id": tracker.slot_id,
                "display_name": tracker.display_name,
                "kind": tracker.kind,
                "params": [self._clone_tensor_to_cpu(tensor) for tensor in tracker.params],
                "staged_params": (
                    [self._clone_tensor_to_cpu(tensor) for tensor in tracker.staged_params]
                    if save_staged
                    else None
                ),
                "global_num_params": tracker.global_num_params,
                "sent_at_step": tracker.sent_at_step,
                "old_sent_at_step": tracker.old_sent_at_step,
                "next_receive_step": tracker.next_receive_step,
                "last_score": tracker.last_score,
                "last_token_load": tracker.last_token_load,
                "token_load_accum": tracker.token_load_accum,
                "sent_at_expert_event": tracker.sent_at_expert_event,
                "moe_module_key": tracker.moe_module_key,
                "local_expert_idx": tracker.local_expert_idx,
                "outer_optimizer": (
                    self._optimizer_state_to_cpu(tracker.outer_optimizer.state_dict())
                    if tracker.outer_optimizer is not None
                    else None
                ),
            }
            entries.append(entry)
        return entries

    def _load_cdc_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        if int(state.get("version", 0)) != 1:
            raise ValueError("CDCOptimizerV2 only loads v2 CDC state version 1.")
        checkpoint_algorithm = str(state.get("algorithm", "")).lower()
        if checkpoint_algorithm != self.algorithm:
            raise ValueError(
                f"Checkpoint algorithm {checkpoint_algorithm} does not match runtime {self.algorithm}."
            )
        self.step_count = int(state.get("step_count", 0))

        if self.algorithm == "diloco":
            self._load_diloco_state(state.get("diloco"))
            return

        if int(state.get("streaming_layout_version", 0)) != self.STREAMING_LAYOUT_VERSION:
            raise ValueError("CDCOptimizerV2 refuses to load mismatched streaming layout version.")
        streaming = state.get("streaming")
        if not streaming:
            raise ValueError("CDCOptimizerV2 streaming checkpoint is missing streaming state.")

        self.next_dense_slot = int(streaming.get("next_dense_slot", self.next_dense_slot))
        self.next_expert_slot = int(streaming.get("next_expert_slot", self.next_expert_slot))
        self.expert_sync_event_count = int(
            streaming.get("expert_sync_event_count", self.expert_sync_event_count)
        )
        self._load_tracker_map(self.dense_trackers, streaming.get("dense_trackers", []))
        self._load_tracker_map(self.router_trackers, streaming.get("router_trackers", []))
        self._load_tracker_map(self.expert_trackers, streaming.get("expert_trackers", []))

    def _load_diloco_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            raise ValueError("CDCOptimizerV2 DiLoCo checkpoint is missing diloco state.")
        snapshot = state.get("snapshot", [])
        if len(snapshot) != len(self.original_snapshot):
            raise ValueError("Mismatch in DiLoCo snapshot length.")
        for target, saved in zip(self.original_snapshot, snapshot):
            self._copy_tensor_data(target, saved)
        outer_state = state.get("outer_optimizer")
        if self.outer_optimizer is not None and outer_state is not None:
            self.outer_optimizer.load_state_dict(outer_state)
            device = self.original_snapshot[0].device if self.original_snapshot else torch.device("cpu")
            self._move_optimizer_state_to_device(self.outer_optimizer, device)

    def _load_tracker_map(
        self, trackers: Dict[int, CDCTracker], entries: List[Dict[str, Any]]
    ) -> None:
        if len(entries or []) != len(trackers):
            raise ValueError("Mismatch in CDC tracker count while loading v2 state.")
        for entry in entries or []:
            slot_id = int(entry["slot_id"])
            if slot_id not in trackers:
                raise ValueError(f"Unknown CDC tracker slot {slot_id} in checkpoint.")
            tracker = trackers[slot_id]
            if entry.get("display_name") != tracker.display_name or entry.get("kind") != tracker.kind:
                raise ValueError(
                    f"CDC tracker layout mismatch for slot {slot_id}: "
                    f"checkpoint=({entry.get('kind')}, {entry.get('display_name')}), "
                    f"runtime=({tracker.kind}, {tracker.display_name})."
                )
            self._copy_tensor_list(tracker.params, entry.get("params", []))
            saved_staged = entry.get("staged_params")
            if saved_staged is not None:
                self._copy_tensor_list(tracker.staged_params, saved_staged)
            tracker.global_num_params = int(entry.get("global_num_params", tracker.global_num_params))
            tracker.sent_at_step = int(entry.get("sent_at_step", tracker.sent_at_step))
            tracker.old_sent_at_step = int(entry.get("old_sent_at_step", tracker.old_sent_at_step))
            tracker.next_receive_step = int(
                entry.get("next_receive_step", tracker.next_receive_step)
            )
            tracker.last_score = float(entry.get("last_score", tracker.last_score))
            tracker.last_token_load = float(entry.get("last_token_load", tracker.last_token_load))
            tracker.token_load_accum = float(entry.get("token_load_accum", tracker.token_load_accum))
            tracker.sent_at_expert_event = int(
                entry.get("sent_at_expert_event", tracker.sent_at_expert_event)
            )
            outer_state = entry.get("outer_optimizer")
            if tracker.outer_optimizer is not None and outer_state is not None:
                tracker.outer_optimizer.load_state_dict(outer_state)
                device = tracker.params[0].device if tracker.params else torch.device("cpu")
                self._move_optimizer_state_to_device(tracker.outer_optimizer, device)

    def _reset_outer_state_from_model(self) -> None:
        self.step_count = 0
        if self.algorithm == "diloco":
            for snapshot, local_param in zip(self.original_snapshot, self.tracked_model_param_list):
                self._copy_tensor_data(snapshot, local_param.data)
            self._all_reduce_flattened(self.original_snapshot, communication_dtype=self.outer_comm_dtype)
            if self.outer_optimizer is not None:
                self.outer_optimizer.state.clear()
            return

        self.next_dense_slot = 0
        self.next_expert_slot = 0
        self.expert_sync_event_count = 0
        for tracker_map in (self.dense_trackers, self.router_trackers, self.expert_trackers):
            for tracker in tracker_map.values():
                for outer_tensor, local_param in zip(tracker.params, tracker.param_refs):
                    self._copy_tensor_data(outer_tensor, local_param.data)
                for staged_tensor, local_param in zip(tracker.staged_params, tracker.param_refs):
                    self._copy_tensor_data(staged_tensor, local_param.data)
                tracker.sent_at_step = 0
                tracker.old_sent_at_step = 0
                tracker.next_receive_step = 0
                tracker.last_score = 0.0
                tracker.last_token_load = 0.0
                tracker.token_load_accum = 0.0
                tracker.sent_at_expert_event = 0
                if tracker.outer_optimizer is not None:
                    tracker.outer_optimizer.state.clear()
                self._all_reduce_flattened(tracker.params, communication_dtype=tracker.comm_dtype)

    # ------------------------------------------------------------------
    # Communication and tensor utilities
    # ------------------------------------------------------------------

    def _clone_param_for_outer_state(self, param: torch.nn.Parameter) -> torch.Tensor:
        target_device = torch.device("cpu") if self.offload_outer_opt else param.device
        return param.detach().to(device=target_device, dtype=self.outer_state_dtype, copy=True)

    def _tracker_payload_size_mb(self, tracker: CDCTracker) -> float:
        total_bytes = sum(param.numel() * param.element_size() for param in tracker.param_refs)
        return total_bytes / (1024 * 1024)

    @staticmethod
    def _copy_tensor_data(target_tensor: torch.Tensor, saved_tensor: torch.Tensor) -> None:
        with torch.no_grad():
            target_tensor.copy_(saved_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype))

    def _copy_tensor_list(self, target_list: List[torch.Tensor], saved_list: List[torch.Tensor]) -> None:
        if len(target_list) != len(saved_list):
            raise ValueError("Mismatch in tensor list length while restoring CDC state.")
        for target, saved in zip(target_list, saved_list):
            self._copy_tensor_data(target, saved)

    @staticmethod
    def _clone_tensor_to_cpu(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        with torch.no_grad():
            return tensor.detach().to(device="cpu").clone()

    @staticmethod
    def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)

    def _optimizer_state_to_cpu(self, optimizer_state: Dict[str, Any]) -> Dict[str, Any]:
        cpu_state = {"state": {}, "param_groups": deepcopy(optimizer_state.get("param_groups", []))}
        for key, value in optimizer_state.get("state", {}).items():
            cpu_entry = {}
            for inner_key, inner_value in value.items():
                if torch.is_tensor(inner_value):
                    cpu_entry[inner_key] = self._clone_tensor_to_cpu(inner_value)
                else:
                    cpu_entry[inner_key] = deepcopy(inner_value)
            cpu_state["state"][key] = cpu_entry
        return cpu_state

    @staticmethod
    def _safe_world_size(group) -> int:
        if group is None:
            return 1
        try:
            return int(dist.get_world_size(group=group))
        except Exception:
            return 1

    def _all_reduce_scalar_sum(self, value: float, group) -> float:
        if self._safe_world_size(group) <= 1:
            return float(value)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return float(tensor.item())

    def _all_reduce_scalar_min(self, value: float, group) -> float:
        if self._safe_world_size(group) <= 1:
            return float(value)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=group)
        return float(tensor.item())

    def _all_reduce_vector_sum(self, values: List[float], group) -> List[float]:
        if not values:
            return []
        if self._safe_world_size(group) <= 1:
            return [float(value) for value in values]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(values, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return [float(value) for value in tensor.cpu().tolist()]

    def _reduce_expert_slot_vector(self, values: List[float]) -> List[float]:
        values = self._all_reduce_vector_sum(values, self.cdc_group)
        values = self._all_reduce_vector_sum(values, self.tp_group)
        values = self._all_reduce_vector_sum(values, self.pp_group)
        return values

    def _gather_unique_strings_across_group(self, local_strings: List[str], group) -> List[str]:
        if self._safe_world_size(group) <= 1:
            return sorted(set(local_strings), key=self._expert_group_sort_key)
        gathered: List[List[str]] = [None for _ in range(self._safe_world_size(group))]  # type: ignore[list-item]
        dist.all_gather_object(gathered, list(local_strings), group=group)
        merged = set()
        for names in gathered:
            if names:
                merged.update(names)
        return sorted(merged, key=self._expert_group_sort_key)

    @torch.no_grad()
    def _all_reduce_flattened(
        self, tensors: List[torch.Tensor], communication_dtype: Optional[torch.dtype] = None
    ) -> None:
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        if not tensors:
            return

        start_time = time.time()
        total_bytes = 0
        dtype_groups: Dict[torch.dtype, List[torch.Tensor]] = {}
        for tensor in tensors:
            dtype_groups.setdefault(tensor.dtype, []).append(tensor)

        for original_dtype, grouped_tensors in dtype_groups.items():
            flat_tensor = _flatten_dense_tensors(grouped_tensors)
            comm_dtype = communication_dtype if communication_dtype is not None else original_dtype

            if flat_tensor.device.type == "cpu":
                comm_tensor = flat_tensor.to(device="cuda", dtype=comm_dtype)
                total_bytes += comm_tensor.numel() * comm_tensor.element_size()
                dist.all_reduce(comm_tensor, group=self.cdc_group)
                comm_tensor.div_(dist.get_world_size(group=self.cdc_group))
                flat_tensor.copy_(comm_tensor.to(device=flat_tensor.device, dtype=original_dtype))
                del comm_tensor
            else:
                comm_tensor = flat_tensor.to(dtype=comm_dtype) if comm_dtype != original_dtype else flat_tensor
                total_bytes += comm_tensor.numel() * comm_tensor.element_size()
                dist.all_reduce(comm_tensor, group=self.cdc_group)
                comm_tensor.div_(dist.get_world_size(group=self.cdc_group))
                if comm_tensor is not flat_tensor:
                    flat_tensor.copy_(comm_tensor.to(dtype=original_dtype))

            for tensor, synced in zip(
                grouped_tensors, _unflatten_dense_tensors(flat_tensor, grouped_tensors)
            ):
                tensor.copy_(synced)

        if self.verbose:
            duration = time.time() - start_time
            size_mb = total_bytes / (1024 * 1024)
            bandwidth = size_mb / duration if duration > 0 else 0.0
            print_rank_0(
                f"[CDC-V2] Communication: {size_mb:.2f} MB in {duration:.4f}s "
                f"({bandwidth:.2f} MB/s)"
            )
