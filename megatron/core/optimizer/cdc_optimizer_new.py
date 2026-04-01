import math
import re
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.optim import SGD

from megatron.core import mpu
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.transformer.module import MegatronModule
from megatron.training.global_vars import get_args
from megatron.training.utils import print_rank_0


@dataclass
class TrackerSlot:
    """Single global sync slot.

    A slot always has a globally consistent meaning:
    - dense slots use global shard ids 0..N-1
    - expert slots use globally consistent expert-group ids

    A rank may own zero local parameters for a slot, which is normal under PP.
    """

    slot_id: int
    display_name: str
    param_refs: List[torch.nn.Parameter]
    params: List[torch.Tensor]
    staged_params: List[torch.Tensor]
    global_num_params: int
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

    def has_global_params(self) -> bool:
        return self.global_num_params > 0

    def has_local_params(self) -> bool:
        return bool(self.param_refs)

    def is_in_flight(self, step_count: int) -> bool:
        return self.next_receive_step > step_count


@dataclass
class DiLoCoBranchState:
    snapshot: List[torch.Tensor] = field(default_factory=list)
    outer_optimizer: Optional[torch.optim.Optimizer] = None


@dataclass
class StreamingBranchState:
    dense_trackers: Dict[int, TrackerSlot] = field(default_factory=dict)
    expert_trackers: Dict[int, TrackerSlot] = field(default_factory=dict)
    next_dense_slot: int = 0
    next_expert_slot: int = 0
    expert_sync_event_count: int = 0


class CDCOptimizerNew(MegatronOptimizer):
    """Cleaner CDC optimizer rewrite.

    Design goals:
    - Keep one self-contained optimizer class.
    - Keep the three algorithm branches explicit: diloco / streaming / dc.
    - Keep common pieces small and obvious: parameter discovery, slot tracking,
      communication helpers, checkpoint helpers.
    - Preserve current algorithm behavior where possible, but remove the old
      weight-decay debias path to keep DC understandable.

    This class is intentionally not wired into the training stack yet.
    """

    def __init__(
        self,
        inner_optimizer: MegatronOptimizer,
        model_chunks: Optional[List[MegatronModule]] = None,
    ):
        self.inner_optimizer = inner_optimizer
        self.model_chunks = (
            model_chunks if model_chunks is not None else getattr(inner_optimizer, "model_chunks", None)
        )
        assert self.model_chunks is not None, "model_chunks must be provided to CDCOptimizerNew."
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
        self.shard_pattern = str(args.cdc_shard_pattern).lower()
        self.offload_outer_opt = bool(args.cdc_offload_outer_opt)
        self.verbose = bool(args.cdc_verbose)

        self.dc_lambda = float(args.cdc_dc_lambda)
        self.dc_lambda_max = float(args.cdc_dc_lambda_max)
        self.dc_lambda_scope = str(args.cdc_dc_lambda_scope).lower()
        self.dc_type = str(args.cdc_dc_type).lower()
        self.dc_N = int(args.cdc_dc_N)

        self.tie_embeddings = not args.untie_embeddings_and_output_weights
        self.mixed_precision = bool(args.bf16 or args.fp16)

        self.moe_param_mode = str(args.cdc_moe_param_mode).lower()
        self.expert_sync_interval = int(args.cdc_moe_expert_sync_interval)
        self.expert_sync_offset = int(args.cdc_moe_expert_sync_offset)
        self.expert_selection = str(args.cdc_moe_expert_selection).lower()
        self.expert_topk = int(args.cdc_moe_expert_topk)
        self.expert_score_mode = str(args.cdc_moe_expert_score_mode).lower()
        self.expert_max_age_slots = int(args.cdc_moe_expert_max_age_slots)
        self.expert_max_staleness = int(args.cdc_moe_expert_max_staleness)

        self.step_count = 0
        self._cdc_state_loaded = False

        self._named_model_param_list = self._iter_named_trainable_params_unique(self.model_chunks)
        self._local_expert_named_params = self._collect_local_routed_expert_named_params(
            self._named_model_param_list
        )
        self._tracked_named_model_params = self._filter_named_params_for_cdc(
            self._named_model_param_list
        )

        tracked_params = self.tracked_model_param_list
        self.model_param_dtype = tracked_params[0].dtype if tracked_params else torch.float32
        self.outer_state_dtype = torch.float32 if self.mixed_precision else self.model_param_dtype
        self.outer_comm_dtype = (
            self.model_param_dtype if self.mixed_precision else self.outer_state_dtype
        )

        self.enable_moe_expert_refresh = (
            self.moe_param_mode == "dense-expert-hybrid" and self.expert_sync_interval > 0
        )
        self.track_expert_token_load = (
            self.enable_moe_expert_refresh
            and self.expert_selection == "score"
            and self.expert_score_mode in {"token_load", "mixed"}
        )

        self._validate_configuration()

        self._moe_module_index = (
            self._build_moe_module_index() if self.track_expert_token_load else {}
        )
        self._local_grouped_expert_params = self._group_local_routed_expert_named_params()

        self.diloco_state: Optional[DiLoCoBranchState] = None
        self.streaming_state: Optional[StreamingBranchState] = None

        if self.algorithm == "diloco":
            self._init_diloco_branch()
        else:
            self._init_streaming_like_branch()

        if self.verbose:
            refresh_state = "on" if self.enable_moe_expert_refresh else "off"
            print_rank_0(
                "[CDC-New] Initialized "
                f"algorithm={self.algorithm}, sync_interval={self.sync_interval}, "
                f"num_shards={self.num_shards}, delay={self.delay}, "
                f"moe_param_mode={self.moe_param_mode}, expert_refresh={refresh_state}, "
                f"expert_topk={self.expert_topk}, expert_score_mode={self.expert_score_mode}, "
                f"outer_state_dtype={self.outer_state_dtype}, outer_comm_dtype={self.outer_comm_dtype}"
            )

    # ------------------------------------------------------------------
    # Basic delegated optimizer interface
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
        self.inner_optimizer.zero_grad(set_to_none)

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

    def save_parameter_state(self, filename: str):
        self.inner_optimizer.save_parameter_state(filename)

    def load_parameter_state(self, filename: str, *, update_legacy_format: bool = False):
        self.inner_optimizer.load_parameter_state(
            filename, update_legacy_format=update_legacy_format
        )

    @torch.no_grad()
    def prepare_grads(self):
        return self.inner_optimizer.prepare_grads()

    @torch.no_grad()
    def step_with_ready_grads(self):
        raise NotImplementedError(
            "CDCOptimizerNew only supports optimizer.step(). "
            "The split prepare_grads()/step_with_ready_grads() API would bypass CDC outer sync."
        )

    # ------------------------------------------------------------------
    # Public state management
    # ------------------------------------------------------------------

    def reload_model_params(self):
        self.inner_optimizer.reload_model_params()
        if self.step_count == 0 and not self._cdc_state_loaded:
            self._reset_outer_state_from_model()

    def state_dict(self, is_loading: bool = False):
        return {
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "cdc_state": self._build_cdc_state(),
        }

    def load_state_dict(self, state_dict):
        if "inner_optimizer" not in state_dict:
            self._cdc_state_loaded = False
            self.inner_optimizer.load_state_dict(state_dict)
            self._reset_outer_state_from_model()
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
    # Configuration / validation
    # ------------------------------------------------------------------

    def _validate_configuration(self) -> None:
        if self.algorithm not in {"diloco", "streaming", "dc"}:
            raise ValueError(f"Unknown CDC algorithm: {self.algorithm}")
        if self.sync_interval <= 0:
            raise ValueError(f"cdc_sync_interval must be > 0, got {self.sync_interval}")
        if self.num_shards <= 0:
            raise ValueError(f"cdc_num_shards must be > 0, got {self.num_shards}")
        if self.delay < 0:
            raise ValueError(f"cdc_delay must be >= 0, got {self.delay}")
        if not (0.0 <= self.streaming_alpha <= 1.0):
            raise ValueError(
                f"cdc_streaming_alpha must be in [0, 1], got {self.streaming_alpha}"
            )
        if self.shard_pattern not in {"sequential", "stride"}:
            raise ValueError(f"Unknown cdc_shard_pattern: {self.shard_pattern}")
        if self.moe_param_mode not in {"all", "dense-only", "dense-expert-hybrid"}:
            raise ValueError(f"Unknown cdc_moe_param_mode: {self.moe_param_mode}")
        if self.expert_selection not in {"round_robin", "score"}:
            raise ValueError(
                f"Unknown cdc_moe_expert_selection: {self.expert_selection}"
            )
        if self.expert_score_mode not in {"update_norm", "token_load", "mixed"}:
            raise ValueError(
                f"Unknown cdc_moe_expert_score_mode: {self.expert_score_mode}"
            )
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
        if self.dc_type not in {"update", "legacy"}:
            raise ValueError(f"Unknown cdc_dc_type: {self.dc_type}")
        if self.dc_lambda_scope not in {"shard", "tensor", "local"}:
            raise ValueError(f"Unknown cdc_dc_lambda_scope: {self.dc_lambda_scope}")

        if self.moe_param_mode == "dense-expert-hybrid" and self.algorithm != "streaming":
            raise ValueError(
                "cdc_moe_param_mode=dense-expert-hybrid currently supports streaming only."
            )
        if self.enable_moe_expert_refresh and int(self.args.expert_model_parallel_size) != 1:
            raise NotImplementedError(
                "CDCOptimizerNew dense-expert-hybrid currently supports expert_model_parallel_size=1 only."
            )

    # ------------------------------------------------------------------
    # Parameter discovery / grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_named_trainable_params_unique(
        model_chunks: List[MegatronModule],
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
        model_chunks: List[MegatronModule],
    ) -> Iterable[Tuple[str, torch.nn.Module]]:
        seen = set()
        for chunk in model_chunks:
            for name, module in chunk.named_modules():
                module_id = id(module)
                if module_id in seen:
                    continue
                seen.add(module_id)
                yield name, module

    @staticmethod
    def _normalize_module_name(name: str) -> str:
        normalized = name
        while normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        return normalized

    @staticmethod
    def _is_routed_expert_param_name(param_name: str) -> bool:
        return ".experts." in param_name and ".shared_experts." not in param_name

    @staticmethod
    def _routed_expert_group_key(param_name: str) -> Optional[str]:
        match = re.search(r"(.*?\.experts\.local_experts\.\d+)\.", param_name)
        if match is None:
            return None
        return match.group(1)

    def _collect_local_routed_expert_named_params(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        return [
            (name, param)
            for name, param in named_params
            if self._is_routed_expert_param_name(name)
        ]

    def _filter_named_params_for_cdc(
        self, named_params: List[Tuple[str, torch.nn.Parameter]]
    ) -> List[Tuple[str, torch.nn.Parameter]]:
        if self.moe_param_mode == "all":
            return list(named_params)

        tracked: List[Tuple[str, torch.nn.Parameter]] = []
        excluded_tensors = 0
        excluded_numel = 0
        for name, param in named_params:
            if self._is_routed_expert_param_name(name):
                excluded_tensors += 1
                excluded_numel += param.numel()
                continue
            tracked.append((name, param))

        if not tracked:
            raise ValueError(
                f"cdc_moe_param_mode={self.moe_param_mode} excluded every trainable parameter."
            )

        if self.verbose and excluded_tensors > 0:
            tracked_numel = sum(param.numel() for _, param in tracked)
            print_rank_0(
                "[CDC-New] MoE mode filtered routed experts from dense queue: "
                f"excluded_tensors={excluded_tensors}, excluded_params={excluded_numel}, "
                f"tracked_tensors={len(tracked)}, tracked_params={tracked_numel}"
            )

        return tracked

    def _group_local_routed_expert_named_params(
        self,
    ) -> Dict[str, List[Tuple[str, torch.nn.Parameter]]]:
        grouped: Dict[str, List[Tuple[str, torch.nn.Parameter]]] = {}
        for name, param in self._local_expert_named_params:
            group_key = self._routed_expert_group_key(name)
            if group_key is None:
                continue
            grouped.setdefault(group_key, []).append((name, param))
        return grouped

    @staticmethod
    def _parse_layer_index(param_name: str) -> Optional[int]:
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", param_name)
        if match is None:
            return None
        return int(match.group(1))

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
        _, layer_idx, local_expert_idx = self._extract_expert_group_metadata(group_name)
        return (
            layer_idx if layer_idx is not None else 10**9,
            local_expert_idx if local_expert_idx is not None else 10**9,
            self._normalize_module_name(group_name),
        )

    def _gather_unique_strings_across_group(
        self, local_strings: List[str], group
    ) -> List[str]:
        if group is None:
            return sorted(set(local_strings), key=self._expert_group_sort_key)
        try:
            world_size = dist.get_world_size(group=group)
        except Exception:
            world_size = 1
        if world_size <= 1:
            return sorted(set(local_strings), key=self._expert_group_sort_key)

        gathered: List[List[str]] = [None for _ in range(world_size)]  # type: ignore[list-item]
        dist.all_gather_object(gathered, list(local_strings), group=group)

        merged = set()
        for names in gathered:
            if not names:
                continue
            merged.update(names)
        return sorted(merged, key=self._expert_group_sort_key)

    def _build_moe_module_index(self) -> Dict[str, torch.nn.Module]:
        module_index: Dict[str, torch.nn.Module] = {}
        for name, module in self._iter_named_modules_unique(self.model_chunks):
            normalized_name = self._normalize_module_name(name)
            if not normalized_name:
                continue
            if hasattr(module, "token_dispatcher") and hasattr(module, "experts"):
                module_index[normalized_name] = module
        return module_index

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

    def _assign_param_to_shard(
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

        layer_idx = self._parse_layer_index(name)
        if layer_idx is not None and num_layers > 0 and decoder_shards > 0:
            return embedding_shards + self._layer_to_decoder_shard(
                layer_idx, num_layers, decoder_shards
            )

        return max(self.num_shards - 1, 0)

    # ------------------------------------------------------------------
    # Branch initialization
    # ------------------------------------------------------------------

    def _init_diloco_branch(self) -> None:
        snapshot: List[torch.Tensor] = []
        for param in self.tracked_model_param_list:
            snapshot.append(self._clone_param_for_outer_state(param).requires_grad_(True))

        outer_optimizer = None
        if self.outer_lr != 1.0 and snapshot:
            outer_optimizer = SGD(snapshot, lr=self.outer_lr, momentum=0.9, nesterov=True)

        self.diloco_state = DiLoCoBranchState(snapshot=snapshot, outer_optimizer=outer_optimizer)
        self._all_reduce_flattened(self.diloco_state.snapshot, communication_dtype=self.outer_comm_dtype)

    def _init_streaming_like_branch(self) -> None:
        args = get_args()
        num_layers = getattr(args, "num_layers", None)
        if num_layers is None:
            num_layers = getattr(args, "decoder_num_layers", None)
        if num_layers is None:
            local_layers = {
                self._parse_layer_index(name)
                for name, _ in self._named_model_param_list
                if self._parse_layer_index(name) is not None
            }
            num_layers = (max(local_layers) + 1) if local_layers else 0

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

        dense_slot_to_local_params: Dict[int, List[torch.nn.Parameter]] = {
            slot_id: [] for slot_id in range(self.num_shards)
        }
        for name, param in self._tracked_named_model_params:
            shard_idx = self._assign_param_to_shard(
                name=name,
                num_layers=num_layers,
                embedding_shards=embedding_shards,
                decoder_shards=decoder_shards,
            )
            dense_slot_to_local_params[shard_idx].append(param)

        dense_trackers: Dict[int, TrackerSlot] = {}
        for shard_idx in range(self.num_shards):
            tracker = self._build_tracker(
                slot_id=shard_idx,
                display_name=f"dense-shard-{shard_idx}",
                param_refs=dense_slot_to_local_params.get(shard_idx, []),
            )
            dense_trackers[shard_idx] = tracker
            if self.verbose:
                print_rank_0(
                    "[CDC-New] Dense shard initialized: "
                    f"slot={shard_idx}, local_tensors={len(tracker.param_refs)}, "
                    f"global_numel={tracker.global_num_params}"
                )

        expert_trackers: Dict[int, TrackerSlot] = {}
        if self.enable_moe_expert_refresh:
            global_expert_group_names = self._gather_unique_strings_across_group(
                sorted(self._local_grouped_expert_params.keys(), key=self._expert_group_sort_key),
                self.pp_group,
            )
            if not global_expert_group_names:
                raise ValueError(
                    "dense-expert-hybrid requested expert refresh, but no routed expert groups were discovered."
                )

            if self._local_expert_named_params and not self._local_grouped_expert_params:
                raise NotImplementedError(
                    "CDCOptimizerNew dense-expert-hybrid currently expects SequentialMLP-style "
                    "local_experts.* parameter names. Grouped expert weights are not yet supported."
                )

            for slot_id, expert_group_name in enumerate(global_expert_group_names):
                local_group = self._local_grouped_expert_params.get(expert_group_name, [])
                local_params = [param for _, param in local_group]
                moe_module_key, _, local_expert_idx = self._extract_expert_group_metadata(
                    expert_group_name
                )
                tracker = self._build_tracker(
                    slot_id=slot_id,
                    display_name=expert_group_name,
                    param_refs=local_params,
                    moe_module_key=moe_module_key,
                    local_expert_idx=local_expert_idx,
                )
                expert_trackers[slot_id] = tracker
                if self.verbose:
                    print_rank_0(
                        "[CDC-New] Expert group initialized: "
                        f"slot={slot_id}, name={expert_group_name}, "
                        f"local_tensors={len(tracker.param_refs)}, "
                        f"global_numel={tracker.global_num_params}"
                    )

        self.streaming_state = StreamingBranchState(
            dense_trackers=dense_trackers,
            expert_trackers=expert_trackers,
            next_dense_slot=0,
            next_expert_slot=0,
            expert_sync_event_count=0,
        )

    def _build_tracker(
        self,
        *,
        slot_id: int,
        display_name: str,
        param_refs: List[torch.nn.Parameter],
        moe_module_key: Optional[str] = None,
        local_expert_idx: Optional[int] = None,
    ) -> TrackerSlot:
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

        tracker = TrackerSlot(
            slot_id=slot_id,
            display_name=display_name,
            param_refs=param_refs,
            params=params,
            staged_params=staged,
            global_num_params=int(global_numel),
            outer_optimizer=outer_optimizer,
            moe_module_key=moe_module_key,
            local_expert_idx=local_expert_idx,
        )

        self._all_reduce_flattened(tracker.params, communication_dtype=self.outer_comm_dtype)
        return tracker

    # ------------------------------------------------------------------
    # Step
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
                start_time = time.time()
                self._run_diloco_sync()
                if self.verbose:
                    duration = time.time() - start_time
                    print_rank_0(
                        f"[CDC-New] Step {self.step_count}: DiLoCo sync completed in {duration:.4f}s."
                    )
        else:
            self._run_streaming_like_sync()

        return update_successful, grad_norm, num_zeros_in_grad

    def _sync_update_success(self, success: bool) -> bool:
        if self.model_parallel_group is None:
            return bool(success)
        try:
            world_size = dist.get_world_size(group=self.model_parallel_group)
        except Exception:
            world_size = 1
        if world_size <= 1:
            return bool(success)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        flag = torch.tensor(1 if success else 0, device=device, dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.model_parallel_group)
        return bool(flag.item())

    # ------------------------------------------------------------------
    # DiLoCo branch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_diloco_sync(self) -> None:
        assert self.diloco_state is not None

        sync_grads: List[torch.Tensor] = []
        for snapshot_tensor, model_param in zip(
            self.diloco_state.snapshot, self.tracked_model_param_list
        ):
            if snapshot_tensor.grad is None:
                snapshot_tensor.grad = torch.zeros_like(snapshot_tensor.data)
            model_data = model_param.data
            if model_data.device != snapshot_tensor.device or model_data.dtype != snapshot_tensor.dtype:
                model_data = model_data.to(device=snapshot_tensor.device, dtype=snapshot_tensor.dtype)
            snapshot_tensor.grad.copy_(snapshot_tensor.data - model_data)
            sync_grads.append(snapshot_tensor.grad)

        self._all_reduce_flattened(sync_grads, communication_dtype=self.outer_comm_dtype)

        if self.diloco_state.outer_optimizer is not None:
            self.diloco_state.outer_optimizer.step()
            self.diloco_state.outer_optimizer.zero_grad(set_to_none=True)
        else:
            for snapshot_tensor in self.diloco_state.snapshot:
                snapshot_tensor.data.sub_(snapshot_tensor.grad)
                snapshot_tensor.grad = None

        for snapshot_tensor, model_param in zip(
            self.diloco_state.snapshot, self.tracked_model_param_list
        ):
            model_param.copy_(
                snapshot_tensor.to(device=model_param.device, dtype=model_param.dtype)
            )

        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()

    # ------------------------------------------------------------------
    # Streaming / DC branch
    # ------------------------------------------------------------------

    def _run_streaming_like_sync(self) -> None:
        assert self.streaming_state is not None

        self._complete_due_dense_syncs()
        if self.enable_moe_expert_refresh:
            self._complete_due_expert_syncs_batched()

        if self.step_count % self.sync_interval == 0:
            dense_slot = self._select_next_dense_slot()
            self._initiate_tracker_sync(
                self.streaming_state.dense_trackers[dense_slot],
                tracker_kind="dense-shard",
            )
            return

        if self._should_open_expert_slot():
            expert_slots = self._select_next_expert_slots()
            if not expert_slots:
                return

            event_id = self.streaming_state.expert_sync_event_count + 1
            batch_immediate_completion = self.delay == 0 and len(expert_slots) > 1
            for slot_id in expert_slots:
                self._initiate_tracker_sync(
                    self.streaming_state.expert_trackers[slot_id],
                    tracker_kind="expert-group",
                    expert_event_id=event_id,
                    defer_completion=batch_immediate_completion,
                )
            self.streaming_state.expert_sync_event_count = event_id

            if batch_immediate_completion:
                trackers = [self.streaming_state.expert_trackers[slot_id] for slot_id in expert_slots]
                self._complete_tracker_batch(trackers, tracker_kind="expert-group")
                for tracker in trackers:
                    tracker.next_receive_step = 0

    def _complete_due_dense_syncs(self) -> None:
        assert self.streaming_state is not None
        for tracker in self.streaming_state.dense_trackers.values():
            if tracker.next_receive_step > 0 and self.step_count >= tracker.next_receive_step:
                self._complete_tracker_sync(tracker, tracker_kind="dense-shard")
                tracker.next_receive_step = 0

    def _complete_due_expert_syncs_batched(self) -> None:
        assert self.streaming_state is not None
        if not self.streaming_state.expert_trackers:
            return

        due_event_to_trackers: Dict[int, List[TrackerSlot]] = {}
        for tracker in self.streaming_state.expert_trackers.values():
            if tracker.next_receive_step <= 0 or self.step_count < tracker.next_receive_step:
                continue
            event_id = tracker.sent_at_expert_event
            if event_id <= 0:
                event_id = -(tracker.slot_id + 1)
            due_event_to_trackers.setdefault(event_id, []).append(tracker)

        for event_id in sorted(due_event_to_trackers):
            trackers = sorted(due_event_to_trackers[event_id], key=lambda item: item.slot_id)
            if len(trackers) == 1:
                self._complete_tracker_sync(
                    trackers[0], tracker_kind="expert-group", reload_main_params=False
                )
                if self.mixed_precision:
                    self.inner_optimizer.reload_model_params()
            else:
                self._complete_tracker_batch(trackers, tracker_kind="expert-group")
            for tracker in trackers:
                tracker.next_receive_step = 0

    def _should_open_expert_slot(self) -> bool:
        if not self.enable_moe_expert_refresh:
            return False
        if self.expert_sync_interval <= 0:
            return False
        shifted_step = self.step_count - self.expert_sync_offset
        return shifted_step >= 0 and shifted_step % self.expert_sync_interval == 0

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select_next_dense_slot(self) -> int:
        assert self.streaming_state is not None
        dense_trackers = self.streaming_state.dense_trackers
        total_slots = len(dense_trackers)
        if total_slots <= 0:
            raise RuntimeError("No dense trackers available.")

        if self.algorithm == "streaming":
            for offset in range(total_slots):
                slot_id = (self.streaming_state.next_dense_slot + offset) % total_slots
                tracker = dense_trackers[slot_id]
                if tracker.has_global_params():
                    self.streaming_state.next_dense_slot = (slot_id + 1) % total_slots
                    return slot_id
            raise RuntimeError("No non-empty dense shard exists globally.")

        horizon = self.dc_N * self.sync_interval
        best_slot_id: Optional[int] = None
        best_score = float("-inf")
        for slot_id in range(total_slots):
            tracker = dense_trackers[slot_id]
            if not tracker.has_global_params():
                continue
            if tracker.is_in_flight(self.step_count):
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
                best_slot_id = slot_id

        if best_slot_id is None:
            raise RuntimeError("No selectable dense shard found for DC.")
        return best_slot_id

    def _select_next_expert_slots(self) -> List[int]:
        assert self.streaming_state is not None
        trackers = self.streaming_state.expert_trackers
        if not trackers:
            return []

        candidate_slots = [
            slot_id
            for slot_id, tracker in trackers.items()
            if tracker.has_global_params() and not tracker.is_in_flight(self.step_count)
        ]
        if not candidate_slots:
            return []

        limit = min(self.expert_topk, len(candidate_slots))
        selected: List[int] = []
        selected_set = set()
        upcoming_event = self.streaming_state.expert_sync_event_count + 1

        stale_candidates: List[Tuple[int, int, int]] = []
        for slot_id in candidate_slots:
            tracker = trackers[slot_id]
            sent_event = tracker.sent_at_expert_event
            sent_step = tracker.sent_at_step
            age_slots = (upcoming_event - sent_event) if sent_event > 0 else 0
            age_steps = (self.step_count - sent_step) if sent_step > 0 else 0

            slot_stale = (
                self.expert_max_age_slots > 0 and sent_event > 0 and age_slots >= self.expert_max_age_slots
            )
            step_stale = (
                self.expert_max_staleness > 0
                and sent_step > 0
                and age_steps >= self.expert_max_staleness
            )
            if slot_stale or step_stale:
                stale_candidates.append((age_slots, age_steps, slot_id))

        if stale_candidates:
            stale_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
            for _, _, slot_id in stale_candidates[:limit]:
                selected.append(slot_id)
                selected_set.add(slot_id)

        remaining = limit - len(selected)
        if remaining > 0:
            unsent = self._collect_round_robin_expert_slots(
                limit=remaining, selected=selected_set, require_unsent=True
            )
            for slot_id in unsent:
                selected.append(slot_id)
                selected_set.add(slot_id)

        remaining = limit - len(selected)
        if remaining <= 0:
            return selected

        if self.expert_selection == "round_robin":
            selected.extend(
                self._collect_round_robin_expert_slots(
                    limit=remaining, selected=selected_set, require_unsent=False
                )
            )
            return selected

        score_candidates = [slot_id for slot_id in candidate_slots if slot_id not in selected_set]
        score_map = self._build_expert_score_map(score_candidates)
        ranked = sorted(score_candidates, key=lambda slot_id: (-score_map.get(slot_id, 0.0), slot_id))
        selected.extend(ranked[:remaining])
        return selected

    def _collect_round_robin_expert_slots(
        self,
        *,
        limit: int,
        selected: Optional[set] = None,
        require_unsent: bool,
    ) -> List[int]:
        assert self.streaming_state is not None
        trackers = self.streaming_state.expert_trackers
        if not trackers or limit <= 0:
            return []

        selected = selected or set()
        tracker_count = len(trackers)
        chosen: List[int] = []
        for offset in range(tracker_count):
            slot_id = (self.streaming_state.next_expert_slot + offset) % tracker_count
            if slot_id in selected or slot_id in chosen:
                continue
            tracker = trackers[slot_id]
            if not tracker.has_global_params() or tracker.is_in_flight(self.step_count):
                continue
            if require_unsent and tracker.sent_at_expert_event > 0:
                continue
            chosen.append(slot_id)
            if len(chosen) >= limit:
                break

        if chosen:
            self.streaming_state.next_expert_slot = (chosen[-1] + 1) % tracker_count
        return chosen

    def _build_expert_score_map(self, candidate_slots: List[int]) -> Dict[int, float]:
        assert self.streaming_state is not None
        if not candidate_slots:
            return {}

        update_scores: Dict[int, float] = {}
        token_scores_local: Dict[int, float] = {}
        for slot_id in candidate_slots:
            tracker = self.streaming_state.expert_trackers[slot_id]
            denom = max(tracker.global_num_params, 1)
            update_scores[slot_id] = math.sqrt(max(tracker.last_score, 0.0) / denom)
            token_scores_local[slot_id] = max(float(tracker.token_load_accum), 0.0)

        if self.expert_score_mode == "update_norm":
            return update_scores

        ordered_slots = sorted(candidate_slots)
        token_values = [token_scores_local[slot_id] for slot_id in ordered_slots]
        token_values = self._all_reduce_vector_sum(token_values, self.cdc_group)
        token_values = self._all_reduce_vector_sum(token_values, self.pp_group)
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

    # ------------------------------------------------------------------
    # Token-load stats
    # ------------------------------------------------------------------

    def _collect_current_expert_token_loads(self) -> Dict[int, float]:
        assert self.streaming_state is not None
        if not self.streaming_state.expert_trackers:
            return {}

        module_load_cache: Dict[str, List[float]] = {}
        loads: Dict[int, float] = {}

        for slot_id, tracker in self.streaming_state.expert_trackers.items():
            load_value = 0.0
            if tracker.moe_module_key is not None and tracker.local_expert_idx is not None:
                if tracker.moe_module_key not in module_load_cache:
                    module = self._moe_module_index.get(tracker.moe_module_key)
                    expert_loads: List[float] = []
                    if module is not None:
                        token_dispatcher = getattr(module, "token_dispatcher", None)
                        load_tensor = None
                        if token_dispatcher is not None:
                            load_tensor = getattr(
                                token_dispatcher, "num_global_tokens_per_local_expert", None
                            )
                            if load_tensor is None:
                                load_tensor = getattr(token_dispatcher, "tokens_per_expert", None)

                        if torch.is_tensor(load_tensor):
                            tensor = load_tensor.detach().to(dtype=torch.float32)
                            if tensor.dim() == 1:
                                expert_loads = [float(value) for value in tensor.cpu().tolist()]
                            else:
                                reduced = tensor.reshape(-1, tensor.shape[-1]).sum(dim=0)
                                expert_loads = [float(value) for value in reduced.cpu().tolist()]

                    module_load_cache[tracker.moe_module_key] = expert_loads

                expert_loads = module_load_cache.get(tracker.moe_module_key, [])
                local_expert_idx = int(tracker.local_expert_idx)
                if 0 <= local_expert_idx < len(expert_loads):
                    load_value = float(expert_loads[local_expert_idx])

            loads[slot_id] = load_value
        return loads

    def _update_moe_expert_token_load_stats(self) -> None:
        if not self.track_expert_token_load:
            return
        current_loads = self._collect_current_expert_token_loads()
        if not current_loads:
            return

        assert self.streaming_state is not None
        for slot_id, load_value in current_loads.items():
            tracker = self.streaming_state.expert_trackers[slot_id]
            tracker.last_token_load = float(load_value)
            tracker.token_load_accum += float(load_value)

    # ------------------------------------------------------------------
    # Sync lifecycle
    # ------------------------------------------------------------------

    def _initiate_tracker_sync(
        self,
        tracker: TrackerSlot,
        *,
        tracker_kind: str,
        expert_event_id: Optional[int] = None,
        defer_completion: bool = False,
    ) -> None:
        if self.verbose:
            size_mb = self._tracker_payload_size_mb(tracker)
            print_rank_0(
                f"[CDC-New] Step {self.step_count}: Initiating sync for {tracker_kind} "
                f"{tracker.display_name} (size={size_mb:.2f} MB)."
            )

        tracker.sync_start_time = time.time()
        for local_param, staged_param in zip(tracker.param_refs, tracker.staged_params):
            self._copy_tensor_data(staged_param, local_param.data)

        tracker.old_sent_at_step = tracker.sent_at_step
        tracker.sent_at_step = self.step_count
        tracker.next_receive_step = self.step_count + self.delay

        if tracker_kind == "expert-group":
            if expert_event_id is not None:
                tracker.sent_at_expert_event = int(expert_event_id)
            tracker.token_load_accum = 0.0

        if self.delay == 0 and not defer_completion:
            self._complete_tracker_sync(tracker, tracker_kind=tracker_kind)
            tracker.next_receive_step = 0

    def _complete_tracker_sync(
        self,
        tracker: TrackerSlot,
        *,
        tracker_kind: str,
        reload_main_params: bool = True,
    ) -> None:
        sync_grads = self._build_sync_grads(tracker.params, tracker.staged_params)
        self._all_reduce_flattened(sync_grads, communication_dtype=self.outer_comm_dtype)

        tracker.last_score = self._compute_global_score(sync_grads)
        self._apply_outer_update(tracker, sync_grads)

        if self.algorithm == "dc" and tracker_kind == "dense-shard":
            self._apply_dc_update(tracker)
        else:
            self._apply_alpha_blend(tracker)

        if self.verbose:
            duration = time.time() - tracker.sync_start_time
            print_rank_0(
                f"[CDC-New] Step {self.step_count}: Completed sync for {tracker_kind} "
                f"{tracker.display_name} in {duration:.4f}s. score={tracker.last_score:.4e}"
            )

        if self.mixed_precision and reload_main_params:
            self.inner_optimizer.reload_model_params()

    def _complete_tracker_batch(
        self, trackers: List[TrackerSlot], *, tracker_kind: str
    ) -> None:
        if not trackers:
            return

        batched_sync_grads: List[torch.Tensor] = []
        per_tracker_sync_grads: List[List[torch.Tensor]] = []
        for tracker in trackers:
            sync_grads = self._build_sync_grads(tracker.params, tracker.staged_params)
            per_tracker_sync_grads.append(sync_grads)
            batched_sync_grads.extend(sync_grads)

        self._all_reduce_flattened(batched_sync_grads, communication_dtype=self.outer_comm_dtype)

        for tracker, sync_grads in zip(trackers, per_tracker_sync_grads):
            tracker.last_score = self._compute_global_score(sync_grads)
            self._apply_outer_update(tracker, sync_grads)
            self._apply_alpha_blend(tracker)
            if self.verbose:
                duration = time.time() - tracker.sync_start_time
                print_rank_0(
                    f"[CDC-New] Step {self.step_count}: Completed sync for {tracker_kind} "
                    f"{tracker.display_name} in {duration:.4f}s. score={tracker.last_score:.4e}"
                )

        if self.mixed_precision:
            self.inner_optimizer.reload_model_params()

    @staticmethod
    def _build_sync_grads(
        global_params: List[torch.Tensor], staged_params: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        sync_grads: List[torch.Tensor] = []
        for global_param, staged_param in zip(global_params, staged_params):
            delta = global_param.data.clone()
            delta.sub_(staged_param.data)
            sync_grads.append(delta)
        return sync_grads

    def _compute_global_score(self, sync_grads: List[torch.Tensor]) -> float:
        local_norm_sq = 0.0
        for grad in sync_grads:
            local_norm_sq += float(grad.float().pow(2).sum().item())
        local_norm_sq = self._all_reduce_scalar_sum(local_norm_sq, self.tp_group)
        local_norm_sq = self._all_reduce_scalar_sum(local_norm_sq, self.pp_group)
        return float(local_norm_sq)

    def _apply_outer_update(self, tracker: TrackerSlot, sync_grads: List[torch.Tensor]) -> None:
        if tracker.outer_optimizer is not None:
            for global_param, grad in zip(tracker.params, sync_grads):
                if global_param.grad is None:
                    global_param.grad = torch.zeros_like(global_param.data)
                global_param.grad.copy_(grad)
            tracker.outer_optimizer.step()
            tracker.outer_optimizer.zero_grad(set_to_none=True)
            return

        for global_param, grad in zip(tracker.params, sync_grads):
            global_param.data.sub_(grad)

    def _apply_alpha_blend(self, tracker: TrackerSlot) -> None:
        for local_param, global_param in zip(tracker.param_refs, tracker.params):
            global_data = global_param.data.to(device=local_param.device, dtype=torch.float32)
            blended = (
                local_param.data.to(torch.float32).mul(self.streaming_alpha).add_(
                    global_data, alpha=1.0 - self.streaming_alpha
                )
            )
            local_param.data.copy_(blended.to(dtype=local_param.dtype))

    def _apply_dc_update(self, tracker: TrackerSlot) -> None:
        tau = max(self.step_count - tracker.sent_at_step, 1)
        eps = 1e-8
        lam0 = self.dc_lambda
        lam_max = self.dc_lambda_max
        scope = self.dc_lambda_scope

        if self.dc_type == "legacy":
            g1_terms: List[torch.Tensor] = []
            d_terms: List[torch.Tensor] = []
            for staged_param, local_param, global_param in zip(
                tracker.staged_params, tracker.param_refs, tracker.params
            ):
                if self.offload_outer_opt:
                    local_data = local_param.detach().to("cpu", dtype=torch.float32)
                else:
                    local_data = local_param.data.to(torch.float32)
                global_data = global_param.data.to(torch.float32)
                staged_data = staged_param.data.to(torch.float32)

                g1 = staged_data.sub(local_data)
                d = global_data.sub(staged_data)
                g1_terms.append(g1)
                d_terms.append(d)

            corrected: List[torch.Tensor] = []
            for g1, d in zip(g1_terms, d_terms):
                numerator = lam0 * torch.norm(g1)
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

        def _reduce_sum(value: float, *, across_tp: bool, across_pp: bool) -> float:
            total = float(value)
            if across_tp:
                total = self._all_reduce_scalar_sum(total, self.tp_group)
            if across_pp:
                total = self._all_reduce_scalar_sum(total, self.pp_group)
            return float(total)

        def _dc_terms(
            staged_param: torch.Tensor, local_param: torch.nn.Parameter, global_param: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.offload_outer_opt:
                local_theta = local_param.detach().to("cpu", dtype=torch.float32)
            else:
                local_theta = local_param.data.to(torch.float32)
            staged_theta = staged_param.data.to(torch.float32)
            global_theta = global_param.data.to(torch.float32)
            d = global_theta - staged_theta
            u = (staged_theta - local_theta) / tau
            c = (u * u) * d
            return u, c, global_theta

        if scope == "shard":
            reduce_tp, reduce_pp = True, True
        elif scope == "tensor":
            reduce_tp, reduce_pp = True, False
        else:
            reduce_tp, reduce_pp = False, False

        if scope in {"shard", "local"}:
            local_u2 = 0.0
            local_c2 = 0.0
            for staged_param, local_param, global_param in zip(
                tracker.staged_params, tracker.param_refs, tracker.params
            ):
                u, c, _ = _dc_terms(staged_param, local_param, global_param)
                local_u2 += _sqsum(u)
                local_c2 += _sqsum(c)

            u2 = _reduce_sum(local_u2, across_tp=reduce_tp, across_pp=reduce_pp)
            c2 = _reduce_sum(local_c2, across_tp=reduce_tp, across_pp=reduce_pp)
            lam = lam0 * math.sqrt(max(u2, 0.0)) / (math.sqrt(max(c2, 0.0)) + eps)
            lam = float(min(lam, lam_max))

            for staged_param, local_param, global_param in zip(
                tracker.staged_params, tracker.param_refs, tracker.params
            ):
                u, c, global_theta = _dc_terms(staged_param, local_param, global_param)
                compensated = global_theta - tau * (u + lam * c)
                local_param.data.copy_(
                    compensated.to(dtype=local_param.dtype, device=local_param.device)
                )
            return

        for staged_param, local_param, global_param in zip(
            tracker.staged_params, tracker.param_refs, tracker.params
        ):
            u, c, global_theta = _dc_terms(staged_param, local_param, global_param)
            u2 = _reduce_sum(_sqsum(u), across_tp=True, across_pp=False)
            c2 = _reduce_sum(_sqsum(c), across_tp=True, across_pp=False)
            lam = lam0 * math.sqrt(max(u2, 0.0)) / (math.sqrt(max(c2, 0.0)) + eps)
            lam = float(min(lam, lam_max))
            compensated = global_theta - tau * (u + lam * c)
            local_param.data.copy_(
                compensated.to(dtype=local_param.dtype, device=local_param.device)
            )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _build_cdc_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "algorithm": self.algorithm,
            "step_count": self.step_count,
        }

        if self.algorithm == "diloco":
            assert self.diloco_state is not None
            state["diloco"] = self._serialize_diloco_state()
        else:
            assert self.streaming_state is not None
            state["streaming"] = {
                "next_dense_slot": self.streaming_state.next_dense_slot,
                "next_expert_slot": self.streaming_state.next_expert_slot,
                "expert_sync_event_count": self.streaming_state.expert_sync_event_count,
                "dense_trackers": self._serialize_tracker_map(self.streaming_state.dense_trackers),
                "expert_trackers": self._serialize_tracker_map(self.streaming_state.expert_trackers),
            }
        return state

    def _serialize_diloco_state(self) -> Dict[str, Any]:
        assert self.diloco_state is not None
        state = {
            "snapshot": [self._clone_tensor_to_cpu(tensor) for tensor in self.diloco_state.snapshot],
            "outer_optimizer": None,
        }
        if self.diloco_state.outer_optimizer is not None:
            state["outer_optimizer"] = self._optimizer_state_to_cpu(
                self.diloco_state.outer_optimizer.state_dict()
            )
        return state

    def _serialize_tracker_map(self, trackers: Dict[int, TrackerSlot]) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        for slot_id in sorted(trackers):
            tracker = trackers[slot_id]
            save_staged = tracker.next_receive_step > self.step_count
            entry: Dict[str, Any] = {
                "slot_id": slot_id,
                "display_name": tracker.display_name,
                "params": [self._clone_tensor_to_cpu(tensor) for tensor in tracker.params],
                "staged_params": (
                    [self._clone_tensor_to_cpu(tensor) for tensor in tracker.staged_params]
                    if save_staged
                    else None
                ),
                "sent_at_step": tracker.sent_at_step,
                "old_sent_at_step": tracker.old_sent_at_step,
                "next_receive_step": tracker.next_receive_step,
                "global_num_params": tracker.global_num_params,
                "last_score": tracker.last_score,
                "last_token_load": tracker.last_token_load,
                "token_load_accum": tracker.token_load_accum,
                "sent_at_expert_event": tracker.sent_at_expert_event,
                "moe_module_key": tracker.moe_module_key,
                "local_expert_idx": tracker.local_expert_idx,
                "outer_optimizer": None,
            }
            if tracker.outer_optimizer is not None:
                entry["outer_optimizer"] = self._optimizer_state_to_cpu(
                    tracker.outer_optimizer.state_dict()
                )
            entries.append(entry)
        return entries

    def _load_cdc_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return

        checkpoint_algorithm = str(state.get("algorithm", self.algorithm)).lower()
        if checkpoint_algorithm != self.algorithm:
            raise ValueError(
                f"Checkpoint algorithm {checkpoint_algorithm} does not match runtime algorithm {self.algorithm}."
            )

        self.step_count = int(state.get("step_count", 0))

        if self.algorithm == "diloco":
            self._load_diloco_state(state.get("diloco"))
            return

        streaming_state = state.get("streaming")
        if not streaming_state:
            return

        assert self.streaming_state is not None
        self.streaming_state.next_dense_slot = int(
            streaming_state.get("next_dense_slot", self.streaming_state.next_dense_slot)
        )
        self.streaming_state.next_expert_slot = int(
            streaming_state.get("next_expert_slot", self.streaming_state.next_expert_slot)
        )
        self.streaming_state.expert_sync_event_count = int(
            streaming_state.get(
                "expert_sync_event_count", self.streaming_state.expert_sync_event_count
            )
        )
        self._load_tracker_map(
            self.streaming_state.dense_trackers, streaming_state.get("dense_trackers", [])
        )
        self._load_tracker_map(
            self.streaming_state.expert_trackers, streaming_state.get("expert_trackers", [])
        )

    def _load_diloco_state(self, state: Optional[Dict[str, Any]]) -> None:
        if not state:
            return
        assert self.diloco_state is not None

        snapshot = state.get("snapshot", [])
        if len(snapshot) != len(self.diloco_state.snapshot):
            raise ValueError("Mismatch in DiLoCo snapshot length while restoring CDC state.")
        for target, saved in zip(self.diloco_state.snapshot, snapshot):
            self._copy_tensor_data(target, saved)

        if self.diloco_state.outer_optimizer is not None and state.get("outer_optimizer") is not None:
            self.diloco_state.outer_optimizer.load_state_dict(state["outer_optimizer"])
            device = (
                self.diloco_state.snapshot[0].device
                if self.diloco_state.snapshot
                else torch.device("cpu")
            )
            self._move_optimizer_state_to_device(self.diloco_state.outer_optimizer, device)

    def _load_tracker_map(
        self, trackers: Dict[int, TrackerSlot], state_entries: List[Dict[str, Any]]
    ) -> None:
        for entry in state_entries or []:
            slot_id = int(entry["slot_id"])
            if slot_id not in trackers:
                raise ValueError(f"Unknown CDC tracker slot {slot_id} in checkpoint.")
            tracker = trackers[slot_id]

            self._copy_tensor_list(tracker.params, entry.get("params", []))
            saved_staged = entry.get("staged_params")
            if saved_staged is not None:
                self._copy_tensor_list(tracker.staged_params, saved_staged)

            tracker.sent_at_step = int(entry.get("sent_at_step", tracker.sent_at_step))
            tracker.old_sent_at_step = int(entry.get("old_sent_at_step", tracker.old_sent_at_step))
            tracker.next_receive_step = int(
                entry.get("next_receive_step", tracker.next_receive_step)
            )
            tracker.global_num_params = int(
                entry.get("global_num_params", tracker.global_num_params)
            )
            tracker.last_score = float(entry.get("last_score", tracker.last_score))
            tracker.last_token_load = float(
                entry.get("last_token_load", tracker.last_token_load)
            )
            tracker.token_load_accum = float(
                entry.get("token_load_accum", tracker.token_load_accum)
            )
            tracker.sent_at_expert_event = int(
                entry.get("sent_at_expert_event", tracker.sent_at_expert_event)
            )
            tracker.moe_module_key = entry.get("moe_module_key", tracker.moe_module_key)
            tracker.local_expert_idx = entry.get("local_expert_idx", tracker.local_expert_idx)

            outer_state = entry.get("outer_optimizer")
            if tracker.outer_optimizer is not None and outer_state is not None:
                tracker.outer_optimizer.load_state_dict(outer_state)
                device = tracker.params[0].device if tracker.params else torch.device("cpu")
                self._move_optimizer_state_to_device(tracker.outer_optimizer, device)

    def _reset_outer_state_from_model(self) -> None:
        self.step_count = 0
        if self.algorithm == "diloco":
            assert self.diloco_state is not None
            for target, local_param in zip(self.diloco_state.snapshot, self.tracked_model_param_list):
                self._copy_tensor_data(target, local_param.data)
            self._all_reduce_flattened(
                self.diloco_state.snapshot, communication_dtype=self.outer_comm_dtype
            )
            if self.diloco_state.outer_optimizer is not None:
                self.diloco_state.outer_optimizer.state.clear()
            return

        assert self.streaming_state is not None
        self.streaming_state.next_dense_slot = 0
        self.streaming_state.next_expert_slot = 0
        self.streaming_state.expert_sync_event_count = 0
        self._reset_tracker_map_from_model(self.streaming_state.dense_trackers)
        self._reset_tracker_map_from_model(self.streaming_state.expert_trackers)

    def _reset_tracker_map_from_model(self, trackers: Dict[int, TrackerSlot]) -> None:
        for tracker in trackers.values():
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

            self._all_reduce_flattened(tracker.params, communication_dtype=self.outer_comm_dtype)

    # ------------------------------------------------------------------
    # Communication / utilities
    # ------------------------------------------------------------------

    def _clone_param_for_outer_state(self, param: torch.nn.Parameter) -> torch.Tensor:
        device = torch.device("cpu") if self.offload_outer_opt else param.device
        return param.detach().to(device=device, dtype=self.outer_state_dtype, copy=True)

    def _tracker_payload_size_mb(self, tracker: TrackerSlot) -> float:
        total_bytes = 0
        for param in tracker.param_refs:
            total_bytes += param.numel() * param.element_size()
        return total_bytes / (1024 * 1024)

    @staticmethod
    def _copy_tensor_data(target_tensor: torch.Tensor, saved_tensor: torch.Tensor) -> None:
        with torch.no_grad():
            target_tensor.copy_(
                saved_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            )

    def _copy_tensor_list(
        self, target_list: List[torch.Tensor], saved_list: List[torch.Tensor]
    ) -> None:
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
    def _move_optimizer_state_to_device(
        optimizer: torch.optim.Optimizer, device: torch.device
    ) -> None:
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)

    def _optimizer_state_to_cpu(self, optimizer_state: Dict[str, Any]) -> Dict[str, Any]:
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

    def _all_reduce_scalar_sum(self, value: float, group) -> float:
        if group is None:
            return float(value)
        try:
            world_size = dist.get_world_size(group=group)
        except Exception:
            world_size = 1
        if world_size <= 1:
            return float(value)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return float(tensor.item())

    def _all_reduce_vector_sum(self, values: List[float], group) -> List[float]:
        if not values:
            return []
        if group is None:
            return [float(value) for value in values]
        try:
            world_size = dist.get_world_size(group=group)
        except Exception:
            world_size = 1
        if world_size <= 1:
            return [float(value) for value in values]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor = torch.tensor(values, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return [float(value) for value in tensor.cpu().tolist()]

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
                if comm_dtype != original_dtype:
                    comm_tensor = flat_tensor.to(dtype=comm_dtype)
                else:
                    comm_tensor = flat_tensor
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
                f"[CDC-New] Communication: {size_mb:.2f} MB in {duration:.4f}s "
                f"({bandwidth:.2f} MB/s)"
            )
