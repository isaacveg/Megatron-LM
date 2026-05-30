import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


def _load_cdc_optimizer_class():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "megatron/core/optimizer/cdc_optimizer.py"
    module_name = "_cdc_optimizer_under_test"

    stubs = {
        "megatron.core.optimizer": types.ModuleType("megatron.core.optimizer"),
        "megatron.core.transformer.module": types.ModuleType(
            "megatron.core.transformer.module"
        ),
        "megatron.training": types.ModuleType("megatron.training"),
        "megatron.training.utils": types.ModuleType("megatron.training.utils"),
        "megatron.training.global_vars": types.ModuleType("megatron.training.global_vars"),
    }
    stubs["megatron.core.optimizer"].MegatronOptimizer = object
    stubs["megatron.core.transformer.module"].MegatronModule = torch.nn.Module
    stubs["megatron.training.utils"].print_rank_0 = lambda *args, **kwargs: None
    stubs["megatron.training.global_vars"].get_args = lambda: SimpleNamespace(
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
    )

    saved_modules = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.CDCOptimizer
    finally:
        sys.modules.pop(module_name, None)
        for name, old_module in saved_modules.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


class FakeExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2))


class FakeExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.local_experts = torch.nn.ModuleList([FakeExpert(), FakeExpert()])

    def forward(self, hidden_states, tokens_per_expert, probs=None):
        return hidden_states


class FakeMoE(torch.nn.Module):
    def __init__(self, layer_number, local_expert_indices):
        super().__init__()
        self.layer_number = layer_number
        self.local_expert_indices = local_expert_indices
        self.token_dispatcher = object()
        self.router = torch.nn.Linear(2, 2, bias=False)
        self.experts = FakeExperts()


class FakeLayer(torch.nn.Module):
    def __init__(self, layer_number, local_expert_indices):
        super().__init__()
        self.layer_number = layer_number
        self.mlp = FakeMoE(layer_number, local_expert_indices)
        self.self_attention = torch.nn.Linear(2, 2, bias=False)


class FakeChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([FakeLayer(9, [2, 3])])


def _new_optimizer_for_helpers():
    CDCOptimizer = _load_cdc_optimizer_class()
    optimizer = CDCOptimizer.__new__(CDCOptimizer)
    optimizer.model_chunks = [FakeChunk()]
    (
        optimizer._layer_prefix_to_global_idx,
        optimizer._local_moe_module_to_global_key,
        optimizer._global_moe_module_index,
    ) = optimizer._build_global_identity_indices()
    return optimizer


def _cdc_args(**overrides):
    args = SimpleNamespace(
        untie_embeddings_and_output_weights=False,
        cdc_sync_interval=10,
        cdc_algorithm="streaming",
        cdc_offload_outer_opt=False,
        cdc_outer_lr=1.0,
        cdc_dense_outer_lr=-1.0,
        cdc_moe_expert_outer_lr=-1.0,
        cdc_num_shards=3,
        cdc_dc_lambda=0.0,
        cdc_streaming_alpha=0.0,
        cdc_dense_alpha=-1.0,
        cdc_moe_router_alpha=-1.0,
        cdc_moe_expert_alpha=-1.0,
        cdc_moe_router_sync_mode="dedicated",
        cdc_delay=1,
        cdc_dc_N=1,
        cdc_shard_pattern="stride",
        cdc_moe_param_mode="dense-expert-hybrid",
        cdc_moe_expert_sync_interval=5,
        cdc_moe_expert_sync_offset=0,
        cdc_moe_expert_selection="score",
        cdc_moe_expert_topk=1,
        cdc_moe_expert_score_mode="token_load",
        cdc_moe_expert_layerwise_selection=True,
        cdc_moe_expert_max_age_slots=8,
        cdc_moe_expert_min_age_slots=0,
        cdc_blocking_full_sync_steps=0,
        cdc_verbose=False,
        bf16=False,
        fp16=False,
        num_layers=12,
        decoder_num_layers=None,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        cdc_parallel_size=2,
        cdc_dc_lambda_max=0.0,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _patch_cdc_runtime(monkeypatch, CDCOptimizer, args):
    globals_dict = CDCOptimizer.__init__.__globals__
    monkeypatch.setitem(globals_dict, "get_args", lambda: args)

    mpu = globals_dict["mpu"]
    monkeypatch.setattr(mpu, "get_cdc_parallel_group", lambda: "cdc", raising=False)
    monkeypatch.setattr(mpu, "get_cdc_parallel_world_size", lambda: 2, raising=False)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: "tp", raising=False)
    monkeypatch.setattr(mpu, "get_pipeline_model_parallel_group", lambda: "pp", raising=False)
    monkeypatch.setattr(
        mpu, "get_expert_tensor_parallel_group", lambda: "expert_tp", raising=False
    )


def _disable_init_collectives(monkeypatch, CDCOptimizer):
    monkeypatch.setattr(
        CDCOptimizer,
        "_all_reduce_flattened",
        lambda self, tensors, communication_dtype=None: None,
    )
    monkeypatch.setattr(
        CDCOptimizer,
        "_all_reduce_scalar_sum_across_groups",
        lambda self, value, groups: float(value),
    )


class FakeInnerOptimizer:
    def __init__(self, model_chunks):
        self.model_chunks = model_chunks
        self.config = SimpleNamespace()
        self.param_groups = [
            {"params": [param for chunk in model_chunks for param in chunk.parameters()]}
        ]
        self.optimizer = None
        self.state = {}


def _tracker_for_load(global_expert_idx=None):
    return {
        "params": [torch.zeros(2)],
        "staged_params": [torch.zeros(2)],
        "sent_at_step": 0,
        "old_sent_at_step": 0,
        "next_receive_step": 0,
        "global_num_params": 0,
        "last_score": 0.0,
        "last_token_load": 0.0,
        "token_load_accum": 0.0,
        "sent_at_expert_event": 0,
        "display_name": "tracker",
        "moe_module_key": None,
        "moe_layer_idx": None,
        "local_expert_idx": None,
        "global_expert_idx": global_expert_idx,
        "outer_optimizer": None,
    }


def test_global_layer_and_expert_identity_are_pp_and_ep_safe():
    optimizer = _new_optimizer_for_helpers()

    assert optimizer._layer_prefix_to_global_idx["decoder.layers.0"] == 8
    assert (
        optimizer._global_layer_index_for_param(
            "decoder.layers.0.self_attention.weight"
        )
        == 8
    )
    assert optimizer._local_moe_module_to_global_key["decoder.layers.0.mlp"] == (
        "decoder.layers.8.mlp"
    )

    metadata = optimizer._routed_expert_param_metadata(
        "decoder.layers.0.mlp.experts.local_experts.1.weight"
    )
    assert metadata == {
        "group_key": "decoder.layers.8.mlp.experts.global_experts.3",
        "moe_module_key": "decoder.layers.8.mlp",
        "moe_layer_idx": 8,
        "local_expert_idx": 1,
        "global_expert_idx": 3,
    }


def test_group_routed_expert_params_uses_global_expert_keys():
    optimizer = _new_optimizer_for_helpers()
    optimizer._expert_named_model_params = [
        (name, param)
        for name, param in optimizer.model_chunks[0].named_parameters()
        if ".experts." in name
    ]

    grouped = optimizer._group_routed_expert_named_params()

    assert sorted(grouped.keys()) == [
        "decoder.layers.8.mlp.experts.global_experts.2",
        "decoder.layers.8.mlp.experts.global_experts.3",
    ]
    assert grouped["decoder.layers.8.mlp.experts.global_experts.2"][
        "local_expert_idx"
    ] == 0
    assert grouped["decoder.layers.8.mlp.experts.global_experts.3"][
        "global_expert_idx"
    ] == 3


def test_tracker_records_kind_specific_score_reduce_groups():
    optimizer = _new_optimizer_for_helpers()
    optimizer.outer_lr = 1.0
    optimizer.streaming_alpha = 0.5
    optimizer.outer_state_dtype = torch.float32
    optimizer.outer_comm_dtype = torch.float32
    optimizer.offload_outer_opt = False
    optimizer._all_reduce_scalar_sum_across_groups = lambda value, groups: value + len(groups)
    optimizer._all_reduce_flattened = lambda tensors, communication_dtype=None: None

    param = torch.nn.Parameter(torch.ones(2, 3))
    tracker = optimizer._build_tracker(
        [param],
        score_reduce_groups=["expert_tp"],
        display_name="expert",
        global_expert_idx=3,
    )

    assert tracker["score_reduce_groups"] == ["expert_tp"]
    assert tracker["global_num_params"] == param.numel() + 1
    assert tracker["global_expert_idx"] == 3


def test_expert_token_values_average_expert_tp_then_sum_cdc(monkeypatch):
    optimizer = _new_optimizer_for_helpers()
    optimizer.expert_shard_tracker = {0: {"score_reduce_groups": ["expert_tp"]}}
    optimizer.cdc_group = "cdc"
    calls = []

    def fake_all_reduce_vector_sum(values, group):
        calls.append((group, list(values)))
        if group == "expert_tp":
            return [value * 2 for value in values]
        if group == "cdc":
            return [value + 10 for value in values]
        raise AssertionError(group)

    optimizer._all_reduce_vector_sum = fake_all_reduce_vector_sum
    dist_module = optimizer._reduce_expert_token_values.__globals__["dist"]
    monkeypatch.setattr(
        dist_module,
        "get_world_size",
        lambda group: 2 if group == "expert_tp" else 1,
    )

    assert optimizer._reduce_expert_token_values([0], [3.0]) == [13.0]
    assert calls == [("expert_tp", [3.0]), ("cdc", [3.0])]


def test_token_load_hook_uses_global_module_key_and_local_expert_slot():
    optimizer = _new_optimizer_for_helpers()
    optimizer.track_expert_token_load = True
    optimizer.verbose = False
    optimizer._token_load_hook_handles = []
    optimizer._token_load_hooked_expert_module_ids = set()
    optimizer._token_load_source_debug_printed = False
    optimizer.expert_shard_tracker = {
        7: {
            "moe_module_key": "decoder.layers.8.mlp",
            "local_expert_idx": 1,
            "token_load_step_accum": 0.0,
        }
    }

    optimizer._register_token_load_expert_hooks()
    experts = optimizer.model_chunks[0].decoder.layers[0].mlp.experts
    experts(torch.ones(3, 2), torch.tensor([4, 9], dtype=torch.int64), torch.ones(3))
    experts(torch.ones(2, 2), torch.tensor([1, 6], dtype=torch.int64), torch.ones(2))

    assert optimizer.expert_shard_tracker[7]["token_load_step_accum"] == 15.0

    for handle in optimizer._token_load_hook_handles:
        handle.remove()


def test_init_builds_global_identity_trackers_and_hooks(monkeypatch):
    CDCOptimizer = _load_cdc_optimizer_class()
    args = _cdc_args()
    _patch_cdc_runtime(monkeypatch, CDCOptimizer, args)
    _disable_init_collectives(monkeypatch, CDCOptimizer)

    model = FakeChunk()
    optimizer = CDCOptimizer(FakeInnerOptimizer([model]), model_chunks=[model])

    assert optimizer._layer_prefix_to_global_idx["decoder.layers.0"] == 8
    assert optimizer._local_moe_module_to_global_key["decoder.layers.0.mlp"] == (
        "decoder.layers.8.mlp"
    )
    assert sorted(optimizer._global_moe_module_index) == ["decoder.layers.8.mlp"]

    tracked_names = [name for name, _ in optimizer._tracked_named_model_params]
    assert all(".experts." not in name for name in tracked_names)
    assert all(".router." not in name for name in tracked_names)

    assert set(optimizer.shard_tracker) == {0, 1, 2}
    assert optimizer.shard_tracker[0]["score_reduce_groups"] == ["tp", "pp"]
    assert optimizer.router_tracker[0]["display_name"] == "moe-router"
    assert optimizer.router_tracker[0]["score_reduce_groups"] == ["tp", "pp"]

    expert_trackers = sorted(
        optimizer.expert_shard_tracker.values(),
        key=lambda tracker: tracker["global_expert_idx"],
    )
    assert [tracker["display_name"] for tracker in expert_trackers] == [
        "decoder.layers.8.mlp.experts.global_experts.2",
        "decoder.layers.8.mlp.experts.global_experts.3",
    ]
    assert [tracker["moe_layer_idx"] for tracker in expert_trackers] == [8, 8]
    assert [tracker["local_expert_idx"] for tracker in expert_trackers] == [0, 1]
    assert [tracker["global_expert_idx"] for tracker in expert_trackers] == [2, 3]
    assert all(tracker["score_reduce_groups"] == ["expert_tp"] for tracker in expert_trackers)
    assert optimizer._expert_layer_order == [8]
    assert optimizer._expert_layer_to_tracker_indices == {8: [0, 1]}

    assert optimizer.track_expert_token_load
    assert optimizer._token_load_module_to_tracker_indices == {"decoder.layers.8.mlp": [0, 1]}
    assert len(optimizer._token_load_hook_handles) == 1

    experts = model.decoder.layers[0].mlp.experts
    experts(torch.ones(4, 2), torch.tensor([7, 11], dtype=torch.int64), torch.ones(4))
    assert optimizer.expert_shard_tracker[0]["token_load_step_accum"] == 7.0
    assert optimizer.expert_shard_tracker[1]["token_load_step_accum"] == 11.0

    for handle in optimizer._token_load_hook_handles:
        handle.remove()


def test_init_rejects_cdc_parallel_size_one(monkeypatch):
    CDCOptimizer = _load_cdc_optimizer_class()
    args = _cdc_args(cdc_parallel_size=1)
    _patch_cdc_runtime(monkeypatch, CDCOptimizer, args)

    model = FakeChunk()
    with pytest.raises(AssertionError, match="cdc_parallel_size > 1"):
        CDCOptimizer(FakeInnerOptimizer([model]), model_chunks=[model])


def test_old_streaming_layout_is_rejected_only_when_global_identity_is_required(
    monkeypatch,
):
    CDCOptimizer = _load_cdc_optimizer_class()
    optimizer = CDCOptimizer.__new__(CDCOptimizer)
    optimizer.algorithm = "streaming"
    optimizer.enable_moe_router_refresh = False
    optimizer.step_count = 0
    optimizer.next_shard_idx = 0
    optimizer.next_expert_group_idx = 0
    optimizer.next_expert_layer_idx = 0
    optimizer.expert_sync_event_count = 0
    optimizer.shard_tracker = {0: _tracker_for_load()}
    optimizer.router_tracker = {}
    optimizer.expert_shard_tracker = {0: _tracker_for_load(global_expert_idx=3)}

    cdc_state = {
        "algorithm": "streaming",
        "streaming_layout_version": 2,
        "shards": [
            {
                "shard_idx": 0,
                "params": [torch.tensor([5.0, 6.0])],
                "staged_params": None,
                "sent_at_step": 4,
            }
        ],
        "expert_shards": [
            {
                "shard_idx": 0,
                "params": [torch.tensor([7.0, 8.0])],
                "staged_params": None,
                "local_expert_idx": 1,
            }
        ],
    }

    monkeypatch.setitem(
        CDCOptimizer.__init__.__globals__,
        "get_args",
        lambda: _cdc_args(pipeline_model_parallel_size=2, expert_model_parallel_size=1),
    )
    with pytest.raises(ValueError, match="without global layer/expert identities"):
        optimizer._load_cdc_state(cdc_state)

    monkeypatch.setitem(
        CDCOptimizer.__init__.__globals__,
        "get_args",
        lambda: _cdc_args(pipeline_model_parallel_size=1, expert_model_parallel_size=1),
    )
    optimizer._load_cdc_state(cdc_state)

    assert optimizer.shard_tracker[0]["sent_at_step"] == 4
    assert torch.equal(optimizer.shard_tracker[0]["params"][0], torch.tensor([5.0, 6.0]))
    assert torch.equal(
        optimizer.expert_shard_tracker[0]["params"][0], torch.tensor([7.0, 8.0])
    )
    assert optimizer.expert_shard_tracker[0]["local_expert_idx"] == 1
    assert optimizer.expert_shard_tracker[0]["global_expert_idx"] == 3
