import importlib.util
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_cdc_optimizer_class():
    module_path = REPO_ROOT / "megatron/core/optimizer/cdc_optimizer.py"
    module_name = f"_cdc_optimizer_shard_sync_test_{dist.get_rank()}"
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
        cdc_dc_lambda_max=0.0,
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


def _run_shard_sync_worker(rank, world_size, init_method):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, init_method=init_method)

    from megatron.core import parallel_state as ps

    ps.initialize_model_parallel(
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        expert_tensor_parallel_size=2,
        context_parallel_size=1,
        cdc_parallel_size=2,
        order="tp-cp-ep-dp-pp",
        create_gloo_process_groups=False,
    )

    CDCOptimizer = _load_cdc_optimizer_class()
    optimizer = CDCOptimizer.__new__(CDCOptimizer)
    optimizer.cdc_group = ps.get_cdc_parallel_group()
    optimizer.outer_comm_dtype = torch.float32
    optimizer.mixed_precision = False
    optimizer.verbose = False
    optimizer.algorithm = "streaming"
    optimizer.streaming_alpha = 0.0

    device = torch.device("cuda", rank)
    tp_slot = rank % 2
    if tp_slot == 0:
        staged_values = [1.0, 2.0] if rank == 0 else [3.0, 6.0]
        global_values = [10.0, 11.0]
        expected_synced = [2.0, 4.0]
    else:
        staged_values = [5.0, 8.0] if rank == 1 else [9.0, 14.0]
        global_values = [20.0, 21.0]
        expected_synced = [7.0, 11.0]

    local_param = torch.nn.Parameter(torch.full((2,), -1000.0 - rank, device=device))
    tracker = {
        "display_name": "dense-shard-0",
        "param_refs": [local_param],
        "params": [torch.tensor(global_values, device=device)],
        "staged_params": [torch.tensor(staged_values, device=device)],
        "outer_optimizer": None,
        "score_reduce_groups": [ps.get_tensor_model_parallel_group()],
        "comm_dtype": torch.float32,
        "apply_alpha": 0.0,
        "sync_start_time": 0.0,
    }

    optimizer._complete_tracker_sync(
        {0: tracker},
        tracker_idx=0,
        tracker_kind="dense-shard",
        reload_main_params=False,
        use_algorithm_specific_update=False,
        force_full_copy=True,
    )
    torch.cuda.synchronize(device)

    local_synced = [round(float(x), 6) for x in local_param.detach().cpu().tolist()]
    global_synced = [round(float(x), 6) for x in tracker["params"][0].detach().cpu().tolist()]
    assert local_synced == expected_synced
    assert global_synced == expected_synced
    assert abs(float(tracker["last_score"]) - 382.0) < 1e-4

    dist.barrier(device_ids=[rank])
    ps.destroy_model_parallel()
    dist.destroy_process_group()


def test_complete_tracker_sync_averages_same_cdc_shard_and_reduces_tp_score():
    if torch.cuda.device_count() < 4:
        pytest.skip("CDC shard-sync smoke requires four CUDA devices for TP=2, CDC=2.")

    with tempfile.TemporaryDirectory() as tmpdir:
        mp.start_processes(
            _run_shard_sync_worker,
            args=(4, f"file://{Path(tmpdir) / 'cdc_shard_sync_init'}"),
            nprocs=4,
            join=True,
            start_method="spawn",
        )
