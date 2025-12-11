# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import torch
import torch.distributed as dist
from torch.optim import SGD
from copy import deepcopy
from typing import List, Dict, Any, Optional
import time

from megatron.core import mpu
from megatron.training.utils import print_rank_0
from megatron.core.optimizer import MegatronOptimizer
from megatron.core.transformer.module import MegatronModule

class CDCOptimizer(MegatronOptimizer):
    """
    Wrapper optimizer for cross data-center training (DiLoCo, Streaming, DC).
    Wraps a standard Megatron optimizer and adds outer optimization loop.
    """

    def __init__(
        self,
        inner_optimizer: MegatronOptimizer,
        args: Any,
        model_chunks: List[MegatronModule],
    ):
        super().__init__(
            inner_optimizer.optimizer,
            inner_optimizer.config,
            inner_optimizer.init_state_fn,
        )
        self.inner_optimizer = inner_optimizer
        self.args = args
        self.model_chunks = model_chunks

        # CDC parallel group and arguments
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
        if torch.distributed.get_rank() == 0: 
            self._check_weight_tying()
        # Initialize DiLoCo state
        self.original_snapshot = None
        self.outer_optimizer = None
        self.shard_tracker = None
        self.next_shard_idx = 0
        
        if self.verbose:
            print_rank_0(f"[CDC] Initialized {self.algorithm} optimizer. Sync interval: {self.sync_interval}, Shards: {self.num_shards}")

        if self.algorithm == 'diloco':
            self._init_diloco_state()
        elif self.algorithm in ['streaming', 'dc']:
            self._init_streaming_state()
        else:
            raise ValueError(f"Unknown DiLoCo algorithm: {self.algorithm}")
    # 在你的代码中加入这段打印
    def _check_weight_tying(self):
        # 只有 rank0 打印，防止刷屏
        if torch.distributed.get_rank() != 0:
            return

        print_rank_0("[CDC Check] Checking for Weight Tying...")
        found_embedding = False
        found_head = False
        emb_id = None
        head_id = None

        # 注意这里要用 self.model_chunks
        for chunk in self.model_chunks:
            for name, param in chunk.named_parameters():
                # 检查 Embedding
                if 'embedding' in name and 'weight' in name:
                    if 'position' not in name:
                        emb_id = id(param)
                        found_embedding = True
                        print_rank_0(f"[CDC Check] Found Embedding: {name} (ID: {emb_id})")
                
                # 检查 Head
                if ('output_layer' in name or 'head' in name) and 'weight' in name:
                    if 'norm' not in name:
                        head_id = id(param)
                        found_head = True
                        print_rank_0(f"[CDC Check] Found Head: {name} (ID: {head_id})")

        if found_embedding and found_head:
            if emb_id == head_id:
                print_rank_0("[CDC Check] >>> DETECTED: Weight Tying is ON. (Embedding and Head share same memory)")
            else:
                print_rank_0("[CDC Check] >>> DETECTED: Weight Tying is OFF. (Different memory addresses)")
        else:
            # 如果没找到，可能是模型结构命名不一样，或者是 Pipeline Parallelism 导致当前 rank 只有部分层
            pass
    @property
    def is_stub_optimizer(self):
        return getattr(self.inner_optimizer, 'is_stub_optimizer', False)

    def get_loss_scale(self):
        return self.inner_optimizer.get_loss_scale()

    def reload_model_params(self):
        self.inner_optimizer.reload_model_params()

    def state_dict(self, is_loading: bool = False):
        """Return optimizer state plus CDC metadata."""
        return {
            "inner_optimizer": self.inner_optimizer.state_dict(),
            "cdc_state": self._build_cdc_state(),
        }

    def load_state_dict(self, state_dict):
        """Load optimizer state including CDC metadata (backward compatible)."""
        if "inner_optimizer" not in state_dict:
            # Backward compatibility with checkpoints saved before CDC support.
            self.inner_optimizer.load_state_dict(state_dict)
            return

        self.inner_optimizer.load_state_dict(state_dict["inner_optimizer"])
        self._load_cdc_state(state_dict.get("cdc_state"))

    def zero_grad(self, set_to_none=True):
        self.inner_optimizer.zero_grad(set_to_none)

    def get_main_param_groups(self):
        if hasattr(self.inner_optimizer, 'get_main_param_groups'):
            return self.inner_optimizer.get_main_param_groups()
        return self.inner_optimizer.param_groups

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _build_cdc_state(self):
        state = {
            "algorithm": self.algorithm,
            "step_count": self.step_count,
            "next_shard_idx": self.next_shard_idx,
        }

        if self.algorithm == 'diloco' and self.original_snapshot is not None:
            state["diloco"] = self._serialize_diloco_state()
        elif self.algorithm in ['streaming', 'dc'] and self.shard_tracker is not None:
            state["shards"] = self._serialize_shard_trackers()

        return state

    def _serialize_diloco_state(self):
        # Optimization: If we just synced, original_snapshot == local params.
        # We can skip saving it to avoid redundancy.
        if self.step_count > 0 and self.step_count % self.sync_interval == 0:
            snapshot = None
        else:
            snapshot = []
            for group in self.original_snapshot or []:
                serialized = [self._clone_tensor_to_cpu(p) for p in group]
                snapshot.append(serialized)

        outer_state = None
        if self.outer_optimizer is not None:
            outer_state = self._optimizer_state_to_cpu(self.outer_optimizer.state_dict())

        return {
            "original_snapshot": snapshot,
            "outer_optimizer": outer_state,
        }

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
        if not cdc_state:
            return

        checkpoint_algorithm = cdc_state.get("algorithm", self.algorithm)
        if checkpoint_algorithm != self.algorithm:
            raise ValueError(
                f"Checkpoint algorithm {checkpoint_algorithm} does not match runtime algorithm {self.algorithm}."
            )

        self.step_count = cdc_state.get("step_count", self.step_count)
        self.next_shard_idx = cdc_state.get("next_shard_idx", self.next_shard_idx)

        if self.algorithm == 'diloco':
            self._load_diloco_state(cdc_state.get("diloco"))
        elif self.algorithm in ['streaming', 'dc']:
            self._load_shard_trackers(cdc_state.get("shards"))

    def _load_diloco_state(self, diloco_state):
        if not diloco_state:
            return

        if self.original_snapshot is None:
            self._init_diloco_state()

        snapshot = diloco_state.get("original_snapshot")
        
        # Optimization: If snapshot is None, it means it was identical to local params.
        if snapshot is None:
            param_groups = self.get_main_param_groups()
            for target_group, local_group in zip(self.original_snapshot, param_groups):
                for target_tensor, local_tensor in zip(target_group, local_group['params']):
                    if self.offload_outer_opt:
                        target_tensor.data.copy_(local_tensor.detach().to("cpu"))
                    else:
                        target_tensor.data.copy_(local_tensor.data)
        else:
            if len(snapshot) != len(self.original_snapshot):
                raise ValueError("Mismatch in DiLoCo snapshot group count while loading checkpoint.")

            for target_group, saved_group in zip(self.original_snapshot, snapshot):
                if len(saved_group) != len(target_group):
                    raise ValueError("Mismatch in DiLoCo snapshot tensor count while loading checkpoint.")
                for target_tensor, saved_tensor in zip(target_group, saved_group):
                    self._copy_tensor_data(target_tensor, saved_tensor)

        if self.outer_optimizer is not None and diloco_state.get("outer_optimizer") is not None:
            self.outer_optimizer.load_state_dict(diloco_state["outer_optimizer"])

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
            tracker["global_num_params"] = shard_state.get("global_num_params", tracker["global_num_params"])
            tracker["last_score"] = shard_state.get("last_score", tracker["last_score"])

            outer_state = shard_state.get("outer_optimizer")
            if tracker.get("outer_optimizer") is not None and outer_state is not None:
                tracker["outer_optimizer"].load_state_dict(outer_state)

    def _copy_tensor_list(self, target_list, saved_list):
        if len(target_list) != len(saved_list):
            raise ValueError("Mismatch in tensor list lengths while restoring checkpoint state.")
        for target_tensor, saved_tensor in zip(target_list, saved_list):
            self._copy_tensor_data(target_tensor, saved_tensor)

    @staticmethod
    def _clone_tensor_to_cpu(tensor):
        if tensor.device.type == 'cpu':
            return tensor.detach().clone()
        return tensor.detach().to('cpu')

    @staticmethod
    def _copy_tensor_data(target_tensor, saved_tensor):
        with torch.no_grad():
            target_tensor.data.copy_(
                saved_tensor.to(device=target_tensor.device, dtype=target_tensor.dtype)
            )

    def _init_diloco_state(self):
        """Initialize state for standard DiLoCo."""
        param_groups = self.get_main_param_groups()
        
        self.original_snapshot = []
        for group in param_groups:
            snapshot_group = []
            for param in group['params']:
                if self.offload_outer_opt:
                    snapshot_group.append(param.detach().to("cpu", copy=True))
                else:
                    snapshot_group.append(param.detach().clone())
            self.original_snapshot.append(snapshot_group)

        # Initialize outer optimizer (SGD)
        if self.outer_lr != 1.0:
            all_snapshot_params = [p for group in self.original_snapshot for p in group]
            for p in all_snapshot_params:
                p.requires_grad_(True)
                
            self.outer_optimizer = SGD(
                all_snapshot_params,
                lr=self.outer_lr,
                momentum=0.9,
                nesterov=True,
            )

    # def _init_streaming_state(self):
    #     """
    #     Initialize state for Streaming/DC DiLoCo using explicit Model Layer Structure.
    #     Uses 'Alignment by Order' to map Model Params (Names) to Optimizer Params (Tensors).
    #     """
    #     # 1. 获取优化器管理的所有参数（展平列表）
    #     # 注意：这通常是 FP32 Master Weights，与 model_chunks 里的参数对象不同，但顺序一致
    #     optim_params_flat = []
    #     for group in self.get_main_param_groups():
    #         for p in group['params']:
    #             optim_params_flat.append(p)
        
    #     total_optim_params = len(optim_params_flat)
    #     self.shard_tracker = {}
        
    #     if self.verbose:
    #         print_rank_0(f"[CDC] Initializing Shards. Found {total_optim_params} optimizer params.")

    #     # -------------------------------------------------------
    #     # 2. 逻辑分组 (Blocks): 遍历模型结构，按顺序认领优化器参数
    #     # -------------------------------------------------------
    #     blocks = []
    #     current_block = []
    #     current_layer_idx = -1 
        
    #     # 指向 optim_params_flat 的当前索引
    #     optim_cursor = 0
        
    #     # 遍历 model chunks
    #     for chunk in self.model_chunks:
    #         # 只遍历需要梯度的参数，以保持和优化器列表对齐
    #         # 注意：这里我们遍历的是 model 的参数名，用来判断结构
    #         for name, param in chunk.named_parameters():
    #             if not param.requires_grad:
    #                 continue 

    #             # 安全检查：防止模型参数多于优化器参数
    #             if optim_cursor >= total_optim_params:
    #                 if self.verbose:
    #                     print_rank_0(f"[CDC] Warning: Model has more params than optimizer at {name}. Stopping alignment.")
    #                 break

    #             # === 关键点：取出对应的优化器参数 ===
    #             # 我们使用 model 的 name 来判断层级，但保存 optimizer 的 tensor
    #             target_param = optim_params_flat[optim_cursor]
    #             optim_cursor += 1

    #             # 解析层号: 'language_model.encoder.layers.0.self_attention...'
    #             parts = name.split('.')
    #             found_layer = False
    #             layer_num = -1

    #             if 'layers' in parts:
    #                 try:
    #                     idx = parts.index('layers')
    #                     if idx + 1 < len(parts) and parts[idx+1].isdigit():
    #                         layer_num = int(parts[idx+1])
    #                         found_layer = True
    #                 except (ValueError, IndexError):
    #                     pass

    #             # 状态机：检测层边界
    #             if found_layer:
    #                 if layer_num != current_layer_idx:
    #                     if current_block:
    #                         blocks.append(current_block)
    #                         current_block = []
    #                     current_layer_idx = layer_num
    #                 current_block.append(target_param)
    #             else:
    #                 if current_layer_idx != -1: # 从 Layer 区域出来 (进入 Head)
    #                     if current_block:
    #                         blocks.append(current_block)
    #                         current_block = []
    #                     current_layer_idx = -1
    #                 current_block.append(target_param)

    #     # 添加最后一个 block (Tail)
    #     if current_block:
    #         blocks.append(current_block)

    #     # 校验对齐情况
    #     if optim_cursor != total_optim_params:
    #         print_rank_0(f"[CDC] WARNING: Parameter count mismatch! Optimizer has {total_optim_params}, "
    #                      f"but matched {optim_cursor} based on model structure. "
    #                      f"This might affect DiLoCo accuracy.")

    #     num_blocks = len(blocks)
    #     if self.verbose:
    #         print_rank_0(f"[CDC] Partitioned model into {num_blocks} blocks.")
    #         print_rank_0(f"[CDC] Shard Pattern: {self.shard_pattern.upper()}")
    #         if num_blocks == 0:
    #             print_rank_0("[CDC] ERROR: 0 Blocks found! Check if model.requires_grad is set correctly.")

    #     # -------------------------------------------------------
    #     # 3. 根据 pattern 将 Blocks 分配给 Shards
    #     # -------------------------------------------------------
        
    #     for shard_idx in range(self.num_shards):
    #         shard_params_refs = []
    #         assigned_block_indices = []

    #         if num_blocks > 0:
    #             if self.shard_pattern == 'contiguous':
    #                 # 方式 A: 连续划分
    #                 blocks_per_shard = (num_blocks + self.num_shards - 1) // self.num_shards
    #                 start_block_idx = shard_idx * blocks_per_shard
    #                 end_block_idx = min((shard_idx + 1) * blocks_per_shard, num_blocks)
                    
    #                 if start_block_idx < end_block_idx:
    #                     for b_idx in range(start_block_idx, end_block_idx):
    #                         shard_params_refs.extend(blocks[b_idx])
    #                         assigned_block_indices.append(b_idx)
                
    #             elif self.shard_pattern == 'stride':
    #                 # 方式 B: 交错划分 (Round-Robin)
    #                 for b_idx, block in enumerate(blocks):
    #                     if b_idx % self.num_shards == shard_idx:
    #                         shard_params_refs.extend(block)
    #                         assigned_block_indices.append(b_idx)
            
    #         # === 初始化 Tracker (即使是空的也要初始化，防止 KeyError) ===
    #         shard_num_params_cnt = sum(p.numel() for p in shard_params_refs)
            
    #         tracker = {
    #             "param_refs": shard_params_refs,
    #             "params": [], 
    #             "staged_params": [],
    #             "sent_at_step": 0,
    #             "old_sent_at_step": 0,
    #             "next_receive_step": 0,
    #             "global_num_params": shard_num_params_cnt,
    #             "last_score": 0.0,
    #         }

    #         # 只有当分配到了参数时才进行 clone/offload
    #         if shard_params_refs:
    #             for p in shard_params_refs:
    #                 if self.offload_outer_opt:
    #                     tracker["params"].append(p.detach().to("cpu", copy=True))
    #                     tracker["staged_params"].append(p.detach().to("cpu", copy=True))
    #                 else:
    #                     tracker["params"].append(p.detach().clone())
    #                     tracker["staged_params"].append(p.detach().clone())

    #             if self.outer_lr != 1.0:
    #                 for p in tracker["params"]:
    #                     p.requires_grad_(True)
    #                 tracker["outer_optimizer"] = SGD(
    #                     tracker["params"],
    #                     lr=self.outer_lr,
    #                     momentum=0.9,
    #                     nesterov=True,
    #                 )
    #             else:
    #                 tracker["outer_optimizer"] = None
    #         else:
    #             tracker["outer_optimizer"] = None

    #         self.shard_tracker[shard_idx] = tracker
            
    #         if self.verbose:
    #             shard_mb = shard_num_params_cnt * 4 / 1024**2 # Approx FP32 size
    #             if assigned_block_indices:
    #                 if self.shard_pattern == 'contiguous':
    #                     idx_info = f"Blocks {assigned_block_indices[0]}-{assigned_block_indices[-1]}"
    #                 else:
    #                     idx_info = f"Blocks {assigned_block_indices[:3]}... (Count: {len(assigned_block_indices)})"
    #             else:
    #                 idx_info = "No Blocks Assigned"
                    
    #             print_rank_0(f"[CDC] Shard {shard_idx}: {idx_info} "
    #                          f"({len(shard_params_refs)} tensors, {shard_mb:.2f} MB)")
    def _init_streaming_state(self):
        """
        Initialize state for Streaming/DC DiLoCo using explicit Model Layer Structure.
        Robust strategy: "Map by Unique ID" to handle Weight Tying.
        """
        # Step 1: 建立 [Model Param ID] -> [Optimizer Param Tensor] 的映射
        # 1.1 收集所有优化器参数
        optim_params_flat = []
        for group in self.get_main_param_groups():
            for p in group['params']:
                optim_params_flat.append(p)
        
        # 1.2 收集所有模型参数
        unique_model_params = []
        for chunk in self.model_chunks:
            for p in chunk.parameters():
                if p.requires_grad:
                    unique_model_params.append(p)
        
        # 1.3 校验数量并建立映射
        if len(optim_params_flat) != len(unique_model_params):
             print_rank_0(f"[CDC] CRITICAL ERROR: Optimizer has {len(optim_params_flat)} params, "
                          f"but model has {len(unique_model_params)} unique params. Alignment impossible.")
        
        param_map = {} # Key: id(model_param), Value: optimizer_param_tensor
        
        min_len = min(len(optim_params_flat), len(unique_model_params))
        for i in range(min_len):
            m_p = unique_model_params[i]
            o_p = optim_params_flat[i]
            param_map[id(m_p)] = o_p
            
        if self.verbose:
            print_rank_0(f"[CDC] Mapped {min_len} unique model params to optimizer params.")

        # Step 2: 划分blocks
        blocks = []
        current_block = []
        current_layer_idx = -1 
        
        self.shard_tracker = {}

        for chunk in self.model_chunks:
            for name, param in chunk.named_parameters():
                if not param.requires_grad:
                    continue 
                
                if id(param) not in param_map:
                    if self.verbose:
                        print_rank_0(f"[CDC] Warning: Param {name} not found in alignment map. Skipping.")
                    continue
                
                target_param = param_map[id(param)]

                parts = name.split('.')
                found_layer = False
                layer_num = -1
                if 'layers' in parts:
                    try:
                        idx = parts.index('layers')
                        if idx + 1 < len(parts) and parts[idx+1].isdigit():
                            layer_num = int(parts[idx+1])
                            found_layer = True
                    except (ValueError, IndexError):
                        pass

                if found_layer:
                    if layer_num != current_layer_idx:
                        if current_block:
                            blocks.append(current_block)
                            current_block = []
                        current_layer_idx = layer_num
                    current_block.append(target_param)
                else:
                    if current_layer_idx != -1: 
                        if current_block:
                            blocks.append(current_block)
                            current_block = []
                        current_layer_idx = -1
                    current_block.append(target_param)

        if current_block:
            blocks.append(current_block)

        num_blocks = len(blocks)
        if self.verbose:
            print_rank_0(f"[CDC] Partitioned model into {num_blocks} blocks.")

        # Step 3: 分配给 Shards
        for shard_idx in range(self.num_shards):
            shard_params_refs = []
            
            if num_blocks > 0:
                if self.shard_pattern == 'sequential':
                    blocks_per_shard = (num_blocks + self.num_shards - 1) // self.num_shards
                    start_block_idx = shard_idx * blocks_per_shard
                    end_block_idx = min((shard_idx + 1) * blocks_per_shard, num_blocks)
                    if start_block_idx < end_block_idx:
                        for b_idx in range(start_block_idx, end_block_idx):
                            shard_params_refs.extend(blocks[b_idx])
                
                elif self.shard_pattern == 'stride':
                    for b_idx, block in enumerate(blocks):
                        if b_idx % self.num_shards == shard_idx:
                            shard_params_refs.extend(block)
            
            # Tracker 初始化
            shard_num_params_cnt = sum(p.numel() for p in shard_params_refs)
            tracker = {
                "param_refs": shard_params_refs,
                "params": [], 
                "staged_params": [],
                "sent_at_step": 0,
                "old_sent_at_step": 0,
                "next_receive_step": 0,
                "global_num_params": shard_num_params_cnt,
                "last_score": 0.0,
            }

            if shard_params_refs:
                for p in shard_params_refs:
                    if self.offload_outer_opt:
                        tracker["params"].append(p.detach().to("cpu", copy=True))
                        tracker["staged_params"].append(p.detach().to("cpu", copy=True))
                    else:
                        tracker["params"].append(p.detach().clone())
                        tracker["staged_params"].append(p.detach().clone())

                if self.outer_lr != 1.0:
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
            else:
                tracker["outer_optimizer"] = None

            self.shard_tracker[shard_idx] = tracker
            
            if self.verbose:
                 print_rank_0(f"[CDC] Shard {shard_idx} initialized with {len(shard_params_refs)} tensors.")

    def step(self):
        """
        Performs a single optimization step.
        1. Inner step (Megatron DP).
        2. Outer step (DiLoCo sync) if interval is met.
        """
        # 1. Inner Step
        update_successful, grad_norm, num_zeros_in_grad = self.inner_optimizer.step()
        
        if update_successful:
            self.step_count += 1
            
            # 2. Outer Step
            if self.algorithm == 'diloco':
                if self.step_count % self.sync_interval == 0:
                    if self.verbose:
                        print_rank_0(f"[CDC] Step {self.step_count}: Triggering DiLoCo sync...")
                    self._sync_diloco()
            elif self.algorithm in ['streaming', 'dc']:
                self._sync_step()
                
        return update_successful, grad_norm, num_zeros_in_grad

    def _sync_diloco(self):
        """Standard DiLoCo synchronization."""
        start_time = time.time()
        param_groups = self.get_main_param_groups()
        
        # Calculate pseudo-gradients: G = Original - Current
        for group_idx, group in enumerate(param_groups):
            snapshot_group = self.original_snapshot[group_idx]
            
            for param_idx, param in enumerate(group['params']):
                original_param = snapshot_group[param_idx]
                
                if original_param.grad is None:
                    original_param.grad = torch.zeros_like(original_param.data)
                
                if self.offload_outer_opt:
                    p_cpu = param.detach().to("cpu")
                    original_param.grad.copy_(original_param.data)
                    original_param.grad.sub_(p_cpu)
                else:
                    original_param.grad.copy_(original_param.data)
                    original_param.grad.sub_(param.data)
                
                if self.offload_outer_opt:
                    pass 

        # Batch All-Reduce for efficiency
        all_grads = []
        for group in self.original_snapshot:
            for p in group:
                all_grads.append(p.grad)
        
        self._all_reduce_flattened(all_grads)
        
        # Outer Optimizer Step
        if self.outer_optimizer:
            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
        else:
            # Simple averaging
            for group in self.original_snapshot:
                for p in group:
                    p.data.sub_(p.grad)

        # Copy back to current model
        for group_idx, group in enumerate(param_groups):
            snapshot_group = self.original_snapshot[group_idx]
            for param_idx, param in enumerate(group['params']):
                original_param = snapshot_group[param_idx]
                if self.offload_outer_opt:
                    param.data.copy_(original_param.data.to(param.device))
                else:
                    param.data.copy_(original_param.data)

        if self.verbose:
            duration = time.time() - start_time
            print_rank_0(f"[CDC] Step {self.step_count}: DiLoCo sync complete in {duration:.4f}s.")

    def _sync_step(self):
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
            
            I_p = self.step_count - tracker["sent_at_step"]
            if I_p == 0: I_p = 1
            
            current_R = update_magnitude_sq * 1e8 / (I_p * tracker["global_num_params"])
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
            if self.offload_outer_opt:
                p_staged.data.copy_(p_local.detach().to("cpu"))
            else:
                p_staged.data.copy_(p_local.data)
                
        tracker["old_sent_at_step"] = tracker["sent_at_step"]
        tracker["sent_at_step"] = self.step_count
        
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
            if self.offload_outer_opt:
                p_staged_dev = p_staged.to(p_global.device) if p_global.device.type != 'cpu' else p_staged
            else:
                p_staged_dev = p_staged
                
            g = p_global.data.clone()
            g.sub_(p_staged_dev.data)
            sync_grads.append(g)
            
        # 2. All-Reduce sync_grads (Across DiLoCo Islands)
        self._all_reduce_flattened(sync_grads)
        
        # 3. Calculate Score for Next Selection (Norm of Global Pseudo-Gradient)
        total_norm_sq = 0.0
        for g in sync_grads:
            total_norm_sq += g.norm(2).item() ** 2
            
        # All-Reduce norm across TP group
        tp_group = mpu.get_tensor_model_parallel_group()
        if tp_group is not None and dist.get_world_size(group=tp_group) > 1:
            norm_tensor = torch.tensor(total_norm_sq, device=sync_grads[0].device)
            dist.all_reduce(norm_tensor, op=dist.ReduceOp.SUM, group=tp_group)
            total_norm_sq = norm_tensor.item()
            
        tracker["last_score"] = total_norm_sq
        
        if self.verbose:
            duration = time.time() - tracker.get("sync_start_time", time.time())
            print_rank_0(f"[CDC] Step {self.step_count}: Completed sync for shard {shard_idx} in {duration:.4f}s. Score (Norm^2): {total_norm_sq:.4e}")

        # 4. Outer Optimizer Step (Update Global)
        if tracker["outer_optimizer"]:
            tracker["outer_optimizer"].zero_grad()
            for p_global, avg_delta in zip(global_params, sync_grads):
                if p_global.grad is None:
                    p_global.grad = torch.zeros_like(p_global.data)
                p_global.grad.copy_(avg_delta)
            tracker["outer_optimizer"].step()
        else:
            # Simple averaging
            for p_global, avg_delta in zip(global_params, sync_grads):
                p_global.data.sub_(avg_delta)
                
        # 5. Update Local Params (Algorithm Specific)
        if self.algorithm == 'dc':
            # Delay Compensation (Taylor Expansion)
            g_1 = []
            D = []
            
            for p_staged, p_local, p_global in zip(staged_params, param_refs, global_params):
                if self.offload_outer_opt:
                    p_local_data = p_local.detach().to("cpu")
                    p_global_data = p_global.data
                else:
                    p_local_data = p_local.data
                    p_global_data = p_global.data
                    
                # g_1 = Staged - Local
                g1_tensor = p_staged.data.clone().sub_(p_local_data)
                g_1.append(g1_tensor)
                
                # D = Global - Staged
                d_tensor = p_global_data.clone().sub_(p_staged.data)
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
                    
        elif self.algorithm == 'streaming':
            # Alpha Blending
            for p_local, p_global in zip(param_refs, global_params):
                if self.offload_outer_opt:
                    p_global_data = p_global.data.to(p_local.device)
                else:
                    p_global_data = p_global.data
                    
                p_local.data.mul_(self.streaming_alpha).add_(p_global_data, alpha=1.0 - self.streaming_alpha)

    def _all_reduce_flattened(self, tensors):
        """Helper to flatten, all-reduce, and unflatten tensors."""
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
        
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
            
            # Track size
            total_bytes += flat_tensor.numel() * flat_tensor.element_size()
            
            # Move to GPU for NCCL if needed
            device = flat_tensor.device
            if self.offload_outer_opt and device.type == 'cpu':
                # Use Gloo or move to GPU
                # Assuming we have a GPU available
                gpu_tensor = flat_tensor.cuda()
                dist.all_reduce(gpu_tensor, group=self.cdc_group)
                flat_tensor.copy_(gpu_tensor.cpu())
            else:
                dist.all_reduce(flat_tensor, group=self.cdc_group)
                
            # Average
            flat_tensor.div_(dist.get_world_size(group=self.cdc_group))
            
            # Unflatten and copy back to grads
            for t, synced_t in zip(group_tensors, _unflatten_dense_tensors(flat_tensor, group_tensors)):
                t.copy_(synced_t)

        end_time = time.time()
        duration = end_time - start_time
        
        if self.verbose:
            size_mb = total_bytes / (1024 * 1024)
            bandwidth = size_mb / duration if duration > 0 else 0
            print_rank_0(f"[CDC] Communication: {size_mb:.2f} MB in {duration:.4f}s ({bandwidth:.2f} MB/s)")

    def prepare_grads(self):
        return self.inner_optimizer.prepare_grads()

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

