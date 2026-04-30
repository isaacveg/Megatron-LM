# 简化版 NPU 顺序任务 Runner

这个脚本是比 `supervisor.py` 更简单的方案：宿主机只负责监控一组 Ascend Phy-ID，等空闲后按顺序执行容器内某个目录下的 `.sh` 脚本。

## 核心行为

1. 从容器内 `--script-dir` 读取所有匹配 `--pattern` 的脚本并排序。
2. 对每个脚本：
   - 等指定 NPU Phy-ID 组空闲。
   - 用 `docker exec` 在容器内执行脚本。
   - 退出码为 `0` 视为完成，清理旧 checkpoint，然后进入下一个脚本。
   - 退出码非 `0` 视为失败，继续重试同一个脚本。
   - 连续失败达到 `--max-failures` 后跳过该脚本。
3. 所有结果写到 `--log-file`。
4. 运行时在终端按 `q` 会请求停止，并尝试终止当前容器内任务。

## 前 8 张卡

这里的卡号使用 `npu-smi info` 中的 `Phy-ID`。

```bash
python3 automation/simple_npu_job_loop.py \
  --name front8 \
  --container YOUR_CONTAINER \
  --workdir /data/yzhu/megatron-lm-moe-exp \
  --script-dir /data/yzhu/megatron-lm-moe-exp/experiment_scripts/front8 \
  --devices 0,1,2,3,4,5,6,7 \
  --log-file automation/simple_logs/front8_$(date +%Y%m%d_%H%M%S).log \
  --state-file automation/simple_state_front8.json \
  --ckpt-root /data/yzhu/megatron-lm-moe-exp/ckpts
```

## 后 8 张卡

```bash
python3 automation/simple_npu_job_loop.py \
  --name back8 \
  --container YOUR_CONTAINER \
  --workdir /data/yzhu/megatron-lm-moe-exp \
  --script-dir /data/yzhu/megatron-lm-moe-exp/experiment_scripts/back8 \
  --devices 8,9,10,11,12,13,14,15 \
  --log-file automation/simple_logs/back8_$(date +%Y%m%d_%H%M%S).log \
  --state-file automation/simple_state_back8.json \
  --ckpt-root /data/yzhu/megatron-lm-moe-exp/ckpts
```

## 空闲判断

默认条件：

```text
HBM <= 8000MB
AICore <= 10%
进程表里没有该 Phy-ID 对应的进程
连续通过 3 次检查
启动前 cooldown 300 秒
```

如果空闲时 HBM 基础占用比 8GB 更高，可以调大：

```bash
--idle-hbm-mb 12000
```

## checkpoint 清理

脚本完成后，runner 会尝试从训练脚本里解析：

```bash
RUN_NAME
CKPT_ROOT
CKPT_SAVE_DIR
```

如果能得到 checkpoint 目录，就只保留最后 `--keep-last-checkpoints` 个 `iter_*` 目录，默认保留 2 个。

关闭清理：

```bash
--keep-last-checkpoints -1
```

如果自动解析不到 checkpoint 目录，可以通过 `--ckpt-root` 辅助；如果所有脚本共用同一个 checkpoint 目录，也可以强制指定 `--ckpt-dir`。

