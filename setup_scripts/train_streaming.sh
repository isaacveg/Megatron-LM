#!/bin/bash

WORKSPACE_ROOT="XXX"           # 工作空间根目录
CODE_DIR="${WORKSPACE_ROOT}/Megatron-LM"         # 代码存放目录
TOKENIZER_DIR="${WORKSPACE_ROOT}/tokenizer"      # 分词器存放目录
DATA_RAW_DIR="${WORKSPACE_ROOT}/data/raw/C4_en"  # 原始数据存放目录
DATA_PROCESSED_PREFIX="${WORKSPACE_ROOT}/data/processed/c4_en_train"  # 处理后数据前缀

ALGORITHM="streaming"                                # 跨数据中心算法
CKPT_SAVE_DIR="${WORKSPACE_ROOT}/checkpoints/${ALGORITHM}"  # 检查点保存路径
LOG_DIR="${WORKSPACE_ROOT}/logs/${ALGORITHM}"               # 日志保存路径
CKPT_LOAD_DIR="${WORKSPACE_ROOT}/checkpoints/${ALGORITHM}"  # 检查点保存路径

# 训练相关配置
VISIBLE_GPUS="0,1,2,3,4,5,6,7"                   # 可见GPU设备
NPUS_PER_NODE=8                          # 每节点GPU数量



echo "=========================================="
echo "启动跨数据中心训练"
echo "=========================================="

mkdir -p "${LOG_DIR}"
mkdir -p "${CKPT_SAVE_DIR}"

LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# 环境变量配置
export NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS}

# 分布式配置
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

# 模型并行参数
TP=2  # 张量并行
PP=1  # 流水线并行
TOKENIZER_MODEL="${TOKENIZER_DIR}/llama-1b-fresh-tokenizer.model"
# 模型训练参数
GPT_ARGS="
    --use-mcore-models
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --num-layers 22
    --hidden-size 2048
    --ffn-hidden-size 5632
    --num-attention-heads 32
    --group-query-attention
    --num-query-groups 4
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --seq-length 1024
    --max-position-embeddings 2048
    --micro-batch-size 16
    --global-batch-size 1024
    --make-vocab-size-divisible-by 1
    --lr 4e-4
    --train-iters 44000
    --lr-decay-style cosine
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --init-method-std 0.02
    --hidden-dropout 0.0
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --use-flash-attn
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --min-lr 1.25e-7
    --weight-decay 1e-1
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --adam-beta1 0.9
    --initial-loss-scale 65536
    --adam-beta2 0.95
    --no-gradient-accumulation-fusion
    --no-load-optim
    --no-load-rng
    --no-rope-fusion
    --overlap-grad-reduce
    --bf16
"

# CDC跨数据中心参数
CDC_ARGS="
    --use-cdc
    --cdc-parallel-size 4
    --cdc-algorithm ${ALGORITHM}
    --cdc-sync-interval 100
    --cdc-outer-lr 0.4
    --cdc-delay 3
    --cdc-num-shards 8
    --cdc-dc-N 12
    --cdc-verbose
"

DATA_ARGS="
    --data-path ${DATA_PROCESSED_PREFIX}_text_document \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 50 \
"

# 检查点加载参数
LOAD_ARGS=""
if [ -n "${CKPT_LOAD_DIR}" ] && [ -d "${CKPT_LOAD_DIR}" ]; then
    LOAD_ARGS="--load ${CKPT_LOAD_DIR}"
    echo "检测到检查点加载路径: ${CKPT_LOAD_DIR}"
fi

echo "训练日志将保存至: ${LOG_FILE}"
echo "检查点将保存至: ${CKPT_SAVE_DIR}"
echo "开始训练..."
echo

cd "${CODE_DIR}"

torchrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${CDC_ARGS} \
    ${LOAD_ARGS} \
    --distributed-backend nccl \
    --save ${CKPT_SAVE_DIR} \
    | tee ${LOG_FILE}

echo
echo "=========================================="
echo "✅ 训练流程完成"
echo "=========================================="
echo "日志文件: ${LOG_FILE}"
echo "检查点目录: ${CKPT_SAVE_DIR}"