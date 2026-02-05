## 使用：
三台机子上，
1. 先create env（conda)；
2. 再download data；
3. 最后在三台机子上分别跑一个train脚本；可以H800跑train diloco，A100跑train dc；H20跑train streaming；

## 需要做的是：
1. conda环境创建完后要手动激活；
2. 把数据和训练脚本里面的WORKSPACE_ROOT替换成工作空间目录
3. 运行train_xx脚本的时候，根据机器显存调整micro batch size大小，需要能被256整除，比如8，16，32，64，128，256；

***** 
# **以下是原理。上面步骤基本足够，下面仅供参考**

## 1. 依赖
```bash
#  代码
git clone https://github.com/isaacveg/Megatron-LM.git
cd Megatron-LM
```
创建conda环境并安装依赖
```bash
conda create -n mgt python==3.10
# 安装好后激活
conda activate mgt
```
安装依赖
```bash
pip install -e .[mlm,dev]
# Transformer Engine可选，可装可不装
pip install --no-build-isolation transformer_engine[pytorch]
pip install modelscope
pip install transformers
cd ..
```

## 2. 数据集
下载数据集
```bash
mkdir data
cd data
modelscope download --dataset 'kxs712/C4' --include 'snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/**' --local_dir ./C4_en
cd ..
```

下载模型tokenizer
```bash
mkdir -p models
cd models
wget https://hf-mirror.com/PrimeIntellect/llama-1b-fresh/resolve/main/tokenizer.model
cd ../Megatron-LM
```

处理数据集
```bash
#!/bin/bash

# Define paths
INPUT_DATA="../data/C4_en/c4-train*.json.gz" # 数据集的存储路径
OUTPUT_PREFIX="../data/c4_en_train" # 转换后数据集的路径以及文件名的前缀
TOKENIZER_MODEL="../models/" # 你的模型的存储路径

# Create output directory
mkdir -p $(dirname $OUTPUT_PREFIX)

# Run preprocessing
python tools/preprocess_data.py \
    --input "$INPUT_DATA" \
    --output-prefix "$OUTPUT_PREFIX" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "$TOKENIZER_MODEL" \
    --append-eod \
    --workers 64 \
    --partitions 8 \
    --json-keys text

# Clean up intermediate partition files
echo "Cleaning up intermediate partition files..."
rm "${OUTPUT_PREFIX}"_*_text_document.bin
rm "${OUTPUT_PREFIX}"_*_text_document.idx

echo "Preprocessing complete. Output saved to ${OUTPUT_PREFIX}_text_document.bin/idx"
```

## 3. 训练
运行训练，需要设置一些路径信息等
```bash
#!/bin/bash
export NCCL_TIMEOUT=1800 
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 可使用的GPU设备
NPUS_PER_NODE=8 # 每个节点的GPU数量
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# 需要自行指定的路径
LOG_FILE=$1 # 日志保存路径
CKPT_SAVE_DIR=$2 # 检查点保存路径
DATA_PATH=$3 # 训练数据路径，即c4的路径，只要.bin和.idx之前的内容，即前缀
TOKENIZER_MODEL=$4 # 分词器模型路径, 即下载的llama-1b-fresh的tokenizer.model文件
CKPT_LOAD_DIR=$5 # 检查点加载路径, 中断后保证训练连续性

# 模型并行参数
TP=2 # 8卡4个数据中心，每个数据中心2卡张量并行
PP=1 # 流水线并行度，默认不开启

mkdir -p $CKPT_SAVE_DIR

# 分布式训练参数
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# 1b模型训练参数，micro-batch-size根据显存可调，但需要整除256
GPT_ARGS="
    --use-mcore-models
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --sequence-parallel  
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
# CDC相关参数，如果需要切换算法，可以修改--cdc-algorithm参数，支持"diloco""streaming""dc"
CDC_ARGS="
    --use-cdc
    --cdc-parallel-size 4
    --cdc-algorithm diloco
    --cdc-sync-interval 100
    --cdc-outer-lr 0.4
    --cdc-delay 3
    --cdc-num-shards 8
    --cdc-dc-N 12
    --cdc-verbose
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 50 \
"

torchrun $DISTRIBUTED_ARGS ../pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $CDC_ARGS \
    --distributed-backend nccl \
    --load $CKPT_LOAD_DIR \
    --save $CKPT_SAVE_DIR \
    | tee $LOG_FILE
```


