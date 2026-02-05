#!/bin/bash

WORKSPACE_ROOT="XXX"           # 工作空间根目录
CODE_DIR="${WORKSPACE_ROOT}/Megatron-LM"         # 代码存放目录
TOKENIZER_DIR="${WORKSPACE_ROOT}/tokenizer"      # 分词器存放目录
DATA_RAW_DIR="${WORKSPACE_ROOT}/data/raw/C4_en"  # 原始数据存放目录
DATA_PROCESSED_PREFIX="${WORKSPACE_ROOT}/data/processed/c4_en_train"  # 处理后数据前缀



echo "=========================================="
echo "下载llama-1b-fresh分词器"
echo "=========================================="

mkdir -p "${TOKENIZER_DIR}"
TOKENIZER_MODEL="${TOKENIZER_DIR}/llama-1b-fresh-tokenizer.model"

if [ -f "${TOKENIZER_MODEL}" ]; then
    echo "⚠️  分词器文件已存在: ${TOKENIZER_MODEL}"
else
    echo "正在下载分词器模型..."
    wget -O "${TOKENIZER_MODEL}" \
        https://hf-mirror.com/PrimeIntellect/llama-1b-fresh/resolve/main/tokenizer.model
fi

echo "✅ 分词器准备完成"
echo

echo "=========================================="
echo "C4数据集下载与预处理"
echo "=========================================="

# 检查是否已存在处理后的数据
PROCESSED_BIN="${DATA_PROCESSED_PREFIX}_text_document.bin"
PROCESSED_IDX="${DATA_PROCESSED_PREFIX}_text_document.idx"

if [ -f "${PROCESSED_BIN}" ] && [ -f "${PROCESSED_IDX}" ]; then
    echo "✅ 检测到已处理完成的数据文件:"
    echo "   ${PROCESSED_BIN}"
    echo "   ${PROCESSED_IDX}"
    echo "跳过数据下载与预处理阶段"
else

    # 下载C4-en数据集
    mkdir -p "${DATA_RAW_DIR}"
    echo "正在下载C4-en数据集..."
    modelscope download --dataset 'kxs712/C4' \
        --include 'snapshots/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/**' \
        --local_dir "${DATA_RAW_DIR}"

    # Megatron格式预处理
    echo "正在执行Megatron数据预处理..."
    mkdir -p "$(dirname ${DATA_PROCESSED_PREFIX})"
    cd "${CODE_DIR}"

    python tools/preprocess_data.py \
        --input "${DATA_RAW_DIR}/c4-train*.json.gz" \
        --output-prefix "${DATA_PROCESSED_PREFIX}" \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model "${TOKENIZER_DIR}" \
        --append-eod \
        --workers 64 \
        --partitions 8 \
        --json-keys text

    # 清理中间分区文件
    echo "清理中间分区文件..."
    rm -f "${DATA_PROCESSED_PREFIX}"_*_text_document.bin
    rm -f "${DATA_PROCESSED_PREFIX}"_*_text_document.idx

    # 删除原始数据以节省存储空间
    echo "正在删除原始数据以释放存储空间..."
    rm -rf "${DATA_RAW_DIR}"
    echo "✅ 原始数据已删除: ${DATA_RAW_DIR}"
    echo "✅ 数据预处理完成"
    echo "  处理后数据位置: ${DATA_PROCESSED_PREFIX}_text_document.{bin,idx}"
fi

echo