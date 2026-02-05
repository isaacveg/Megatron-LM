#!/bin/bash
#==============================================================================
WORKSPACE_ROOT="XXX"           # 工作空间根目录
CODE_DIR="${WORKSPACE_ROOT}/Megatron-LM"

echo "=========================================="
echo "克隆Megatron-LM代码仓库"
echo "=========================================="
if [ -d "${CODE_DIR}" ]; then
    echo "跳过代码克隆步骤"
else
    mkdir -p "$(dirname ${CODE_DIR})"
    git clone https://github.com/isaacveg/Megatron-LM "${CODE_DIR}"
fi
echo "✅ 代码仓库准备完成"
echo

# 安装PyTorch及相关库
echo "正在安装PyTorch 2.7.0"
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 安装Megatron-LM
echo "正在安装megatron-core..."
cd "${CODE_DIR}"
pip install -e .[mlm,dev]

# 安装nccl
echo "正在安装NCCL..."
conda install -c nvidia -y nccl

# 安装Transformer Engine
echo "正在安装Transformer Engine..."
pip install --no-build-isolation transformer_engine[pytorch]

# 安装其他依赖
echo "正在安装额外依赖包..."
pip install transformers
pip install psutil
pip install modelscope

echo
echo "=========================================="
echo "✅ 环境配置完成"
echo "=========================================="
