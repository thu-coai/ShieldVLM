#!/bin/bash

set -e

# --- 配置 ---
# 配置文件的路径
CONFIG_FILE="train_code/train_shieldvlm.yaml"
export WANDB_DISABLED="true"


# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误：配置文件不存在于 '${CONFIG_FILE}'"
    exit 1
fi

llamafactory-cli train ${CONFIG_FILE}

