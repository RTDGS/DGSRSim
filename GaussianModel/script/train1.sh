#!/bin/bash

echo "Legacy training snapshot disabled. Use script/train.sh." >&2
exit 2

# Check if the user provided an argument
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name>"
    exit 1
fi

dataset_name="$1"
scale="$2"
dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi

# --- 关键修改部分 Start ---
# Gaussian Grouping training (双 GPU 版本)
# --nproc_per_node=2 表示使用 2 个 GPU
# --master_port 是为了防止端口冲突，可选
torchrun --nproc_per_node=2 --master_port=29500 train_multi_GPU.py \
    -s "$dataset_folder" \
    -r "${scale}" \
    -m "output/${dataset_name}" \
    --config_file "config/gaussian_dataset/train.json"
# --- 关键修改部分 End ---

# Segmentation rendering using trained model
# 渲染通常不需要多卡，单卡运行即可（使用 GPU 0）
python render.py -m "output/${dataset_name}" --num_classes 256
