#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <dataset_name> <scale>"
    exit 1
fi

dataset_name="$1"
scale="$2"
dataset_folder="data/$dataset_name"

if [ ! -d "$dataset_folder" ]; then
    echo "Error: Folder '$dataset_folder' does not exist."
    exit 2
fi

# 关键：减少显存碎片/大块分配失败概率
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

cd Tracking-Anything-with-DEVA/

if [ "$scale" = "1" ]; then
    img_path="../data/${dataset_name}/images"
else
    img_path="../data/${dataset_name}/images_${scale}"
fi

# 关键：降低分辨率 + 降chunk（chunk越大越吃显存）
SIZE=384          # 如果还OOM，改成 320
CHUNK=2           # 如果还OOM，改成 1

COMMON_ARGS=(
  --chunk_size "$CHUNK"
  --img_path "$img_path"
  --amp
  --temporal_setting semionline
  --size "$SIZE"
  --output "./example/output_gaussian_dataset/${dataset_name}"
  --suppress_small_objects
  --SAM_PRED_IOU_THRESHOLD 0.7
)

# 1) colored mask
python demo/demo_automatic.py "${COMMON_ARGS[@]}"

# 如果第一次失败，Annotations 不存在，这里会报错；所以加判断更稳
if [ -d "./example/output_gaussian_dataset/${dataset_name}/Annotations" ]; then
  mv ./example/output_gaussian_dataset/${dataset_name}/Annotations \
     ./example/output_gaussian_dataset/${dataset_name}/Annotations_color
fi

# 2) gray mask
python demo/demo_automatic.py "${COMMON_ARGS[@]}" --use_short_id

# 3) copy gray mask
mkdir -p ../data/${dataset_name}/object_mask
cp -r ./example/output_gaussian_dataset/${dataset_name}/Annotations \
      ../data/${dataset_name}/object_mask

cd ..
