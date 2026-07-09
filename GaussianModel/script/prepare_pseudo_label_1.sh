#!/bin/bash

# Check arguments
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

cd Tracking-Anything-with-DEVA/

# Image path
if [ "$scale" = "1" ]; then
    img_path="../data/${dataset_name}/images"
else
    img_path="../data/${dataset_name}/images_${scale}"
fi

###############################################################
# Step 1 — colored mask on GPU 0 (SAM-H, high quality)
###############################################################
CUDA_VISIBLE_DEVICES=0 python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset/${dataset_name}" \
  --suppress_small_objects \
  --SAM_PRED_IOU_THRESHOLD 0.7 \
  --sam_variant original \
  --SAM_NUM_POINTS_PER_SIDE 32

mv ./example/output_gaussian_dataset/${dataset_name}/Annotations \
   ./example/output_gaussian_dataset/${dataset_name}/Annotations_color

###############################################################
# Step 2 — gray mask on GPU 1 (SAM-H, high quality)
###############################################################
CUDA_VISIBLE_DEVICES=1 python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path "$img_path" \
  --amp \
  --temporal_setting semionline \
  --size 480 \
  --output "./example/output_gaussian_dataset/${dataset_name}" \
  --use_short_id \
  --suppress_small_objects \
  --SAM_PRED_IOU_THRESHOLD 0.7 \
  --sam_variant original \
  --SAM_NUM_POINTS_PER_SIDE 32

###############################################################
# Step 3 — copy outputs
###############################################################
cp -r ./example/output_gaussian_dataset/${dataset_name}/Annotations \
      ../data/${dataset_name}/object_mask

cd ..
echo "Pseudo-label generation COMPLETE!"
