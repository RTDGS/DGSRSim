#!/usr/bin/env bash
set -euo pipefail

# One-shot pipeline:
# 1) filter/crop Gaussian PLY while preserving all 3DGS attributes
# 2) convert the filtered Gaussian PLY to a visual USDZ with 3DGRUT
# 3) create a 3DGRUT-compatible triangle mesh PLY for collision/proxy geometry
# 4) add the mesh PLY into the visual USDZ
#
# Usage:
#   bash tools/run_3dgrut_usdz_collision_pipeline.sh \
#     /media/ubuntu/L/output/hu/point_cloud_object_removal/iteration_10000/point_cloud.ply
#
# Optional second argument:
#   output directory. Defaults to the input PLY directory.
#
# Useful environment overrides:
#   PYTHON_BIN=python
#   FILTER_MIN_OPACITY=0.005
#   FILTER_MODE=cluster-distance
#   FILTER_CLUSTER_MARGIN=0.12
#   FILTER_CLUSTER_INCLUDE_RADIUS=0.12
#   MESH_MIN_OPACITY=0.16
#   SIMPLIFY_TARGET_FACES=25000
#   SET_MESH_INVISIBLE=0
#   SKIP_FILTER=0

export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
export CUDAHOSTCXX=/usr/bin/g++-12

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_PLY="${1:-/media/ubuntu/L/output/hu/point_cloud_object_removal/iteration_10000/point_cloud.ply}"
OUT_DIR="${2:-$(dirname "${INPUT_PLY}")}"

FILTERED_PLY="${OUT_DIR}/point_cloud_filtered.ply"
VISUAL_USDZ="${OUT_DIR}/point_cloud.usdz"
MESH_PLY="${OUT_DIR}/point_cloud_mesh.ply"
OUTPUT_USDZ="${OUT_DIR}/point_cloudoutput.usdz"

FILTER_MIN_OPACITY="${FILTER_MIN_OPACITY:-0.005}"
FILTER_MODE="${FILTER_MODE:-cluster-distance}"
FILTER_CLUSTER_MARGIN="${FILTER_CLUSTER_MARGIN:-0.12}"
FILTER_CLUSTER_INCLUDE_RADIUS="${FILTER_CLUSTER_INCLUDE_RADIUS:-0.12}"
MESH_MIN_OPACITY="${MESH_MIN_OPACITY:-0.16}"
SIMPLIFY_TARGET_FACES="${SIMPLIFY_TARGET_FACES:-25000}"
SET_MESH_INVISIBLE="${SET_MESH_INVISIBLE:-0}"
SKIP_FILTER="${SKIP_FILTER:-0}"

mkdir -p "${OUT_DIR}"

echo "[env] CC=${CC}"
echo "[env] CXX=${CXX}"
echo "[env] CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "[input] ${INPUT_PLY}"
echo "[out_dir] ${OUT_DIR}"

if [[ ! -f "${INPUT_PLY}" ]]; then
  echo "[error] input PLY not found: ${INPUT_PLY}" >&2
  exit 1
fi

echo
echo "== Step 1/4: filter Gaussian PLY =="
if [[ "${SKIP_FILTER}" == "1" || "${SKIP_FILTER}" == "true" || "${SKIP_FILTER}" == "TRUE" ]]; then
  FILTERED_PLY="${INPUT_PLY}"
  echo "[filter] skipped; using input PLY directly"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/crop_gaussian_ply.py" \
    "${INPUT_PLY}" \
    --output "${FILTERED_PLY}" \
    --min-opacity "${FILTER_MIN_OPACITY}" \
    --filter-mode "${FILTER_MODE}" \
    --cluster-margin "${FILTER_CLUSTER_MARGIN}" \
    --cluster-include-radius "${FILTER_CLUSTER_INCLUDE_RADIUS}"
fi

echo
echo "== Step 2/4: convert filtered Gaussian PLY to visual USDZ =="
"${PYTHON_BIN}" -m threedgrut.export.scripts.ply_to_usd \
  "${FILTERED_PLY}" \
  --output_file "${VISUAL_USDZ}"

echo
echo "== Step 3/4: generate 3DGRUT-compatible triangle mesh PLY =="
"${PYTHON_BIN}" "${SCRIPT_DIR}/gaussian_ply_to_3dgrut_mesh_ply.py" \
  "${FILTERED_PLY}" \
  --output "${MESH_PLY}" \
  --simplify-target-faces "${SIMPLIFY_TARGET_FACES}" \
  --min-opacity "${MESH_MIN_OPACITY}" \
  --no-auto-crop-largest-cluster

echo
echo "== Step 4/4: add mesh PLY into USDZ =="
ADD_ARGS=(
  --input_usdz "${VISUAL_USDZ}"
  --output_usdz "${OUTPUT_USDZ}"
  --mesh_ply "${MESH_PLY}"
  --set_collision
)

if [[ "${SET_MESH_INVISIBLE}" == "1" || "${SET_MESH_INVISIBLE}" == "true" || "${SET_MESH_INVISIBLE}" == "TRUE" ]]; then
  ADD_ARGS+=(--set_invisible)
fi

"${PYTHON_BIN}" -m threedgrut.export.scripts.add_mesh_to_usdz "${ADD_ARGS[@]}"

echo
echo "== Done =="
echo "[filtered_ply] ${FILTERED_PLY}"
echo "[visual_usdz]  ${VISUAL_USDZ}"
echo "[mesh_ply]     ${MESH_PLY}"
echo "[output_usdz]  ${OUTPUT_USDZ}"
