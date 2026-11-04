#!/usr/bin/env bash
# 本文方法：图像级聚合改为 top-1% patch 均值
# 用法： bash scripts/run_c2.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA:-/path/to/mvtec_ad}"
OUT_DIR="${OUT:-${REPO_DIR}/results}"
GPU="${GPU:-0}"
RATIO="${RATIO:-0.01}"                 # 聚合比例 rho

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export PATCHCORE_TOPK_RATIO="${RATIO}"
export TORCH_HOME="${TORCH_HOME:-${OUT_DIR}/../.cache/torch}"
export MPLBACKEND=Agg

CLASSES=(bottle cable capsule carpet grid hazelnut leather metal_nut \
         pill screw tile toothbrush transistor wood zipper)

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}/c2"

for C in "${CLASSES[@]}"; do
    echo "=== ${C} (rho=${RATIO}) ==="
    python -u bin/run_patchcore.py \
        --gpu 0 --seed 0 \
        --save_patchcore_model \
        --save_segmentation_images \
        --log_project c2 \
        --log_group "${C}_topk01" \
        "${OUT_DIR}" \
        patch_core \
        -b wideresnet50 -le layer2 -le layer3 \
        --pretrain_embed_dimension 1024 \
        --target_embed_dimension 1024 \
        --anomaly_scorer_num_nn 1 \
        --patchsize 3 \
        --faiss_num_workers 2 \
        sampler -p 0.1 approx_greedy_coreset \
        dataset --resize 256 --imagesize 224 \
        --batch_size 2 --num_workers 2 \
        -d "${C}" mvtec "${DATA_DIR}"
done

echo "完成。结果目录：${OUT_DIR}/c2"
