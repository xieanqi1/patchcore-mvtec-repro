#!/usr/bin/env bash
# 消融：聚合比例 rho 的敏感性（3 个代表类别 x 4 个比例）
# 用法： bash scripts/run_ablation.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA:-/path/to/mvtec_ad}"
OUT_DIR="${OUT:-${REPO_DIR}/results}"
GPU="${GPU:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU}}"
export TORCH_HOME="${TORCH_HOME:-${OUT_DIR}/../.cache/torch}"
export MPLBACKEND=Agg

CLASSES=(capsule screw pill)
RATIOS=(0.001 0.01 0.05 1.0)

cd "${REPO_DIR}"
mkdir -p "${OUT_DIR}/ablate"

for C in "${CLASSES[@]}"; do
    for R in "${RATIOS[@]}"; do
        echo "=== ${C} rho=${R} ==="
        PATCHCORE_TOPK_RATIO="${R}" python -u bin/run_patchcore.py \
            --gpu 0 --seed 0 \
            --log_project ablate \
            --log_group "${C}_r${R}" \
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
done

echo "完成。结果目录：${OUT_DIR}/ablate"
