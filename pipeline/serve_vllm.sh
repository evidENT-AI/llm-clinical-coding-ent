#!/usr/bin/env bash
# Serve a local model with vLLM as an OpenAI-compatible endpoint.
#
# Llama 3.1 8B (one GPU):
#   ./pipeline/serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct llama-3.1-8b-instruct
#
# Llama 4 Scout, 4-bit (two 32 GB GPUs, part offloaded to CPU memory):
#   CUDA_VISIBLE_DEVICES=0,1 TP=2 MAX_MODEL_LEN=49152 \
#   EXTRA_ARGS="--cpu-offload-gb 10 --enforce-eager --gpu-memory-utilization 0.92" \
#   ./pipeline/serve_vllm.sh RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16 llama-4-scout
#
# The served name must match `model` in pipeline/config.py.
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
SERVED_NAME="${2:-llama-3.1-8b-instruct}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# shellcheck disable=SC2086
exec vllm serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size "${TP:-1}" \
    --port "${PORT:-8000}" \
    --max-model-len "${MAX_MODEL_LEN:-65536}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --seed 42 \
    ${EXTRA_ARGS:-}
