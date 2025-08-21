#!/bin/bash

# Generic HuggingFace Embedding Service Entrypoint
# Starts vLLM server with OpenAI-compatible embedding API for any HuggingFace model

set -e

# Default values with environment variable fallbacks
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-Embedding-0.6B"}
MODEL_REVISION=${MODEL_REVISION:-"main"}
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-"8080"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"8192"}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-"16"}
MAX_BATCH_TOKENS=${MAX_BATCH_TOKENS:-"4096"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-"0.8"}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-"true"}

# Auto-generate served model name if not provided
if [ -z "$SERVED_MODEL_NAME" ]; then
    # Extract model name and convert to lowercase with hyphens
    SERVED_MODEL_NAME=$(echo "${MODEL_NAME}" | tr '/' '-' | tr '[:upper:]' '[:lower:]')
fi

echo "🚀 Starting HuggingFace Embedding Service with vLLM"
echo "📋 Model: ${MODEL_NAME}@${MODEL_REVISION}"
echo "🌐 Host: ${HOST}:${PORT}"
echo "🏷️  Served as: ${SERVED_MODEL_NAME}"
echo "⚙️  Mode: ${CUDA_VISIBLE_DEVICES:-'CPU-only'}"
echo "🧠 Trust Remote Code: ${TRUST_REMOTE_CODE}"

# Validate model exists (optional pre-check)
echo "🔍 Validating model availability..."
python3 -c "
from huggingface_hub import model_info
try:
    info = model_info('${MODEL_NAME}', revision='${MODEL_REVISION}')
    print(f'✅ Model {MODEL_NAME} validated successfully')
    print(f'   Repository: {info.id}')
    print(f'   Downloads: {info.downloads}')
    if hasattr(info, 'pipeline_tag'):
        print(f'   Pipeline: {info.pipeline_tag}')
except Exception as e:
    print(f'⚠️  Model validation warning: {e}')
    print(f'   Proceeding anyway - vLLM will attempt to load the model')
"

echo "🎯 Starting vLLM server..."

# Build the command arguments
ARGS=(
    --model "${MODEL_NAME}"
    --revision "${MODEL_REVISION}"
    --host "${HOST}"
    --port "${PORT}"
    --task "embed"
    --device cpu
    --disable-log-stats
    --disable-log-requests
    --disable-async-output-proc
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-num-batched-tokens "${MAX_BATCH_TOKENS}"
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
    --pipeline-parallel-size 1
    --block-size 16
    --swap-space 4
    --enforce-eager
    --disable-sliding-window
    --api-key "embedding-service-key"
    --served-model-name "${SERVED_MODEL_NAME}"
)

# Add trust-remote-code flag only if enabled
if [ "${TRUST_REMOTE_CODE}" = "true" ]; then
    ARGS+=(--trust-remote-code)
fi

# Start vLLM server with configurable parameters
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"