# Generic HuggingFace Embedding Service with vLLM and OpenAI-Compatible API
# Supports any embedding model from HuggingFace Hub
FROM vllm/vllm-openai:v0.6.5

# Install additional dependencies for better model support
RUN pip install --no-cache-dir \
    sentence-transformers \
    transformers>=4.51.0 \
    huggingface-hub \
    safetensors

# Set environment variables with defaults
ENV MODEL_NAME="Qwen/Qwen3-Embedding-0.6B"
ENV MODEL_REVISION="main"
ENV HOST="0.0.0.0"
ENV PORT="8080"
ENV MAX_MODEL_LEN="8192"
ENV MAX_NUM_SEQS="16"
ENV MAX_BATCH_TOKENS="4096"
ENV TENSOR_PARALLEL_SIZE="1"
ENV GPU_MEMORY_UTILIZATION="0.8"
ENV TRUST_REMOTE_CODE="true"
ENV SERVED_MODEL_NAME=""

# Resource optimization for CPU/minimal GPU deployments  
ENV CUDA_VISIBLE_DEVICES=""
ENV VLLM_WORKER_MULTIPROC_METHOD="spawn"
ENV TOKENIZERS_PARALLELISM="false"

# Use dedicated HF cache location, configurable via HF_HOME
ENV HF_HOME="/cache/huggingface"

# Create app directory
WORKDIR /app

# Copy startup script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create non-root user and cache directory
RUN useradd -m -u 10002 embeduser && \
    mkdir -p ${HF_HOME} && \
    chown -R embeduser:embeduser ${HF_HOME} /app
USER 10002:10002

# Expose port
EXPOSE 8080

# Health check with model-agnostic endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]