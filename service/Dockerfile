FROM python:3.11-slim

WORKDIR /app
# torch (a sentence-transformers dep) is large; CPU wheels are the linux default — no CUDA needed for L1/L2.
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir -r service/requirements.txt

# The service imports the `quorum` library, so ship both.
COPY quorum/ ./quorum/
COPY service/ ./service/

# Layer 1 defaults to the calibration-free zero-shot ensemble (PrismNLI + bge-large + bge-base). Set DEVICE=cuda
# on a GPU box. Layer 1.5 (24-shot anchor) additionally needs RLCD_ADAPTER access + a GPU.
ENV REGISTRY_DIR=/data/registry \
    HF_HOME=/data/hf \
    DEVICE=cpu \
    EMBED_MODEL=BAAI/bge-base-en-v1.5

EXPOSE 8080
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8080"]
