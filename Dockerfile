FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
# ↓ tell CMake to use the compilers that are already installed
ENV CC=gcc CXX=g++

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential ninja-build pkg-config libopenblas-dev git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN uv sync --frozen
RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" \
    uv pip install llama_cpp_python==0.3.8

CMD ["uv", "run", "python", "-m", "src.main", "--stream"]
