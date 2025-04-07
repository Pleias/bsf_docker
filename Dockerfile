FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

 RUN apt update && apt install -y libopenblas-dev ninja-build build-essential git pkg-config \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/* /tmp/*

WORKDIR /app
COPY . /app
RUN uv sync --frozen
RUN pip list
RUN which python
RUN uv --version

RUN CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS" uv pip install llama_cpp_python

CMD ["uv", "run", "python", "-m", "src.main"]