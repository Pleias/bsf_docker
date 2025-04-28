# bsf_docker

## Usage
### 1. Set up
We recommend using this app with uv as package manager
``` bash
pip install uv
uv sync
```

Download the RAG GGUF model you want and place it in the models folder (we recommend the 350m version):
- https://huggingface.co/PleIAs/Pleias-RAG-350M-gguf
- https://huggingface.co/PleIAs/Pleias-RAG-1B-gguf 



### Launching the app
``` bash
python -m src.main
```

CLI arguments:
You can add an argument to choose if you want to use only the French, English or both versions (default is both. You can also set up debug mode and stream mode, and control the host and port from there (see --help for more)
``` bash
python -m src.main -t both_fts --debug --stream --p 8081
```