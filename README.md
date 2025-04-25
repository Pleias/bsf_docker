# bsf_docker

## Usage
### 1. Set up
We recommend using this app with uv as package manager
``` bash
pip install uv
uv sync
```

Download the rag gguf model and place it in the models folder:
https://huggingface.co/PleIAs/pleias_350_feb_gguf/tree/main


### Launching the app
``` python
python -m src.main
```

You can add an argument to choose if you want to use only the French, English or both version (default is "both_fts"). You can also set up debug mode (see --help for more)
``` python
python -m src.main --table-name fr_fts --debug
```