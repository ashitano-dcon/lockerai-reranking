# Reranking Model for Locker.ai

Reranking Model for Locker.ai is a model for more accurate searches that takes into account the validity of declarations through domain-specific reevaluation.

## Core Contributors 🛠️

|                                           shio                                           |                                           ituki                                            |
| :--------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: |
| [<img src="https://github.com/dino3616.png" width="160px">](https://github.com/dino3616) | [<img src="https://github.com/ituki0426.png" width="160px">](https://github.com/ituki0426) |
|                                 `#llama-model-composer`                                  |                                   `#bert-model-composer`                                   |

## Setup with Dev Containers 📦

You can easily launch the development environment of Reranking Model for Locker.ai with Dev Containers.  
Here is the step-by-step guide.

### Attention

- You need to install [Docker](https://docs.docker.com/get-docker) and [VSCode](https://code.visualstudio.com) before.

### 1. clone git repository

```bash
git clone "https://github.com/ashitano-dcon/lockerai-reranking" && cd "./lockerai-reranking/"
```

### 2. launch dev containers

Launch containers using the VSCode extension [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).

### 3. pin python version

```bash
rye pin $(cat "./.python-version")
```

### 4. install dependencies

```bash
rye sync
```

### 5. activate virtual environment

```bash
source "./.venv/bin/activate"
```

### 6. install FlashAttention-2

```bash
uv pip install flash-attn --no-build-isolation
```

### 7. train model

```bash
rye run python -m llama.train
```

## Setup locally 🖥️

If you want to build an environment more quickly without Docker, you can follow these steps to build your environment locally.

### Attention

- You need to install [rye](https://rye.astral.sh/guide/installation) before.
- [Optional] You should install project recommended VSCode extensions that specified in [`.devcontainer/devcontainer.json`](./.devcontainer/devcontainer.json#L8C7-L16C8) before.

### 1. clone git repository

```bash
git clone "https://github.com/ashitano-dcon/lockerai-reranking" && cd "./lockerai-reranking/"
```

### 2. pin python version

```bash
rye pin $(cat "./.python-version")
```

### 3. install dependencies

```bash
rye sync
```

### 4. activate virtual environment

```bash
source "./.venv/bin/activate"
```

### 5. install FlashAttention-2

```bash
uv pip install flash-attn --no-build-isolation
```

### 6. train model

```bash
rye run python -m llama.train
```
