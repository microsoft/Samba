FROM nvcr.io/nvidia/pytorch:23.07-py3 
RUN pip install --upgrade pip
RUN pip install --user azureml-mlflow tensorboard
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip install packaging torch>=2.1.0dev lightning==2.1.2 lightning[app]
RUN pip install jsonargparse[signatures] tokenizers sentencepiece wandb lightning[data] torchmetrics 
RUN pip install tensorboard zstandard pandas pyarrow huggingface_hub 
RUN pip install -U flash-attn --no-build-isolation
RUN git clone https://github.com/Dao-AILab/flash-attention
WORKDIR flash-attention
WORKDIR csrc/rotary 
RUN pip install .
WORKDIR ../layer_norm 
RUN pip install .
WORKDIR ../xentropy
RUN pip install .
RUN pip install causal-conv1d
RUN pip install mamba-ssm
RUN pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
RUN pip install einops
RUN pip install opt_einsum
RUN pip install -U git+https://github.com/sustcsonglin/flash-linear-attention@98c176e
