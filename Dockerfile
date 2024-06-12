FROM nvcr.io/nvidia/pytorch:23.07-py3 
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
RUN pip install causal-conv1d==1.2.0.post2
RUN pip install mamba-ssm==1.2.0.post1
RUN pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
Run pip install einops
Run pip install opt_einsum
Run pip install -U git+https://github.com/sustcsonglin/flash-linear-attention