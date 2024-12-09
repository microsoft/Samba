# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from pathlib import Path

import transformers
from transformers import AutoTokenizer

from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model
from lm_eval.__main__ import cli_evaluate
from lit_gpt.model import GPT, Config
import datasets
datasets.builder.has_sufficient_disk_space = lambda needed_bytes, directory='.': True

def load_model(checkpoint_path, config, device, dtype):
    config = Config.from_name(config)
    model = GPT(config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint["model"])
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


@register_model("Samba")
class SambaEvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained=None, config=None, max_length=4096, batch_size=1, device="cuda",
                 dtype=torch.bfloat16, trust_remote_code=True): # if not, use dtype=torch.float16/bfloat16/float32
        LM.__init__(self)
        self.backend = "causal"
        self.revision = "main"
        self.pretrained = pretrained
        self.delta = None
        self.peft = None
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = 64
        self.batch_size_per_gpu = int(batch_size)
        checkpoint_path = Path(pretrained)
        # tokenizer_name = "meta-llama/Llama-2-7b"
        tokenizer_name = "Orkhan/llama-2-7b-absa"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._model = load_model(checkpoint_path, config, device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.add_bos_token = False
        self.logits_cache = False
        self.truncation = True
        self.trust_remote_code = True
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    def _model_generate(self, context, max_length, stop, max_new_tokens=1024, temperature=0.8, top_k=200, **generation_kwargs):
        raise NotImplementedError()

if __name__ == "__main__":
    cli_evaluate()