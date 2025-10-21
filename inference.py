#
# Use this tokenizer
# Meta-Llama-3-8B-Instruct/tokenizer.model
#

from dataclasses import dataclass
from typing import List
import torch

from model import Transformer

if torch.cuda.is_bf16_supported():
    torch.set_default_tensor_type(torch.cuda.BFloat16Storage)  # type: ignore


@dataclass
class GenerateToken:
    token: str
    logprob: float


def generate(
    prompts: List[str],
    model: Transformer,
    temperature: float = 1,
    max_gen_len: int | None = None,
):
    # build tokenizer
    # determine prefill/ decode correct lengths
    pass
