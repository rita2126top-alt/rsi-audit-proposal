"""Offline Hugging Face inference. Importing the package does not import torch."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from importlib.metadata import version

from .common import BudgetExceeded


@dataclass(frozen=True)
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    seconds: float


class TransformersBackend:
    def __init__(self, model_id: str, revision: str, cache_dir: str | None = None, device: str = "cuda:0"):
        if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("A full model revision is required")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise RuntimeError("Install the gpu extra and prefetch the pinned model before starting a real run") from exc
        if transformers.__version__ != "4.51.3" or torch.__version__.split("+")[0] != "2.6.0":
            raise RuntimeError("This adapter requires transformers==4.51.3 and torch==2.6.0; re-register before changing them")
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("The real inference adapter requires a CUDA GPU; use demo for CPU workflow checks")
        self.torch, self.device = torch, device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, local_files_only=True, trust_remote_code=False,
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, cache_dir=cache_dir, local_files_only=True,
            trust_remote_code=False, use_safetensors=True, torch_dtype=torch.bfloat16,
        ).to(device).eval()
        if not self.tokenizer.chat_template:
            raise RuntimeError("Pinned tokenizer does not provide a chat template")
        self.gpu_count = 1
        self.identity = {
            "kind": "transformers", "model_id": model_id, "revision": revision,
            "transformers": transformers.__version__, "torch": torch.__version__,
            "tokenizers": version("tokenizers"), "cuda_runtime": torch.version.cuda,
            "device": device, "gpu_name": torch.cuda.get_device_name(device),
            "dtype": "bfloat16", "local_files_only": True, "trust_remote_code": False,
        }

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, messages: list[dict], *, max_input_tokens: int, max_new_tokens: int,
                 temperature: float, top_p: float, seed: int, max_seconds: float | None = None) -> Generation:
        if max_new_tokens <= 0 or max_input_tokens <= 0:
            raise BudgetExceeded("No generation allowance remains")
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        count = inputs["input_ids"].shape[-1]
        if count > max_input_tokens:
            raise BudgetExceeded(f"Input would use {count} tokens; only {max_input_tokens} remain")
        context_limit = int(getattr(self.model.config, "max_position_embeddings", 32768))
        if count + max_new_tokens > context_limit:
            raise BudgetExceeded("Requested generation exceeds the model context window")
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        kwargs = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0, "num_beams": 1,
                  "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                  "use_cache": True}
        if temperature > 0:
            kwargs.update(temperature=temperature, top_p=top_p)
        if max_seconds is not None:
            if max_seconds <= 0:
                raise BudgetExceeded("No phase time remains")
            kwargs["max_time"] = max_seconds
        self.torch.cuda.synchronize(self.device)
        start = time.monotonic()
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, **kwargs)
        self.torch.cuda.synchronize(self.device)
        seconds = time.monotonic() - start
        tokens = result[0, count:]
        return Generation(self.tokenizer.decode(tokens, skip_special_tokens=True), int(count), int(tokens.shape[0]), seconds)
