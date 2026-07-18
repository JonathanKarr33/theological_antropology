"""Model backends: local MLX (Qwen / Llama / Phi-4 Mini).

Local models run via mlx_lm on Apple Silicon.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model_id: str
    backend: str
    latency_s: float
    error: Optional[str] = None


# Default Apple-Silicon 4-bit MLX community builds (~3–5 GB each on disk).
DEFAULT_LOCAL_MODELS: dict[str, str] = {
    "qwen7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "llama": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "phi4mini": "mlx-community/Phi-4-mini-instruct-4bit",
}

# Smaller / faster fallbacks if the 7–9B downloads are too heavy.
FALLBACK_LOCAL_MODELS: dict[str, str] = {
    "qwen7b": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "llama": "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "phi4mini": "mlx-community/Phi-4-mini-instruct-4bit",
}


class MlxBackend:
    """Lazy-load one MLX model; reuse across many prompts."""

    def __init__(self, model_id: str, *, max_tokens: int = 512, temp: float = 1.0):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temp = temp
        self._model = None
        self._tokenizer = None
        self._sampler = None

    def load(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler

        print(f"[mlx] Loading {self.model_id} ...", flush=True)
        t0 = time.time()
        self._model, self._tokenizer = load(self.model_id)
        self._sampler = make_sampler(temp=self.temp)
        print(f"[mlx] Loaded in {time.time() - t0:.1f}s", flush=True)

    def unload(self) -> None:
        """Free Metal weights before loading another model (avoids OOM)."""
        if self._model is None and self._tokenizer is None:
            return
        print(f"[mlx] Unloading {self.model_id} ...", flush=True)
        self._model = None
        self._tokenizer = None
        self._sampler = None
        import gc

        gc.collect()
        try:
            import mlx.core as mx

            clear = getattr(getattr(mx, "metal", None), "clear_cache", None)
            if callable(clear):
                clear()
        except Exception:
            pass

    def generate(self, prompt: str) -> GenerationResult:
        from mlx_lm import generate

        self.load()
        assert self._model is not None and self._tokenizer is not None
        tok = self._tokenizer
        if hasattr(tok, "apply_chat_template"):
            try:
                formatted = tok.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                formatted = prompt
        else:
            formatted = prompt

        t0 = time.time()
        try:
            text = generate(
                self._model,
                tok,
                prompt=formatted,
                max_tokens=self.max_tokens,
                sampler=self._sampler,
                verbose=False,
            )
            # mlx_lm.generate often returns the full prompt+completion; strip prompt if present.
            if isinstance(text, str) and text.startswith(formatted):
                text = text[len(formatted) :]
            return GenerationResult(
                text=(text or "").strip(),
                model_id=self.model_id,
                backend="mlx",
                latency_s=time.time() - t0,
            )
        except Exception as e:
            return GenerationResult(
                text="",
                model_id=self.model_id,
                backend="mlx",
                latency_s=time.time() - t0,
                error=str(e),
            )


def build_backend(name: str, *, max_tokens: int = 512, temp: float = 1.0, small: bool = False) -> Any:
    """name: one of qwen7b | llama | phi4mini"""
    key = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "phi4": "phi4mini",
        "phi4miniinstruct": "phi4mini",
    }
    key = aliases.get(key, key)
    table = FALLBACK_LOCAL_MODELS if small else DEFAULT_LOCAL_MODELS
    if key not in table:
        raise ValueError(f"Unknown model '{name}'. Choose from: qwen7b, llama, phi4mini")
    return MlxBackend(table[key], max_tokens=max_tokens, temp=temp)
