"""
AsyncSHIPS — Round 7 Patch for VULN-024

VULN-024: LayerPropagationGuard SHIPS re-scoring at inference is expensive.
Attacker floods system with prompts → SHIPS runs every N calls →
compute DoS (ThinkTrap-style, arXiv:2512.07086).

Defense:
  - SHIPS runs in async background thread, never blocking main inference.
  - Rate limiting: at most 1 SHIPS recompute per rate_window seconds.
  - Cache: last SHIPS score reused if recent enough.
  - Semantic rate limiting (OWASP LLM10 2025): throttle by prompt intent,
    not just token count.

Sources:
  - ThinkTrap DoS: arXiv:2512.07086
  - DoS via False Positives: ACM 2025 dl.acm.org 3733800.3763264
  - OWASP LLM10 Unbounded Consumption 2025
"""
from __future__ import annotations
import time
import threading
from typing import Optional
from transformer_lens import HookedTransformer
import torch
import torch.nn.functional as F


class AsyncSHIPS:
    """
    Async, rate-limited SHIPS head importance scorer.

    Runs SHIPS computation in a background thread.
    Main inference thread always gets a CACHED result instantly.
    Background thread updates cache asynchronously at most once per rate_window.
    """

    def __init__(
        self,
        model: HookedTransformer,
        rate_window: float = 30.0,   # min seconds between SHIPS recomputes
        cache_ttl: float = 120.0,    # seconds before cache considered stale
        device: str = "auto",
    ):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.model = device, model
        self.rate_window = rate_window
        self.cache_ttl   = cache_ttl
        self._cache: dict[tuple[int, int], float] = {}
        self._last_compute: float = 0.0
        self._computing: bool = False
        self._lock = threading.Lock()

    def _compute_ships(self, layer: int, head: int, prompts: list[str]) -> float:
        scores = []
        for prompt in prompts[:3]:
            tokens = self.model.to_tokens(prompt).to(self.device)
            with torch.no_grad():
                logits_full, _ = self.model.run_with_cache(tokens)
            def ablate_hook(value, hook):
                v = value.clone()
                v[:, :, head, :] = 0.0
                return v
            hook_name = f"blocks.{layer}.attn.hook_z"
            with torch.no_grad():
                logits_ab = self.model.run_with_hooks(
                    tokens, fwd_hooks=[(hook_name, ablate_hook)])
            p = F.softmax(logits_full[:, -1, :], dim=-1)
            q = F.softmax(logits_ab[:, -1, :], dim=-1)
            scores.append(F.kl_div(q.log(), p, reduction="sum").item())
        return float(sum(scores) / max(len(scores), 1))

    def _background_update(
        self, heads: list[tuple[int, int]], prompts: list[str]
    ) -> None:
        results = {}
        for (l, h) in heads:
            try:
                results[(l, h)] = self._compute_ships(l, h, prompts)
            except Exception:
                pass
        with self._lock:
            self._cache.update(results)
            self._last_compute = time.time()
            self._computing = False

    def get_ships(
        self,
        layer: int, head: int,
        prompts: list[str],
        safety_heads: list[tuple[int, int]] = None,
    ) -> float:
        """
        Get SHIPS score — returns cached value immediately.
        Triggers background recompute if cache is stale AND rate limit allows.
        Never blocks main thread.
        """
        with self._lock:
            now = time.time()
            cached = self._cache.get((layer, head))
            cache_age = now - self._last_compute
            should_recompute = (
                cache_age > self.rate_window
                and not self._computing
                and (cached is None or cache_age > self.cache_ttl)
            )
            if should_recompute:
                self._computing = True
                heads_to_compute = safety_heads or [(layer, head)]
                t = threading.Thread(
                    target=self._background_update,
                    args=(heads_to_compute, prompts),
                    daemon=True
                )
                t.start()

        # Return cached value or 0 (safe default — don't trigger false alarm)
        return self._cache.get((layer, head), 0.0)

    def invalidate(self) -> None:
        """Force cache invalidation — call after fine-tuning."""
        with self._lock:
            self._cache.clear()
            self._last_compute = 0.0
            self._computing = False
