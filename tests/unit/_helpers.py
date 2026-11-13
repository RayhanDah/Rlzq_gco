"""Outils de test partagés par `tests/unit`.

`assert_close` remplace `torch.testing.assert_close` : le premier appel de ce dernier importe `torch.distributed.tensor`
et `sympy` (mesuré : 7,0 s sur le laptop, 2026-11-13, `python -c` avec `time.perf_counter` autour du premier appel),
soit plus de 10 % du budget de la suite rapide. Même contrôle : forme, dtype, device identiques, puis
|actual − expected| ≤ atol + rtol·|expected| élément par élément (NaN jamais égaux), via `torch.allclose`.
"""

from __future__ import annotations

import torch


def assert_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float) -> None:
    """Égalité à tolérance explicite (pas de défaut : chaque appel écrit sa tolérance)."""
    assert actual.shape == expected.shape, f"forme {tuple(actual.shape)} ≠ {tuple(expected.shape)}"
    assert actual.dtype == expected.dtype, f"dtype {actual.dtype} ≠ {expected.dtype}"
    assert actual.device == expected.device, f"device {actual.device} ≠ {expected.device}"
    assert torch.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False), (
        f"écart max {float((actual - expected).abs().max())} (atol={atol}, rtol={rtol})\n{actual}\n{expected}"
    )
