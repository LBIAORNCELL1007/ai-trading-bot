"""
DEPRECATED: this module has been consolidated into ``tbm_labeler.py``.

The previous implementation (ATR-multiplier barriers + ternary labels {-1,0,1})
diverged from every other call site in the codebase, none of which imported it.
All TBM logic now lives in a single canonical place — ``tbm_labeler.py`` — with
symmetric ±1.5% percentage thresholds, intra-bar high/low resolution, and an
exposed ``barrier_hit_time`` column for sample-uniqueness weighting
(López de Prado §4.5).

This shim is kept solely so any forgotten import site does not silently break.
"""

from tbm_labeler import apply_triple_barrier  # noqa: F401


def triple_barrier_labels(*_args, **_kwargs):
    raise NotImplementedError(
        "tbm.triple_barrier_labels has been removed. Use "
        "tbm_labeler.apply_triple_barrier(df, tp_pct, sl_pct, time_limit) instead."
    )
