"""
backend/reguaz/retrieval/fusion.py

Reciprocal Rank Fusion (RRF) implementation.

Reference
---------
Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
"Reciprocal rank fusion outperforms condorcet and individual rank learning
methods."  SIGIR '09, pp. 758–759.

Both public functions are **pure** — they have no side effects, take no
optional mutable state, and always return new objects.  This makes them
trivial to unit-test and safe to call concurrently.

Usage example
-------------
    from backend.reguaz.retrieval.fusion import reciprocal_rank_fusion

    semantic_ids = ["c1", "c3", "c2", "c4"]
    bm25_ids     = ["c2", "c1", "c5", "c3"]

    fused = reciprocal_rank_fusion([semantic_ids, bm25_ids], k=60)
    # fused → ["c1", "c2", "c3", ...]
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rrf_scores(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    Compute a raw RRF score for every document ID that appears in at least
    one of the supplied ranked lists.

    The RRF score of document *d* is defined as::

        score(d) = Σ  1 / (k + rank(d, list))
                  lists

    where ``rank(d, list)`` is the 1-based position of *d* in *list*
    (IDs that do not appear in a list contribute 0 from that list).

    Parameters
    ----------
    ranked_lists : list[list[str]]
        Each inner list is an ordered sequence of document / chunk IDs,
        position 0 being rank 1 (highest relevance).
    k : int
        The RRF smoothing constant.  The canonical value from the original
        paper is 60.  A smaller *k* amplifies high-rank differences;
        a larger *k* smooths them out.

    Returns
    -------
    dict[str, float]
        Mapping of ``{document_id: rrf_score}``.  All IDs that appear in
        any of the input lists are included.
    """
    if not ranked_lists:
        logger.warning("compute_rrf_scores: received empty ranked_lists — returning {}.")
        return {}

    scores: dict[str, float] = {}

    for list_index, ranked_list in enumerate(ranked_lists):
        if not ranked_list:
            logger.debug(
                "compute_rrf_scores: ranked_list[%d] is empty — skipped.", list_index
            )
            continue

        for rank_zero_based, doc_id in enumerate(ranked_list):
            # Convert to 1-based rank for the RRF formula.
            rank_one_based = rank_zero_based + 1
            contribution = 1.0 / (k + rank_one_based)
            scores[doc_id] = scores.get(doc_id, 0.0) + contribution

    logger.debug(
        "compute_rrf_scores: scored %d unique document(s) from %d ranked list(s).",
        len(scores),
        len(ranked_lists),
    )
    return scores


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[str]:
    """
    Merge multiple ranked lists into a single consensus ranking using RRF.

    Internally calls :func:`compute_rrf_scores` and sorts the result in
    descending score order.

    Parameters
    ----------
    ranked_lists : list[list[str]]
        Each inner list is an ordered sequence of document / chunk IDs,
        position 0 being rank 1 (highest relevance).
    k : int
        The RRF smoothing constant (default 60).

    Returns
    -------
    list[str]
        Merged, deduplicated list of document / chunk IDs ordered by
        descending RRF score (position 0 = highest RRF score).
    """
    scores = compute_rrf_scores(ranked_lists, k=k)

    if not scores:
        return []

    # Sort by score descending; break ties lexicographically for determinism.
    fused: list[str] = sorted(
        scores.keys(),
        key=lambda doc_id: (-scores[doc_id], doc_id),
    )

    logger.debug(
        "reciprocal_rank_fusion: merged %d unique document(s), k=%d.",
        len(fused),
        k,
    )
    return fused
