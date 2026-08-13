from __future__ import annotations

from typing import Any

from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore
from backend.reguaz.services.generation.v2_models import (
    EvidencePath,
    ExpansionDiagnostic,
    V2GenerationSettings,
)


RELATION_PRIORITY = {
    "continuation_of": 2,
    "continuation": 2,
    "parent": 3,
    "previous_sibling": 4,
    "next_sibling": 4,
    "reference": 5,
    "referenced_by": 5,
    "annex_parent": 6,
    "table_parent": 6,
    "approved_document": 7,
}


class RelationAwareExpander:
    """Bounded, deterministic depth-one expansion from final reranked seeds."""

    def __init__(
        self,
        artifacts: V2ArtifactStore,
        settings: V2GenerationSettings,
    ) -> None:
        self.artifacts = artifacts
        self.settings = settings

    def expand(
        self, final_results: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[ExpansionDiagnostic]]:
        candidates: list[dict[str, Any]] = []
        diagnostics: list[ExpansionDiagnostic] = []
        expanded_by_id: dict[str, dict[str, Any]] = {}

        for result in sorted(final_results, key=lambda item: item["final_rank"]):
            payload = dict(result.get("payload") or result)
            seed_id = str(payload["chunk_id"])
            candidates.append(
                {
                    "artifact_type": "child",
                    "record": payload,
                    "role": "seed",
                    "relation_type": None,
                    "priority": 1,
                    "seed_rank": int(result["final_rank"]),
                    "reranker_score": result.get("reranker_score"),
                    "selection_reason": "final-five reranked seed",
                    "provenance": [
                        EvidencePath(
                            seed_chunk_id=seed_id,
                            expansion_depth=0,
                            selection_reason="final-five reranked seed",
                        )
                    ],
                }
            )
            inspected = self._inspect_seed(result, diagnostics)
            inspected.sort(
                key=lambda item: (
                    item["priority"],
                    self._artifact_id(item["record"], item["artifact_type"]),
                )
            )
            included = inspected[: self.settings.expansion_per_seed_cap]
            for rejected in inspected[self.settings.expansion_per_seed_cap :]:
                diagnostics.append(
                    ExpansionDiagnostic(
                        seed_chunk_id=seed_id,
                        relation_type=rejected["relation_type"],
                        target_id=self._artifact_id(
                            rejected["record"], rejected["artifact_type"]
                        ),
                        decision="excluded",
                        reason="per-seed expansion cap",
                    )
                )
            for item in included:
                artifact_id = self._artifact_id(item["record"], item["artifact_type"])
                existing = expanded_by_id.get(artifact_id)
                if existing is None:
                    expanded_by_id[artifact_id] = item
                else:
                    existing["provenance"].extend(item["provenance"])

        expanded = sorted(
            expanded_by_id.values(),
            key=lambda item: (
                item["priority"],
                item["seed_rank"],
                self._artifact_id(item["record"], item["artifact_type"]),
            ),
        )
        accepted = expanded[: self.settings.expansion_global_cap]
        for rejected in expanded[self.settings.expansion_global_cap :]:
            diagnostics.append(
                ExpansionDiagnostic(
                    seed_chunk_id=rejected["provenance"][0].seed_chunk_id,
                    relation_type=rejected["relation_type"],
                    target_id=self._artifact_id(
                        rejected["record"], rejected["artifact_type"]
                    ),
                    decision="excluded",
                    reason="global expansion cap",
                )
            )
        candidates.extend(accepted)
        return candidates, diagnostics

    def _inspect_seed(
        self,
        result: dict[str, Any],
        diagnostics: list[ExpansionDiagnostic],
    ) -> list[dict[str, Any]]:
        seed = dict(result.get("payload") or result)
        seed_id = str(seed["chunk_id"])
        seed_rank = int(result["final_rank"])
        found: list[dict[str, Any]] = []

        def add(
            relation_type: str,
            target_id: str | None,
            *,
            direction: str = "structural",
            confidence: float | None = None,
            supplied: dict[str, Any] | None = None,
        ) -> None:
            if not target_id:
                return
            resolved = self._resolve_target(target_id, supplied=supplied)
            if resolved is None:
                diagnostics.append(
                    ExpansionDiagnostic(
                        seed_chunk_id=seed_id,
                        relation_type=relation_type,
                        target_id=target_id,
                        decision="excluded",
                        reason="relation target missing",
                    )
                )
                return
            artifact_type, record = resolved
            reason = f"depth-one {relation_type} context"
            found.append(
                {
                    "artifact_type": artifact_type,
                    "record": record,
                    "role": "expanded",
                    "relation_type": relation_type,
                    "priority": RELATION_PRIORITY[relation_type],
                    "seed_rank": seed_rank,
                    "reranker_score": result.get("reranker_score"),
                    "selection_reason": reason,
                    "provenance": [
                        EvidencePath(
                            seed_chunk_id=seed_id,
                            relation_type=relation_type,
                            relation_direction=direction,  # type: ignore[arg-type]
                            expansion_depth=1,
                            resolution_confidence=confidence,
                            selection_reason=reason,
                        )
                    ],
                }
            )
            diagnostics.append(
                ExpansionDiagnostic(
                    seed_chunk_id=seed_id,
                    relation_type=relation_type,
                    target_id=target_id,
                    decision="included",
                    reason=reason,
                )
            )

        parent = result.get("parent")
        add("parent", seed.get("parent_chunk_id"), supplied=parent)

        continuation_of = seed.get("continuation_of")
        add("continuation_of", continuation_of)
        for continuation in self.artifacts.get_continuations(seed_id):
            add("continuation", continuation.get("chunk_id"), supplied=continuation)

        relations = seed.get("relations") or {}
        for relation_type in ("previous_sibling", "next_sibling"):
            target_id = relations.get(relation_type)
            if not target_id:
                continue
            target = self.artifacts.get_child(str(target_id))
            if target is None:
                add(relation_type, str(target_id))
                continue
            if self._is_structural_sibling(seed, target):
                add(relation_type, str(target_id), supplied=target)
            else:
                diagnostics.append(
                    ExpansionDiagnostic(
                        seed_chunk_id=seed_id,
                        relation_type=relation_type,
                        target_id=str(target_id),
                        decision="excluded",
                        reason="ordinary sibling lacks structural-completion signal",
                    )
                )

        for relation_key, relation_type, direction in (
            ("references", "reference", "outbound"),
            ("referenced_by", "referenced_by", "inbound"),
        ):
            for reference in relations.get(relation_key) or []:
                target_id = reference.get("target_chunk_id")
                confidence = reference.get("confidence")
                if not reference.get("resolved") or not target_id:
                    diagnostics.append(
                        ExpansionDiagnostic(
                            seed_chunk_id=seed_id,
                            relation_type=relation_type,
                            target_id=target_id,
                            decision="excluded",
                            reason="unresolved reference",
                        )
                    )
                    continue
                if (
                    confidence is not None
                    and float(confidence) < self.settings.reference_confidence_min
                ):
                    diagnostics.append(
                        ExpansionDiagnostic(
                            seed_chunk_id=seed_id,
                            relation_type=relation_type,
                            target_id=str(target_id),
                            decision="excluded",
                            reason="reference confidence below threshold",
                        )
                    )
                    continue
                add(
                    relation_type,
                    str(target_id),
                    direction=direction,
                    confidence=float(confidence) if confidence is not None else None,
                )

        for relation_type in (
            "annex_parent",
            "table_parent",
            "approved_document",
        ):
            add(relation_type, relations.get(relation_type))
        return found

    def _resolve_target(
        self, target_id: str, *, supplied: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]] | None:
        if supplied:
            artifact_type = "child" if supplied.get("chunk_id") else "parent"
            return artifact_type, dict(supplied)
        return self.artifacts.get_artifact(str(target_id))

    @staticmethod
    def _is_structural_sibling(seed: dict[str, Any], sibling: dict[str, Any]) -> bool:
        return bool(
            seed.get("split_count", 1) > 1
            or sibling.get("split_count", 1) > 1
            or sibling.get("continuation_of") == seed.get("chunk_id")
            or seed.get("continuation_of") == sibling.get("chunk_id")
            or (
                seed.get("logical_chunk_id")
                and seed.get("logical_chunk_id") == sibling.get("logical_chunk_id")
            )
        )

    @staticmethod
    def _artifact_id(record: dict[str, Any], artifact_type: str) -> str:
        key = "chunk_id" if artifact_type == "child" else "parent_chunk_id"
        return str(record[key])
