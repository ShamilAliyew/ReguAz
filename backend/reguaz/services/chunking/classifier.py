from __future__ import annotations

import re
from collections import Counter


PROFILES = (
    "LEGAL_ACT",
    "REGULATION",
    "GUIDANCE",
    "TABLE_OR_FORM",
    "DECISION_OR_AMENDMENT",
)


def classify_document(text: str, category: str, title: str) -> tuple[str, list[str]]:
    sample = f"{title}\n{text[:20000]}".casefold()
    scores: Counter[str] = Counter()
    if category == "laws" or re.search(r"(?m)^\s*m\s*a\s*d\s*d\s*ə\s+\d+", sample):
        scores["LEGAL_ACT"] += 4
    if re.search(r"(?m)^\s*\d+(?:[-.]\d+)+\.?(?:\s|$)", sample):
        scores["REGULATION"] += 3
    if re.search(r"(?m)^#{1,6}\s+", text) or any(
        word in sample for word in ("metodoloji", "rəhbərlik", "təlimat")
    ):
        scores["GUIDANCE"] += 2
    if re.search(r"(?m)^\s*\|.+\|\s*$", text) or any(
        word in sample for word in ("forma", "sorğu", "hesabat cədvəli")
    ):
        scores["TABLE_OR_FORM"] += 3
    if any(word in sample for word in ("qərara alır", "təsdiq edilməsi barədə", "dəyişiklik")):
        scores["DECISION_OR_AMENDMENT"] += 3
    if not scores:
        scores["GUIDANCE"] = 1
    detected = [profile for profile in PROFILES if scores[profile] > 0]
    primary = max(PROFILES, key=lambda profile: (scores[profile], -PROFILES.index(profile)))
    return primary, detected


def infer_document_type(profile: str, title: str) -> str:
    folded = title.casefold()
    if profile == "LEGAL_ACT" or "qanun" in folded:
        return "law"
    if profile == "DECISION_OR_AMENDMENT":
        return "decision_or_amendment"
    if profile == "TABLE_OR_FORM":
        return "table_or_form"
    if profile == "REGULATION" or any(x in folded for x in ("qayda", "əsasnamə")):
        return "regulation"
    return "guidance"
