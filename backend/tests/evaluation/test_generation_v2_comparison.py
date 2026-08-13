from backend.reguaz.evaluation.generation_v2_comparison import (
    score_record,
    summarize,
    token_f1,
)


def test_token_f1_is_unicode_casefolded_and_number_aware() -> None:
    assert token_f1("ƏMSAL 12,5", "əmsal 12,5") == 1.0
    assert 0 < token_f1("kapital tələbi", "kapital limiti") < 1


def test_generation_summary_uses_gold_citations_and_nearest_rank_p95() -> None:
    record = {
        "reference_answer": "Bank kapital saxlamalıdır.",
        "relevant_chunk_ids": ["c1", "c2"],
        "relevant_document_version_ids": ["d1"],
    }
    output = {
        "answer": "Bank kapital saxlamalıdır. [1]",
        "status": "answered",
        "answer_blocks": [{"citation_numbers": [1]}],
        "citations": [
            {
                "chunk_id": "c1",
                "document_version_id": "d1",
                "source_validated": True,
            }
        ],
        "warnings": [],
    }
    quality = score_record(record, output)
    assert quality["gold_child_citation_recall"] == 0.5
    assert quality["gold_document_citation_hit"] is True
    results = [
        {
            "completed": True,
            "status": "answered",
            "generation_ms": value,
            "attempt_ms": value,
            "projected_e2e_ms": value + 5,
            "quality": quality,
        }
        for value in range(1, 11)
    ]
    summary = summarize(results)
    assert summary["generation_latency_ms"]["p95"] == 10
    assert summary["mean_gold_child_citation_recall"] == 0.5


def test_insufficient_evidence_without_citations_is_not_source_failure() -> None:
    record = {
        "reference_answer": "Tələb",
        "relevant_chunk_ids": ["c1"],
        "relevant_document_version_ids": ["d1"],
    }
    quality = score_record(
        record,
        {
            "answer": "Etibarlı cavab mümkün deyil.",
            "status": "insufficient_evidence",
            "answer_blocks": [],
            "citations": [],
            "warnings": [],
        },
    )
    assert quality["all_sources_validated"] is True
