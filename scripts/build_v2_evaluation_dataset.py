#!/usr/bin/env python3
"""Build the frozen, source-first ReguAZ V2 evaluation dataset."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.reguaz.evaluation.v2_dataset import (  # noqa: E402
    DatasetManifest,
    EvaluationRecord,
    sha256_file,
    sha256_files,
    stable_record_id,
    validate_dataset,
)
from backend.reguaz.retrieval.v2_artifacts import V2ArtifactStore  # noqa: E402
from backend.reguaz.retrieval.v2_contract import V2RetrievalContract  # noqa: E402


SEED = 20260805
OUTPUT = Path("data/evaluation/reguaz_v2_evaluation_50.jsonl")
MANIFEST = Path("data/evaluation/reguaz_v2_evaluation_50_manifest.json")
SOURCE_WORKBOOK = Path("data/evaluation/gold_dataset_for_llm_generation.xlsx")


CURATED: list[tuple[str, str, str, list[str]]] = [
    ("Q001", "easy", "definition", ["chunk_d242f9a0c7bb4c171c14c1cd"]),
    ("Q006", "easy", "deadline", ["chunk_de9b2364353659f9c48f8f4d"]),
    ("Q011", "easy", "deadline", ["chunk_c032f99bcd5735307e1f6872"]),
    ("Q016", "medium", "procedure", ["chunk_50377e837ca7239606254bce"]),
    ("Q021", "medium", "threshold", ["chunk_60ce8d64297e800acb0ed488"]),
    ("Q022", "medium", "calculation_rule", ["chunk_9f369ea8d9472a4e07b73438"]),
    ("Q015", "hard", "exception", ["chunk_97cca170f6afea0c698bcc1a"]),
    ("Q009", "hard", "conditions", ["chunk_f60cf8800d312613f571e0e3"]),
    ("Q023", "easy", "definition", ["chunk_45a0987348942f31a4249cf5"]),
    ("Q026", "easy", "obligation", ["chunk_99b2a45eed13e393b6f0c6f6"]),
    ("Q029", "medium", "deadline", ["chunk_ebff8f68af80a4fdafce8ac7"]),
    ("Q030", "medium", "conditions", ["chunk_af1e992a53b9ce0e9ec74e57"]),
    ("Q032", "hard", "approval_rule", ["chunk_965c1939648766babbb1eb3f"]),
    (
        "Q034",
        "hard",
        "conflict_of_interest",
        [
            "chunk_9492faa942915becc2b49fba",
            "chunk_aebd395148611ad5c9ec2347",
            "chunk_b22d40883d4867281d834c77",
        ],
    ),
    (
        "Q035",
        "hard",
        "required_documents",
        [
            "chunk_9acf97a166476914605ee37a",
            "chunk_ba711ff25f108c2dd8aff19e",
            "chunk_a9bc388590d773b8500a8483",
            "chunk_b7428585e54b56b1a28149f7",
            "chunk_45e65eef75f3bb683c889168",
            "chunk_f19c21b617f2cc25b6f40160",
            "chunk_3a15cafb220fbf0e51e6d414",
        ],
    ),
    ("Q155", "easy", "direct_fact", ["chunk_41c99f179fe60fe612899b79"]),
    ("Q156", "easy", "definition", ["chunk_208d1e14515e49ecdda7aa7f"]),
    ("Q160", "medium", "technology_list", ["chunk_d079a02aa856131badf0237b"]),
    ("Q161", "medium", "scope", ["chunk_201d20a9211450e6cab9c7bb"]),
    ("Q163", "easy", "definition", ["chunk_94d603b803e95b6c922f863f"]),
    ("Q164", "easy", "definition", ["chunk_fbc556f5e4b134bdedfe9fc5"]),
    ("Q166", "easy", "definition", ["chunk_cf44caeba56816a9ce8ed805"]),
    ("Q171", "medium", "frequency", ["chunk_c554229a9918277509a6db57"]),
    ("Q173", "medium", "threshold", ["chunk_552175cdcebe0e00eb3ad2fa"]),
    ("Q174", "medium", "definition", ["chunk_78264c9fa51ec7b211a938f0"]),
    ("Q184", "easy", "definition", ["chunk_bfa9e3a9fa08f4ddb5833712"]),
    ("Q189", "easy", "default_rule", ["chunk_ca73182fe129def40415f4bc"]),
    ("Q192", "medium", "deadline", ["chunk_0d5954dc17ba619fbd58b025"]),
    ("Q200", "medium", "reporting_frequency", ["chunk_9bcc736e7e5e62f17fdfb123"]),
    (
        "Q205",
        "hard",
        "recognition_conditions",
        ["chunk_49ab523f4480f65282233f54", "chunk_3202aba95dc0bf6902cc1e2c"],
    ),
    (
        "Q207",
        "hard",
        "distinguish_concepts",
        ["chunk_756270e6b0eaeb2962b179a3", "chunk_2774de982172b64202fe17b0"],
    ),
    ("Q044", "easy", "purpose", ["chunk_ee4b4e0748dd7a69118d679b"]),
    ("Q051", "easy", "frequency", ["chunk_0e2a248192e46e10d0edf55d"]),
    ("Q053", "easy", "definition", ["chunk_90b92f820b7abb831a50dcba"]),
    ("Q047", "medium", "risk_types", ["chunk_9a227cd1069c0621bcf7f32a"]),
    ("Q049", "medium", "measurement_method", ["chunk_c99657afd8df38fcc1e05c6f"]),
    (
        "Q050",
        "hard",
        "policy_requirements",
        [
            "chunk_1871a90432b0eec5ecc6ed2d",
            "chunk_a6954f4c0f73a5d6b07a77ea",
            "chunk_fc6a387090b7c3ecc6d516b5",
            "chunk_08b53dec8fe78dbc94a7ecf1",
            "chunk_ef5f67d43245c0579e0d81ed",
            "chunk_37a6c8a85fbb096747c7e84d",
            "chunk_593f4965335b7fc6eaee50db",
            "chunk_a2855c502128c5cdeca22919",
            "chunk_44f64c3a90a93f2f5d45a287",
            "chunk_0688ee1b34ce56163fd8453a",
        ],
    ),
    ("Q085", "easy", "prohibition", ["chunk_e1fcba094bb0263794b4fd61"]),
    ("Q105", "easy", "definition", ["chunk_86947b9e5e2c46d7382e9322"]),
    ("Q072", "medium", "required_information", ["chunk_ebe89a193334db503423de80"]),
    (
        "Q074",
        "medium",
        "identification_requirements",
        ["chunk_ff94ebb17642052cec5b7447"],
    ),
    (
        "Q086",
        "medium",
        "threshold",
        [
            "chunk_6857399a157a200df101fc51",
            "chunk_971625e1c4b8da54db5d7c23",
            "chunk_e71a150eaaf2db6a31465343",
            "chunk_f6abc49467224b7b34574105",
        ],
    ),
    ("Q114", "medium", "definition_threshold", ["chunk_782a67dafd2df929c126b860"]),
    (
        "Q082",
        "hard",
        "international_sources",
        [
            "chunk_41d4641fc420a988b03e6008",
            "chunk_61d89a225707b37dcddb7a5d",
            "chunk_51273a77425c477675d2be6d",
            "chunk_dbd5d321b2362f6c6ccbc961",
            "chunk_f382b444607e738024b89386",
            "chunk_9c61a9c1ce8fcedb70c1b568",
        ],
    ),
]


PAYMENTS = [
    (
        "P073",
        "easy",
        "choice",
        "Hesab üzrə sərəncam hüququ bir neçə şəxsə verildikdə bu hüquq hansı formalarda həyata keçirilə bilər?",
        "Hesab üzrə sərəncam hüququ olan şəxslər həmin hüquqdan ayrı-ayrılıqda və ya birlikdə istifadə edə bilərlər.",
        ["chunk_d462aacc6cfd9a1a7de1a6ed"],
    ),
    (
        "P079",
        "easy",
        "permitted_operations",
        "Müvəqqəti cari hesab üzrə hansı əməliyyatların aparılmasına icazə verilir?",
        "Müvəqqəti cari hesab üzrə yalnız nizamnamə kapitalının, şərikli kapitalın və ya payların formalaşdırılması və bank xidmətləri haqqının ödənilməsi ilə bağlı əməliyyatlar aparıla bilər; Qeydin 4-cü hissəsindəki hal istisnadır.",
        ["chunk_ba3d0b188535826eb4c0b71e"],
    ),
    (
        "P081",
        "easy",
        "finality_rule",
        "Ödəniş sisteminin qəbul etdiyi ödəniş sərəncamı hansı andan etibarən geri götürülə bilməz?",
        "Ödəniş sərəncamı ödəniş sisteminə dair qaydalarla müəyyən edilmiş andan etibarən iştirakçı və ya digər şəxs tərəfindən geri götürülə bilməz.",
        ["chunk_48f01e0d54fb03f6ae5c76c1"],
    ),
    (
        "P075",
        "medium",
        "threshold",
        "Fiziki şəxslər nağd milli valyutanı hansı məbləğdən etibarən gömrük orqanlarına yazılı formada bəyan etməlidirlər?",
        "20 000 manat və ondan yuxarı məbləğdə nağd milli valyuta gömrük orqanlarına yazılı formada tam məbləğdə bəyan edilməlidir.",
        ["chunk_25ce39de2f5cbe91425c1c5c"],
    ),
    (
        "P072",
        "medium",
        "verification_steps",
        "Kassir məxaric sənədini qəbul edərkən hansı məlumatları və elementləri yoxlamalıdır?",
        "Kassir tərtib tarixini, məbləğin rəqəm və yazı ilə eyniliyini, səlahiyyətli şəxslərin imzalarını, ƏİS-dəki məbləğin sənədlə uyğunluğunu və şəxsiyyət sənədi məlumatlarını yoxlamalı; digər qiymətlilər olduqda örtük rekvizitləri və plombları da tutuşdurmalıdır.",
        ["chunk_a4e7dbfa4c6ba66b3de92d99"],
    ),
    (
        "P082",
        "hard",
        "nested_conditions",
        "Ödəniş əməliyyatının aşağı riskli hesab edilməsi üçün hansı şərtlərin hamısı ödənilməlidir?",
        "Fırıldaqçılıq dərəcəsi müəyyən edilmiş həddi aşmamalı, əməliyyat məbləği uyğun limitdən yüksək olmamalı və real vaxt risk analizində qeyri-adi xərcləmə və davranış, cihaz və proqram çıxışı üzrə qeyri-adi məlumat, zərərverici proqram, fırıldaqçılıq ssenarisi, şübhəli ölkə halı və ya alanın yüksək riskli ölkədə olması aşkar edilməməlidir.",
        [
            "chunk_69c6d14358a7a0e3f879f3fb",
            "chunk_67ee4af442c894b526f1b60a",
            "chunk_273d07b2aced2b76c2b48105",
            "chunk_70073c462c9b3b50959af12b",
            "chunk_b62fe06b4e263ade3cbba248",
            "chunk_75d94b0aa591bf08df884fb3",
            "chunk_37ba94b9ea3717c218edbc99",
            "chunk_d726208de43b83d77b2ddce9",
            "chunk_218fd5580057e7439a3fd0b0",
            "chunk_17cb99ad6f2b9fb378ff3625",
        ],
    ),
]


def main() -> int:
    contract = V2RetrievalContract.load(Path("data/processed/v2"))
    store = V2ArtifactStore(contract=contract)
    workbook = pd.read_excel(SOURCE_WORKBOOK).set_index("Question ID")
    definitions = []
    for source_id, difficulty, question_type, chunk_ids in CURATED:
        row = workbook.loc[source_id]
        definitions.append(
            (
                source_id,
                difficulty,
                question_type,
                str(row["Question"]).strip(),
                str(row["Ground Truth Answer"]).strip(),
                chunk_ids,
                "Adapted from the repository's pre-existing generation gold workbook; evidence remapped source-first to V2.",
            )
        )
    for source_id, difficulty, question_type, question, answer, chunk_ids in PAYMENTS:
        definitions.append(
            (
                source_id,
                difficulty,
                question_type,
                question,
                answer,
                chunk_ids,
                "Drafted source-first from the selected authoritative V2 payment chunk(s).",
            )
        )

    profiles = _profiles(contract.v2_root)
    prepared = []
    for (
        source_id,
        difficulty,
        question_type,
        question,
        answer,
        chunk_ids,
        note,
    ) in definitions:
        evidence = []
        citations = []
        categories = set()
        source_profiles = set()
        for chunk_id in chunk_ids:
            child = store.get_child(chunk_id)
            if child is None:
                raise ValueError(f"curated evidence is missing: {chunk_id}")
            categories.add(child["category"])
            source_profiles.add(profiles[child["document_id"]])
            evidence.append(
                {
                    key: child[key]
                    for key in (
                        "chunk_id",
                        "logical_chunk_id",
                        "document_version_id",
                        "document_title",
                        "canonical_locator",
                        "page_start",
                        "page_end",
                        "content_sha256",
                        "content",
                    )
                }
            )
            citations.append(
                {
                    key: child[key]
                    for key in (
                        "document_version_id",
                        "document_title",
                        "canonical_locator",
                        "page_start",
                        "page_end",
                    )
                }
            )
        if len(categories) != 1:
            raise ValueError(
                f"evidence categories disagree for {source_id}: {categories}"
            )
        record_id = stable_record_id(question, chunk_ids)
        prepared.append(
            {
                "source_id": source_id,
                "record_id": record_id,
                "difficulty": difficulty,
                "question_type": question_type,
                "question": question,
                "answer": answer,
                "category": categories.pop(),
                "profiles": sorted(source_profiles),
                "evidence": evidence,
                "citations": citations,
                "note": note,
            }
        )

    dev_ids = {
        item["record_id"]
        for item in sorted(
            prepared,
            key=lambda item: hashlib.sha256(
                f"{SEED}:{item['record_id']}".encode()
            ).hexdigest(),
        )[:10]
    }
    records = []
    for item in prepared:
        evidence = item["evidence"]
        records.append(
            EvaluationRecord.model_validate(
                {
                    "id": item["record_id"],
                    "language": "az",
                    "split": "dev" if item["record_id"] in dev_ids else "test",
                    "difficulty": item["difficulty"],
                    "category": item["category"],
                    "question_type": item["question_type"],
                    "question": item["question"],
                    "user_input": item["question"],
                    "reference": item["answer"],
                    "reference_answer": item["answer"],
                    "reference_contexts": [entry["content"] for entry in evidence],
                    "relevant_chunk_ids": [entry["chunk_id"] for entry in evidence],
                    "relevant_logical_chunk_ids": list(
                        dict.fromkeys(entry["logical_chunk_id"] for entry in evidence)
                    ),
                    "relevant_document_version_ids": list(
                        dict.fromkeys(
                            entry["document_version_id"] for entry in evidence
                        )
                    ),
                    "evidence": evidence,
                    "expected_citations": item["citations"],
                    "metadata": {
                        "source_profile": ",".join(item["profiles"]),
                        "requires_multiple_chunks": len(evidence) > 1,
                        "generation_notes": item["note"],
                    },
                }
            )
        )
    records.sort(key=lambda record: record.id)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
    )

    chunks = list((contract.v2_root / "documents").glob("*/chunks.jsonl"))
    parents = list((contract.v2_root / "documents").glob("*/parents.jsonl"))
    manifest = DatasetManifest(
        dataset_name="reguaz_v2_evaluation_50",
        dataset_version="1.0.0",
        creation_timestamp=contract.chunk_manifest.build_timestamp,
        deterministic_seed=SEED,
        total_record_count=len(records),
        dev_count=sum(record.split == "dev" for record in records),
        test_count=sum(record.split == "test" for record in records),
        difficulty_distribution=dict(
            sorted(Counter(record.difficulty for record in records).items())
        ),
        category_distribution=dict(
            sorted(Counter(record.category for record in records).items())
        ),
        question_type_distribution=dict(
            sorted(Counter(record.question_type for record in records).items())
        ),
        corpus_id=contract.chunk_manifest.corpus_version,
        v2_manifest_sha256=sha256_file(contract.chunk_manifest_path),
        source_chunk_artifact_sha256=sha256_files(chunks, contract.v2_root),
        source_parent_artifact_sha256=sha256_files(parents, contract.v2_root),
        generator_implementation_version="1.0.0",
        drafting_model_id="OpenAI Codex",
        drafting_model_revision=None,
        validation_status="pending",
        dataset_jsonl_sha256=sha256_file(OUTPUT),
        near_duplicate_threshold=0.90,
        known_limitations=[
            "No manual expert review has been claimed or performed.",
            "Gold answers are limited to the frozen V2 corpus state.",
            "Some source titles inherit extraction noise from authoritative V2 metadata.",
        ],
    )
    MANIFEST.write_text(
        json.dumps(
            manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str
        )
        + "\n",
        encoding="utf-8",
    )
    validation = validate_dataset(OUTPUT, MANIFEST, v2_root=contract.v2_root)
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["validation_status"] = "valid"
    MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {"dataset": str(OUTPUT), "manifest": str(MANIFEST), **validation},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _profiles(v2_root: Path) -> dict[str, str]:
    output = {}
    for path in (v2_root / "documents").glob("*/quality_report.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        output[report["document_id"]] = report["profile"]
    return output


if __name__ == "__main__":
    raise SystemExit(main())
