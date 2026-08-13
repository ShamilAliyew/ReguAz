

banking-compliance-assistant/
│
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── docker-compose.yml
│
├── docs/
│   ├── project_scope.md
│   ├── architecture.md
│   ├── evaluation_plan.md
│   ├── api_specification.md
│   └── experiment_results.md
│
├── data/
│   │
│   ├── raw/
│   │   ├── laws/
│   │   ├── aml_kyc/
│   │   ├── regulations/
│   │   ├── governance/
│   │   └── cybersecurity/
│   │
│   ├── processed/
│   │   ├── cleaned_documents/
│   │   ├── chunks/
│   │   └── metadata/
│   │
│   ├── evaluation/
│   │   ├── gold_dataset.csv
│   │   ├── retrieval_test_set.csv
│   │   ├── cross_lingual_test_set.csv
│   │   └── unanswerable_questions.csv
│   │
│   └── feedback/
│
├── backend/
│   │
│   ├── app/
│   │   │
│   │   ├── api/
│   │   │   ├── chat.py
│   │   │   ├── upload.py
│   │   │   ├── sessions.py
│   │   │   ├── feedback.py
│   │   │   └── evaluation.py
│   │   │
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── logger.py
│   │   │   └── constants.py
│   │   │
│   │   ├── services/
│   │   │   │
│   │   │   ├── rag/
│   │   │   │   ├── retrieval.py
│   │   │   │   ├── hybrid_search.py
│   │   │   │   ├── reranker.py
│   │   │   │   ├── query_rewriter.py
│   │   │   │   ├── citation_builder.py
│   │   │   │   └── response_generator.py
│   │   │   │
│   │   │   ├── llm/
│   │   │   │   ├── gemini.py
│   │   │   │   ├── qwen.py
│   │   │   │   └── llm_factory.py
│   │   │   │
│   │   │   ├── embeddings/
│   │   │   │   ├── e5.py
│   │   │   │   ├── bge_m3.py
│   │   │   │   └── embedding_factory.py
│   │   │   │
│   │   │   ├── ingestion/
│   │   │   │   ├── parser.py
│   │   │   │   ├── chunker.py
│   │   │   │   ├── metadata_extractor.py
│   │   │   │   └── index_builder.py
│   │   │   │
│   │   │   └── guardrails/
│   │   │       ├── hallucination_check.py
│   │   │       ├── confidence_score.py
│   │   │       └── refusal_logic.py
│   │   │
│   │   ├── database/
│   │   │   ├── pgvector.py
│   │   │   ├── chroma.py
│   │   │   └── models.py
│   │   │
│   │   └── main.py
│   │
│   └── tests/
│       ├── test_chat.py
│       ├── test_retrieval.py
│       ├── test_citations.py
│       └── test_api.py
│
├── frontend/
│   │
│   ├── src/
│   │   ├── pages/
│   │   │   ├── ChatPage.jsx
│   │   │   ├── EvaluationPage.jsx
│   │   │   └── UploadPage.jsx
│   │   │
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx
│   │   │   ├── CitationCard.jsx
│   │   │   ├── SourceViewer.jsx
│   │   │   ├── FeedbackButtons.jsx
│   │   │   └── Sidebar.jsx
│   │   │
│   │   ├── services/
│   │   │   └── api.js
│   │   │
│   │   └── App.jsx
│   │
│   └── package.json
│
├── experiments/
│   │
│   ├── embeddings/
│   │   ├── e5_experiment.py
│   │   ├── bge_experiment.py
│   │   └── comparison.py
│   │
│   ├── retrieval/
│   │   ├── recall_at_k.py
│   │   ├── mrr.py
│   │   └── hybrid_vs_vector.py
│   │
│   ├── cross_lingual/
│   │   ├── az_to_ru.py
│   │   ├── az_to_en.py
│   │   └── report.py
│   │
│   └── reports/
│
├── evaluation/
│   │
│   ├── ragas/
│   │   ├── faithfulness.py
│   │   ├── answer_relevancy.py
│   │   ├── context_recall.py
│   │   └── context_precision.py
│   │
│   ├── manual_review/
│   │   ├── citation_check.csv
│   │   └── human_scores.csvs
│   │
│   └── evaluation_runner.py
│
├── notebooks/
│   ├── data_exploration.ipynb
│   ├── embedding_analysis.ipynb
│   └── evaluation_analysis.ipynb
│
└── scripts/
    ├── ingest_documents.py
    ├── rebuild_index.py
    ├── run_evaluation.py
    └── export_results.py