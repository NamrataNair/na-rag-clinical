# NA-RAG: Negation- and Assertion-Aware Retrieval-Augmented Generation for Clinical Records

This repository contains the reference implementation for the paper:

**Negation- and Assertion-Aware Retrieval-Augmented Generation for Clinical Case and Medical Record Retrieval**

The system demonstrates how clinical **assertion status (asserted, negated, hypothetical, historical)** and **temporal relevance** can be integrated directly into the retrieval stage of a RAG pipeline.

The implementation is evaluated on the **i2b2 2012 Temporal Relations dataset**.

# Citation
If you use this repository, please cite:
Nair, N. (2025). Negation- and Assertion-Aware Retrieval-Augmented Generation for Clinical Case and Medical Record Retrieval.

---

## Key Contributions Implemented

- **Robust Assertion-Aware Extraction**: Uses prioritized regex matching and word boundaries to accurately capture clinical event assertions (asserted, negated, historical, hypothetical, family history) while avoiding sub-word false positives.
- **Event-centric Indexing**: Chunks and indexes clinical notes with mapped assertion metadata.
- **Hybrid Retrieval (Dense + BM25)**: Fuses dense representations with keyword matching.
- **Query-Aware Assertion-Constrained Re-ranking**: Dynamically parses the user's intent. Instead of always penalizing negated conditions, if the query specifically asks for "ruled out" conditions or "family history", the system intelligently promotes those negatively-asserted findings to the top.
- **Negation-Aware Evaluation Metrics**: Evaluates `precision@k` specifically for asserted traits, and measures Negation False Positive Rates.
- **Faithfulness Checks**: Post-generation guardrails to ensure the LLM doesn't incorrectly assert a condition that the evidence marks as negated.

---

## Dataset

The **i2b2 2012 dataset is not distributed** due to licensing restrictions.

After obtaining access, place the data as:

```text
data/
├── train/
├── dev/
└── test/
```

Each document should contain sentence-level annotations for:
- EVENT
- ASSERTION
- TEMPORAL relations

---

## Installation

We recommend using a virtual environment.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running Experiments

You can test the RAG pipeline interactively using several provided scripts:

1. **End-to-End Test (Mock Data):**
   Demonstrates the entire pipeline locally, including chunking, retrieval, reranking, and guardrail checking against a mocked clinical corpus.
   ```bash
   python experiments/e2e_test.py
   ```

2. **Run NA-RAG Retrieval Pipeline:**
   Allows testing the core NA-RAG logic against mocked sample results.
   ```bash
   python experiments/run_na_rag.py
   ```

3. **Run Baseline RAG:**
   Placeholder script to benchmark traditional retrieval against NA-RAG.
   ```bash
   python experiments/run_baselines.py
   ```

4. **Run Retrieval Evaluations:**
   Generates metrics for precision@k and Negation FP rate.
   ```bash
   python evaluation/evaluate_retrieval.py
   ```

---

## Testing

This project uses `pytest` for rigorous unit and integration testing across all pipeline modules. The test suite covers logic precedence, guardrail validity, dense retrieval accuracy, and dynamic query-aware reranking intent.

To execute the test suite:

```bash
python -m pytest tests/
```

---

## License
Research-only, non-commercial use.