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

- Assertion-aware clinical event extraction
- Event-centric indexing of clinical notes
- Hybrid retrieval (Dense + BM25)
- Query-aware assertion-constrained re-ranking
- Negation-aware evaluation metrics
- Faithfulness checks for generated output

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

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Running Experiments

You can run the end-to-end pipeline on a mock clinical corpus:
```bash
python experiments/e2e_test.py
```

To run baseline or other scripts:
```bash
python experiments/run_baselines.py
python experiments/run_na_rag.py
python evaluation/evaluate_retrieval.py
```

---

## Testing

We use `pytest` for validating the end-to-end extraction and negation query behaviors. To execute tests:

```bash
python -m pytest tests/
```

---

## License
Research-only, non-commercial use.
