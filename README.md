# PubMed Reasoner

A 3-stage biomedical QA agent that searches PubMed, refines queries with self-critique, retrieves article evidence with early stopping, and generates an evidence-grounded final answer.

## Overview

This project implements a multi-stage reasoning pipeline for biomedical question answering over PubMed:

1. **Search with Self-Critic Query Refinement**
   - Converts a natural language biomedical question into a PubMed query
   - Iteratively evaluates and refines the query using self-critique
   - Optimizes for coverage, alignment, and low redundancy

2. **Reflective Article Retrieval with Early Stopping**
   - Retrieves candidate PubMed articles
   - Filters articles by title/abstract relevance
   - Extracts evidence in batches
   - Stops early once enough evidence has been collected

3. **Evidence-Grounded Response Generation**
   - Synthesizes extracted evidence into a coherent summary
   - Generates a final answer grounded in the evidence
   - Returns citations with PMID references

---

## File Structure

```text
pubmed/
├── pubmed_reasoner.py   # Main agent with all 3 stages
├── pubmed_client.py     # PubMed E-utilities API wrapper
├── prompts.py           # All prompt templates
├── requirements.txt     # Dependencies
└── main.py              # CLI entry point / demo
```

---

## Quickstart

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Set API key
```bash
export OPENAI_API_KEY="your_key_here"
```

3. Run
```bash
python main.py "Do leukotrienes play a key role in asthma?" \
  --task "Answer yes/no with justification" -v
```


# Publication
Please cite our papers if you use our idea or code:

```text
@article{zhang2026pubmed,
  title={PubMed Reasoner: Dynamic Reasoning-based Retrieval for Evidence-Grounded Biomedical Question Answering},
  author={Zhang, Yiqing and Liu, Xiaozhong and Murai, Fabricio},
  journal={arXiv preprint arXiv:2603.27335},
  year={2026}
}
```