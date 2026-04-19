from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------------------
# Pairwise Comparison (from Appendix H)
# ----------------------------------------------------------------

PAIRWISE_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a neutral medical evaluator. Compare two answers from medical "
        "language models for a PubMedQA-style question. Judge *reasoning quality "
        "only* (not model identity)."
    )),
    ("human", (
        "Question: {natural_language_question}\n\n"
        "Answer A: {answer_a}\n\n"
        "Answer B: {answer_b}\n\n"
        "Evaluate each answer independently on four dimensions (1-5):\n"
        "1) Reasoning Soundness - logical, coherent, internally consistent.\n"
        "2) Evidence Grounding - claims supported by biomedical evidence; no hallucinations.\n"
        "3) Clinical Relevance - directly addresses the question in an evidence-based manner.\n"
        "4) Trustworthiness - safe, conforms to biomedical knowledge; not misleading.\n\n"
        "Instructions:\n"
        "- Assign a numeric score (1-5) for each dimension to both A and B.\n"
        "- Give a brief justification (less than 2 sentences) for each score.\n"
        "- Provide an overall verdict based on reasoning quality: \"A\", \"B\", or \"tie\".\n"
        "- Do not mention model names or speculate on sources.\n"
        "- Output strictly valid JSON matching this schema (and nothing else):\n"
        "{\n"
        '  "Answer A": {\n'
        '    "Reasoning Soundness": {"score": 0, "justification": ""},\n'
        '    "Evidence Grounding": {"score": 0, "justification": ""},\n'
        '    "Clinical Relevance": {"score": 0, "justification": ""},\n'
        '    "Trustworthiness": {"score": 0, "justification": ""}\n'
        "  },\n"
        '  "Answer B": {\n'
        '    "Reasoning Soundness": {"score": 0, "justification": ""},\n'
        '    "Evidence Grounding": {"score": 0, "justification": ""},\n'
        '    "Clinical Relevance": {"score": 0, "justification": ""},\n'
        '    "Trustworthiness": {"score": 0, "justification": ""}\n'
        "  },\n"
        '  "verdict": "A or B or tie"\n'
        "}"
    )),
])

# ----------------------------------------------------------------
# Independent Likert Scoring (single-answer evaluation)
# ----------------------------------------------------------------

LIKERT_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a neutral medical evaluator. Evaluate a single answer from a "
        "medical language model for a PubMedQA-style question. Judge *reasoning "
        "quality only*."
    )),
    ("human", (
        "Question: {natural_language_question}\n\n"
        "Answer: {answer}\n\n"
        "Evaluate the answer on four dimensions (1-5):\n"
        "1) Reasoning Soundness - logical, coherent, internally consistent.\n"
        "2) Evidence Grounding - claims supported by biomedical evidence; no hallucinations.\n"
        "3) Clinical Relevance - directly addresses the question in an evidence-based manner.\n"
        "4) Trustworthiness - safe, conforms to biomedical knowledge; not misleading.\n\n"
        "Instructions:\n"
        "- Assign a numeric score (1-5) for each dimension.\n"
        "- Give a brief justification (less than 2 sentences) for each score.\n"
        "- Output strictly valid JSON matching this schema (and nothing else):\n"
        "{\n"
        '  "Reasoning Soundness": {"score": 0, "justification": ""},\n'
        '  "Evidence Grounding": {"score": 0, "justification": ""},\n'
        '  "Clinical Relevance": {"score": 0, "justification": ""},\n'
        '  "Trustworthiness": {"score": 0, "justification": ""}\n'
        "}"
    )),
])