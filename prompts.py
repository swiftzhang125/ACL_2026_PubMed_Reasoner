"""
Prompt templates for PubMed Reasoner, transcribed from the paper's Appendix E.
"""

from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------------------------------------------
# Stage 1: Search with Self-Critic Query Refinement
# ----------------------------------------------------------------------

QUERY_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert in PubMed search syntax.\n"
        "Your task is to convert the provided natural language description of the "
        "desired literature, together with any contextual information, into a single "
        "valid PubMed query string.\n\n"
        "Requirements:\n"
        "1. Use only the Boolean operator AND to connect terms.\n"
        "2. Field tag priority:\n"
        "   (a) Place MeSH terms first and apply the [mesh] tag whenever possible.\n"
        "   (b) Place date ranges next, formatted as either YYYY:YYYY[pdat] or "
        "YYYY/MM/DD:YYYY/MM/DD[pdat].\n"
        "   (c) Place all remaining terms last. Do not apply special field tags "
        "unless explicitly specified.\n"
        "3. Spacing: Separate each term and each AND with exactly one space.\n"
        "4. Date range format: YYYY:YYYY[pdat] or YYYY/MM/DD:YYYY/MM/DD[pdat]\n"
    )),
    ("human", (
        "Natural Language Question:\n{natural_language_question}\n\n"
        "Additional Context (if any):\n{context}\n\n"
        "Output:\n"
        "Return the answer strictly as JSON following this schema:\n"
        "{\n"
        '  "query": "The final PubMed query string as a single string.",\n'
        '  "rationale": "A brief explanation of term selection, field tags, ordering, '
        'use of AND, and how the context was incorporated."\n'
        "}\n\n"
        "Do not add any explanation or additional text outside the JSON."
    )),
])


SELF_CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a PubMed search planning assistant.\n"
        "Your task is to produce one improved PubMed query for the next search step by:\n"
        "- Interpreting the natural-language question and any additional context.\n"
        "- Using search history to avoid ineffective or repetitive patterns.\n"
        "- Evolving the candidate term set using coverage, alignment, and redundancy feedback.\n"
        "- Ensuring the final query strictly follows the requirements.\n\n"
        "Feedback Signals:\n"
        "If no context is provided, return -1 for the corresponding signals.\n"
        "In producing the improved query, you must incorporate evolving feedback from:\n"
        "1. Coverage - 1 if the provided context sufficiently represents the concepts "
        "relevant to the question; 0 otherwise.\n"
        "2. Alignment - 1 if the provided context is relevant and appropriately focused "
        "on the question; 0 otherwise.\n"
        "3. Redundancy - 1 if there are no overlapping, unnecessary, or logically "
        "unintended terms; 0 otherwise.\n\n"
        "Logical operators:\n"
        "- Use AND to combine independent, parallel constraints. Every term connected "
        "by AND must be justified.\n"
        "- Use OR only for similar or interchangeable concepts. When using OR, you must "
        "enclose the entire OR-group in parentheses, e.g.: (term1[mesh] OR term2[mesh]).\n\n"
        "Requirements:\n"
        "1. Use only the Boolean operator AND to connect terms.\n"
        "2. Field tag priority:\n"
        "   (a) Place MeSH terms first and apply the [mesh] tag whenever possible.\n"
        "   (b) Place date ranges next, formatted as either YYYY:YYYY[pdat] or "
        "YYYY/MM/DD:YYYY/MM/DD[pdat].\n"
        "   (c) Place all remaining terms last. Do not apply special field tags "
        "unless explicitly specified.\n"
        "3. Spacing: Separate each term and each AND with exactly one space.\n"
        "4. Date range format: YYYY:YYYY[pdat] or YYYY/MM/DD:YYYY/MM/DD[pdat]\n"
    )),
    ("human", (
        "Natural Language Question:\n{natural_language_question}\n\n"
        "Additional Context (if any):\n{search_note}\n\n"
        "Search History (if any):\n{search_history}\n\n"
        "Output:\n"
        "Return the answer strictly as JSON following this schema:\n"
        "{\n"
        '  "query": "The final PubMed query string as a single string.",\n'
        '  "rationale": "A brief explanation of term selection, field tags, ordering, '
        'use of logical operators, and how the context and feedback were incorporated.",\n'
        '  "feedback": {\n'
        '    "coverage": 0,\n'
        '    "coverage_suggestion": "Suggested improvements",\n'
        '    "alignment": 0,\n'
        '    "alignment_suggestion": "Suggested improvements",\n'
        '    "redundancy": 0,\n'
        '    "redundancy_suggestion": "Suggested improvements"\n'
        "  }\n"
        "}\n\n"
        "Do not add any explanation or additional text outside the JSON."
    )),
])


# ------------------------------------------------------------
# Stage 2: Reflective Article Retrieval with Early Stopping
# ------------------------------------------------------------

COARSE_FILTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a biomedical article relevance filter.\n"
        "Your task is to determine whether each article is potentially relevant to "
        "the given question based on its title and abstract.\n"
        "Be inclusive - retain articles that are even marginally relevant."
    )),
    ("human", (
        "Natural Language Question:\n{natural_language_question}\n\n"
        "Articles:\n{articles_str}\n\n"
        "Output:\n"
        "Return your answer strictly as JSON following this schema:\n"
        "{\n"
        '  "decisions": [\n'
        '    {"pmid": "...", "keep": true, "rationale": "..."}\n'
        "  ]\n"
        "}\n"
        "Do not include any explanation or text outside the JSON."
    )),
])

EVIDENCE_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a biomedical evidence extraction assistant.\n"
        "Your task is to extract the key evidence from the provided article that "
        "directly addresses the given question. Preserve the PMID for citation."
    )),
    ("human", (
        "Natural Language Question:\n{natural_language_question}\n\n"
        "Article (PMID: {pmid}):\n"
        "Title: {title}\n"
        "Abstract: {abstract}\n\n"
        "Output:\n"
        "Return your answer strictly as JSON following this schema:\n"
        "{\n"
        '  "evidence": "The extracted evidence passage that addresses the question.",\n'
        '  "aligned": true,\n'
        '  "rationale": "Why this evidence is or is not aligned with the question."\n'
        "}\n"
        "Do not include any explanation or text outside the JSON."
    )),
])

REFLECTIVE_SUFFICIENCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a reflection assistant.\n"
        "Your task is to determine whether the provided search results with current "
        "context (if provided) contain enough relevant and specific information to "
        "answer the question."
    )),
    ("human", (
        "Natural Language Question:\n{natural_language_question}\n\n"
        "Search Results:\n{search_results_str}\n\n"
        "Additional Context (if any):\n{context}\n\n"
        "Output:\n"
        "Return your answer strictly as JSON following this schema:\n"
        "{\n"
        '  "is_sufficient": true,\n'
        '  "rationale": "Concise explanation of why the information is sufficient '
        'or insufficient.",\n'
        '  "needed_pmids": ["PMID1", "PMID2"]\n'
        "}\n"
        "Do not include any explanation or text outside the JSON."
    )),
])

# ------------------------------------------------------------
# Stage 3: Evidence-Grounded Response Generation
# ------------------------------------------------------------

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a professional academic rewriting assistant.\n"
        "Your task is to transform the provided raw sources into a single, "
        "semantically coherent, and well-structured paragraph.\n\n"
        "Requirements:\n"
        "1. Use only the information from the provided raw sources, without adding "
        "external content.\n"
        "2. Preserve all original in-text citations exactly as they appear "
        "(e.g., [PMID: xxxx]).\n"
        "3. Ensure the paragraph is logically connected, concise, and scientifically "
        "rigorous."
    )),
    ("human", (
        "Raw Sources:\n{raw_sources}\n\n"
        "Output:\n"
        "Return the answer strictly as JSON following this schema:\n"
        "{\n"
        '  "verified_sources": "The final rewritten paragraph as a single string."\n'
        "}\n"
        "Do not add any explanation or additional text outside the JSON."
    )),
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an expert assistant.\n"
        "When sources are provided, you should primarily base your answer on the "
        "information in the sources. If the sources do not contain enough information "
        "to fully answer the question, you may supplement your answer using your own "
        "knowledge.\n"
        "Provide your answer and a clear rationale explaining how you arrived at it."
    )),
    ("human", (
        "Natural Language Question:\n{natural_language_question}\n\n"
        "Task Instruction:\n{task_instruction}\n\n"
        "Additional Context (if any):\n{context}\n\n"
        "Sources:\n{sources}\n\n"
        "Output:\n"
        "Return the answer strictly as JSON following this schema:\n"
        "{\n"
        '  "answer": "Your answer according to the task instruction.",\n'
        '  "rationale": "A clear explanation of how you arrived at the answer."\n'
        "}\n"
        "Do not add any explanation or additional text outside the JSON."
    )),
])