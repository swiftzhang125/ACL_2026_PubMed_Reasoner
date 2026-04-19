import json
import logging

from langchain_openai import ChatOpenAI
from prompts import QA_PROMPT

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


class LLMBaseline:
    """Direct LLM answering — no retrieval, no search."""

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)

    def run(
        self,
        question: str,
        task_instruction: str = "",
        context: str = "",
    ) -> dict:
        """
        Answer a biomedical question using only the LLM.

        Returns dict with 'answer' and 'rationale'.
        """
        logger.info("=== LLM Baseline (no retrieval) ===")

        chain = QA_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "task_instruction": task_instruction
            or "Provide a comprehensive answer with supporting evidence.",
            "context": context or "None",
            "sources": "No external sources provided.",
        })

        parsed = _parse_json(result.content)
        parsed["citations"] = []
        parsed["query"] = None

        return parsed