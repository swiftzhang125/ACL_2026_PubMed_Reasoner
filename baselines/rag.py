import json
import logging

from langchain_openai import ChatOpenAI
from prompts import QA_PROMPT, QUERY_GENERATION_PROMPT, SUMMARY_PROMPT
from pubmed_client import PubMedClient

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


class RAGBaseline:
    """Single-pass RAG: query → retrieve → summarize → answer."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_articles: int = 20,
        pubmed_api_key: str | None = None,
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.pubmed = PubMedClient(api_key=pubmed_api_key)
        self.max_articles = max_articles

    def run(
        self,
        question: str,
        task_instruction: str = "",
        context: str = "",
    ) -> dict:
        """
        Run single-pass RAG pipeline.

        Returns dict with 'answer', 'rationale', 'citations', 'query'.
        """
        logger.info("=== RAG Baseline ===")

        # Step 1: One-shot query generation (no self-critic)
        chain = QUERY_GENERATION_PROMPT | self.llm
        query_result = chain.invoke({
            "natural_language_question": question,
            "context": context or "None",
        })

        query_parsed = _parse_json(query_result.content)
        query = query_parsed["query"]
        logger.info(f"Generated query: {query}")

        # Step 2: Retrieve articles
        articles = self.pubmed.search_and_fetch(
            query,
            max_results=self.max_articles,
        )
        logger.info(f"Retrieved {len(articles)} articles")

        if not articles:
            return {
                "answer": "No relevant articles found.",
                "rationale": "PubMed search returned no results.",
                "citations": [],
                "query": query,
            }

        # Step 3: Summarize all retrieved evidence
        raw_sources = "\n\n".join(
            f"[PMID:{a.pmid}] Title: {a.title}\nAbstract: {a.abstract}"
            for a in articles
        )

        summary_chain = SUMMARY_PROMPT | self.llm
        summary_result = summary_chain.invoke({"raw_sources": raw_sources})
        summary = _parse_json(summary_result.content).get("verified_sources", "")

        logger.info(f"Summary: {summary[:200]}...")

        # Step 4: Generate final answer
        qa_chain = QA_PROMPT | self.llm
        qa_result = qa_chain.invoke({
            "natural_language_question": question,
            "task_instruction": task_instruction
            or "Provide a comprehensive answer with supporting evidence.",
            "context": context or "None",
            "sources": summary,
        })

        response = _parse_json(qa_result.content)

        # Attach citations
        response["citations"] = [
            {"pmid": a.pmid, "title": a.title}
            for a in articles
        ]
        response["query"] = query

        return response