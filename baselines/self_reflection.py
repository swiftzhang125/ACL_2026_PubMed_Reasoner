import json
import logging

from langchain_openai import ChatOpenAI
from baselines.prompts import SELF_REFLECTION_PROMPT
from prompts import QA_PROMPT, QUERY_GENERATION_PROMPT, SUMMARY_PROMPT
from pubmed_client import Article, PubMedClient

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


class SelfReflectionAgent:
    """Self-reflection baseline: generate → reflect → re-retrieve → refine."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        reflection_temperature: float = 1.0,
        max_reflection_iterations: int = 3,
        max_articles: int = 20,
        pubmed_api_key: str | None = None,
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.reflection_llm = ChatOpenAI(
            model=model,
            temperature=reflection_temperature,
        )
        self.pubmed = PubMedClient(api_key=pubmed_api_key)
        self.max_reflection_iterations = max_reflection_iterations
        self.max_articles = max_articles

    # -----------------------------
    # Core steps
    # -----------------------------

    def _generate_query(self, question: str, context: str) -> str:
        chain = QUERY_GENERATION_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "context": context or "None",
        })
        return _parse_json(result.content)["query"]

    def _retrieve_and_summarize(
        self,
        query: str,
        question: str,
    ) -> tuple[list[Article], str]:
        """Retrieve articles and produce a summary."""
        articles = self.pubmed.search_and_fetch(
            query,
            max_results=self.max_articles,
        )
        if not articles:
            return [], ""

        raw_sources = "\n\n".join(
            f"[PMID:{a.pmid}] Title: {a.title}\nAbstract: {a.abstract}"
            for a in articles
        )

        chain = SUMMARY_PROMPT | self.llm
        result = chain.invoke({"raw_sources": raw_sources})
        summary = _parse_json(result.content).get("verified_sources", "")

        return articles, summary

    def _generate_answer(
        self,
        question: str,
        summary: str,
        task_instruction: str,
        context: str,
    ) -> dict:
        chain = QA_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "task_instruction": task_instruction
            or "Provide a comprehensive answer with supporting evidence.",
            "context": context or "None",
            "sources": summary,
        })
        return _parse_json(result.content)

    def _reflect(
        self,
        question: str,
        verified_sources: str,
        rationale_answer: str,
        search_history: list[str],
    ) -> str | None:
        """Reflect on gaps and produce a revised query."""
        chain = SELF_REFLECTION_PROMPT | self.reflection_llm
        result = chain.invoke({
            "natural_language_question": question,
            "verified_sources": verified_sources or "None",
            "rationale_answer": rationale_answer,
            "search_history": "\n".join(search_history) if search_history else "None",
        })

        parsed = _parse_json(result.content)
        revised_query = parsed.get("query", "")
        rationale = parsed.get("rationale", "")
        logger.info(f"Reflection rationale: {rationale}")

        if not revised_query:
            return None
        return revised_query

    # -----------------------------
    # Full pipeline
    # -----------------------------

    def run(
        self,
        question: str,
        task_instruction: str = "",
        context: str = "",
    ) -> dict:
        """
        Run the self-reflection pipeline.

        Returns dict with 'answer', 'rationale', 'citations', 'query'.
        """
        logger.info("=== Self-Reflection Agent ===")

        # Step 1: Initial query
        query = self._generate_query(question, context)
        logger.info(f"Initial query: {query}")
        search_history = [query]

        # Step 2: Retrieve + summarize
        all_articles, summary = self._retrieve_and_summarize(query, question)
        logger.info(f"Retrieved {len(all_articles)} articles")

        # Step 3: Initial answer
        answer = self._generate_answer(
            question,
            summary,
            task_instruction,
            context,
        )
        logger.info(f"Initial answer: {answer.get('answer', '')}")

        # Step 4+: Reflection loop
        for iteration in range(self.max_reflection_iterations):
            logger.info(f"\n--- Reflection Iteration {iteration + 1} ---")

            rationale_answer = (
                f"Answer: {answer.get('answer', '')}\n"
                f"Rationale: {answer.get('rationale', '')}"
            )

            revised_query = self._reflect(
                question=question,
                verified_sources=summary,
                rationale_answer=rationale_answer,
                search_history=search_history,
            )

            if not revised_query:
                logger.info("No concept gaps identified. Stopping reflection.")
                break

            logger.info(f"Revised query: {revised_query}")
            search_history.append(revised_query)

            # Re-retrieve
            new_articles, new_summary = self._retrieve_and_summarize(
                revised_query,
                question,
            )
            logger.info(f"Retrieved {len(new_articles)} new articles")

            if new_articles:
                all_articles.extend(new_articles)

                # Merge summaries
                merged_sources = (
                    f"Previous evidence:\n{summary}\n\n"
                    f"New evidence:\n{new_summary}"
                )

                merge_chain = SUMMARY_PROMPT | self.llm
                merge_result = merge_chain.invoke({"raw_sources": merged_sources})
                summary = _parse_json(merge_result.content).get(
                    "verified_sources",
                    summary,
                )

            # Regenerate answer
            answer = self._generate_answer(
                question,
                summary,
                task_instruction,
                context,
            )
            logger.info(f"Refined answer: {answer.get('answer', '')}")

        # Deduplicate articles
        seen_pmids = set()
        unique_articles = []
        for a in all_articles:
            if a.pmid not in seen_pmids:
                seen_pmids.add(a.pmid)
                unique_articles.append(a)

        answer["citations"] = [
            {"pmid": a.pmid, "title": a.title}
            for a in unique_articles
        ]
        answer["query"] = search_history[-1]

        return answer