"""
PubMed Reasoner: 3-stage biomedical QA agent implementing the paper's pipeline.
"""

import json
import logging

from langchain_openai import ChatOpenAI

from prompts import (
    COARSE_FILTER_PROMPT,
    EVIDENCE_EXTRACTION_PROMPT,
    QA_PROMPT,
    QUERY_GENERATION_PROMPT,
    REFLECTIVE_SUFFICIENCY_PROMPT,
    SELF_CRITIC_PROMPT,
    SUMMARY_PROMPT,
)

from pubmed_client import Article, PubMedClient

logger = logging.getLogger(__name__)


def _parse_json(text: str) -> dict:
    """Extract and parse JSON from LLM output, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` wrapper
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


class PubMedReasoner:
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_self_critic_iterations: int = 3,
        max_articles: int = 20,
        batch_size: int = 5,
        pubmed_api_key: str | None = None,
    ):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.pubmed = PubMedClient(api_key=pubmed_api_key)

        self.max_self_critic_iterations = max_self_critic_iterations
        self.max_articles = max_articles
        self.batch_size = batch_size


    # ------------------------------------------------------------
    # Stage 1: Search with Self-Critic Query Refinement
    # ------------------------------------------------------------

    def _generate_initial_query(self, question: str, context: str = "") -> dict:
        """Generate initial PubMed query from natural language question."""
        chain = QUERY_GENERATION_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "context": context or "None",
        })
        return _parse_json(result.content)


    def _self_critic_refine(
        self,
        question: str,
        search_meta: str,
        search_history: str,
    ) -> dict:
        """Run self-critic to evaluate and refine the query."""
        chain = SELF_CRITIC_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "search_meta": search_meta or "None",
            "search_history": search_history or "None",
        })
        return _parse_json(result.content)


    def search_with_self_critic(
        self,
        question: str,
        context: str = "",
    ) -> tuple[list, list[Article]]:
        """
        Stage 1: Generate and iteratively refine PubMed query.

        Returns the optimized query and the final set of retrieved articles.
        """
        logger.info("=== Stage 1: Search with Self-Critic Query Refinement ===")

        # Step 1: Generate initial query
        initial = self._generate_initial_query(question, context)
        query = initial["query"]

        logger.info(f"Initial query: {query}")
        logger.info(f"Rationale: {initial.get('rationale', '')}")

        search_history = [query]

        # Step 2: Iterative self-critic refinement
        for iteration in range(self.max_self_critic_iterations):
            logger.info(f"\n--- Self-Critic Iteration {iteration + 1} ---")

            # Search PubMed with current query
            articles = self.pubmed.search_and_fetch(query, max_results=10)

            if not articles:
                logger.info("No results found, refining query...")
                search_meta = "No articles returned for the current query."
            else:
                # Build metadata string from titles/abstracts
                search_meta = "\n".join(
                    f"PMID:{a.pmid} | Title: {a.title} | Abstract: {a.abstract[:200]}..."
                    for a in articles[:5]
                )

            # Self-critic evaluation
            critic_result = self._self_critic_refine(
                question=question,
                search_meta=search_meta,
                search_history="\n".join(search_history),
            )

            feedback = critic_result.get("feedback", {})
            coverage = feedback.get("coverage", -1)
            alignment = feedback.get("alignment", -1)
            redundancy = feedback.get("redundancy", -1)

            logger.info(
                f"Feedback - Coverage: {coverage}, Alignment: {alignment}, "
                f"Redundancy: {redundancy}"
            )

            # If all signals are positive, stop refinement
            if coverage == 1 and alignment == 1 and redundancy == 1:
                logger.info("All self-critic signals positive. Query is optimal.")
                query = critic_result["query"]
                search_history.append(query)
                break

            # Use the refined query for next iteration
            query = critic_result["query"]
            search_history.append(query)
            logger.info(f"Refined query: {query}")

        # Final search with optimized query
        logger.info(f"\nFinal optimized query: {query}")
        final_articles = self.pubmed.search_and_fetch(
            query,
            max_results=self.max_articles,
        )

        logger.info(f"Retrieved {len(final_articles)} articles")

        return query, final_articles
    

    # ------------------------------------------------------------
    # Stage 2: Reflective Article Retrieval with Early Stopping
    # ------------------------------------------------------------

    def _coarse_filter(
        self,
        question: str,
        articles: list[Article],
    ) -> list[Article]:
        """Filter articles by title/abstract relevance."""
        if not articles:
            return []

        articles_str = "\n\n".join(
            f"PMID: {a.pmid}\nTitle: {a.title}\nAbstract: {a.abstract[:300]}"
            for a in articles
        )

        chain = COARSE_FILTER_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "articles_str": articles_str,
        })

        decisions = _parse_json(result.content).get("decisions", [])
        keep_pmids = {d["pmid"] for d in decisions if d.get("keep", False)}

        filtered = [a for a in articles if a.pmid in keep_pmids]
        logger.info(
            f"Coarse filter: {len(filtered)}/{len(articles)} articles retained"
        )
        return filtered


    def _extract_evidence(
        self,
        question: str,
        article: Article,
    ) -> dict | None:
        """Extract evidence from a single article."""
        chain = EVIDENCE_EXTRACTION_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "pmid": article.pmid,
            "title": article.title,
            "abstract": article.abstract,
        })

        parsed = _parse_json(result.content)
        if parsed.get("aligned", False):
            return {
                "pmid": article.pmid,
                "title": article.title,
                "evidence": parsed["evidence"],
            }
        return None


    def _check_sufficiency(
        self,
        question: str,
        evidence_pool: list[dict],
        context: str = "",
    ) -> bool:
        """Check if accumulated evidence is sufficient to answer the question."""
        if not evidence_pool:
            return False

        search_results_str = "\n\n".join(
            f"[PMID:{e['pmid']}] {e['evidence']}" for e in evidence_pool
        )

        chain = REFLECTIVE_SUFFICIENCY_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "search_results_str": search_results_str,
            "context": context or "None",
        })

        parsed = _parse_json(result.content)
        is_sufficient = parsed.get("is_sufficient", False)
        logger.info(
            f"Sufficiency check: {is_sufficient} - {parsed.get('rationale', '')}"
        )
        return is_sufficient


    def reflective_retrieval(
        self,
        question: str,
        articles: list[Article],
        context: str = "",
    ) -> list[dict]:
        """
        Stage 2: Reflective article retrieval with early stopping.

        Returns the curated evidence pool.
        """
        logger.info("\n=== Stage 2: Reflective Article Retrieval ===")

        # Coarse filtering
        filtered = self._coarse_filter(question, articles)

        # Process in batches with early stopping
        evidence_pool: list[dict] = []

        for batch_idx in range(0, len(filtered), self.batch_size):
            batch = filtered[batch_idx : batch_idx + self.batch_size]
            batch_num = batch_idx // self.batch_size + 1
            logger.info(
                f"\n--- Processing batch {batch_num} "
                f"({len(batch)} articles) ---"
            )

            # Extract evidence from each article in the batch
            for article in batch:
                evidence = self._extract_evidence(question, article)
                if evidence:
                    evidence_pool.append(evidence)
                    logger.info(
                        f"  Extracted evidence from PMID:{article.pmid}"
                    )

            # Check if we have sufficient evidence
            if self._check_sufficiency(question, evidence_pool, context):
                logger.info(
                    f"Early stopping at batch {batch_num} - "
                    f"sufficient evidence ({len(evidence_pool)} pieces)"
                )
                break

        logger.info(f"Total evidence collected: {len(evidence_pool)} pieces")
        return evidence_pool
    


    # ------------------------------------------------------------
    # Stage 3: Evidence-Grounded Response Generation
    # ------------------------------------------------------------

    def _generate_summary(self, evidence_pool: list[dict]) -> str:
        """Generate summary-of-evidence from the evidence pool."""
        raw_sources = "\n\n".join(
            f"[PMID:{e['pmid']}] {e['evidence']}" for e in evidence_pool
        )

        chain = SUMMARY_PROMPT | self.llm
        result = chain.invoke({"raw_sources": raw_sources})
        parsed = _parse_json(result.content)

        return parsed.get("verified_sources", "")


    def _generate_response(
        self,
        question: str,
        summary: str,
        task_instruction: str = "",
        context: str = "",
    ) -> dict:
        """Generate final evidence-grounded response."""
        chain = QA_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "task_instruction": task_instruction
            or "Provide a comprehensive answer with supporting evidence.",
            "context": context or "None",
            "sources": summary,
        })
        return _parse_json(result.content)


    def generate_response(
        self,
        question: str,
        evidence_pool: list[dict],
        task_instruction: str = "",
        context: str = "",
    ) -> dict:
        """
        Stage 3: Evidence-grounded response generation.

        Returns dict with 'answer', 'rationale', 'citations'.
        """
        logger.info("\n=== Stage 3: Evidence-Grounded Response Generation ===")

        # Summary-of-Evidence
        summary = self._generate_summary(evidence_pool)
        logger.info(f"Summary of Evidence: {summary[:200]}...")

        # Final response
        response = self._generate_response(
            question, summary, task_instruction, context
        )

        # Attach citation list
        citations = [
            {"pmid": e["pmid"], "title": e["title"]}
            for e in evidence_pool
        ]

        response["citations"] = citations
        response["summary_of_evidence"] = summary

        return response
    

    # ------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------

    def run(
        self,
        question: str,
        task_instruction: str = "",
        context: str = "",
    ) -> dict:
        """
        Run the full PubMed Reasoner pipeline.

        Args:
            question: The biomedical question to answer.
            task_instruction: Optional task specification (e.g., "answer yes/no").
            context: Optional additional context.

        Returns:
            dict with 'answer', 'rationale', 'citations', 'summary_of_evidence',
            and 'query'.
        """

        # Stage 1
        query, articles = self.search_with_self_critic(question, context)

        # Stage 2
        evidence_pool = self.reflective_retrieval(question, articles, context)

        if not evidence_pool:
            logger.warning("No evidence found. Falling back to LLM-only answer.")
            return {
                "answer": "Insufficient evidence found in PubMed.",
                "rationale": "No relevant articles were retrieved.",
                "citations": [],
                "summary_of_evidence": "",
                "query": query,
            }

        # Stage 3
        response = self.generate_response(
            question, evidence_pool, task_instruction, context
        )
        response["query"] = query

        return response