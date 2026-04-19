import json
import logging
import random
from dataclasses import dataclass
from typing import List, Dict

from langchain_openai import ChatOpenAI
from evaluation.prompts import LIKERT_JUDGE_PROMPT, PAIRWISE_JUDGE_PROMPT

logger = logging.getLogger(__name__)

DIMENSIONS = [
    "Reasoning Soundness",
    "Evidence Grounding",
    "Clinical Relevance",
    "Trustworthiness",
]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


@dataclass
class PairwiseResult:
    verdict: str  # "A", "B", or "tie"
    scores_a: Dict[str, dict]
    scores_b: Dict[str, dict]
    swapped: bool  # whether A/B order was randomized


@dataclass
class LikertResult:
    scores: Dict[str, dict]  # dimension -> {score, justification}

    @property
    def average(self) -> float:
        return sum(d["score"] for d in self.scores.values()) / len(self.scores)


class Judge:
    """LLM-as-Judge for biomedical QA evaluation."""

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        randomize_order: bool = True,
    ):
        """
        Args:
            model: Judge LLM model.
            temperature: Sampling temperature.
            randomize_order: If True, swap A/B to reduce positional bias.
        """
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.randomize_order = randomize_order

    def pairwise(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
    ) -> PairwiseResult:
        """Pairwise comparison of two answers."""

        swapped = False
        if self.randomize_order and random.random() < 0.5:
            answer_a, answer_b = answer_b, answer_a
            swapped = True

        chain = PAIRWISE_JUDGE_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "answer_a": answer_a,
            "answer_b": answer_b,
        })

        parsed = _parse_json(result.content)

        verdict = parsed.get("verdict", "tie")
        scores_a = parsed.get("Answer A", {})
        scores_b = parsed.get("Answer B", {})

        # Map back to original ordering
        if swapped:
            scores_a, scores_b = scores_b, scores_a
            if verdict == "A":
                verdict = "B"
            elif verdict == "B":
                verdict = "A"

        return PairwiseResult(
            verdict=verdict,
            scores_a=scores_a,
            scores_b=scores_b,
            swapped=swapped,
        )

    def likert(self, question: str, answer: str) -> LikertResult:
        """Independent Likert scoring of a single answer."""

        chain = LIKERT_JUDGE_PROMPT | self.llm
        result = chain.invoke({
            "natural_language_question": question,
            "answer": answer,
        })

        parsed = _parse_json(result.content)
        return LikertResult(scores=parsed)

    def evaluate_dataset(
        self,
        records: List[dict],
        model_a_key: str = "answer_a",
        model_b_key: str = "answer_b",
        question_key: str = "question",
    ) -> dict:
        """
        Run pairwise evaluation over a dataset.

        Each record must contain:
            - question
            - answer_a
            - answer_b
        """

        wins_a, wins_b, ties = 0, 0, 0
        all_scores_a = {d: [] for d in DIMENSIONS}
        all_scores_b = {d: [] for d in DIMENSIONS}

        for i, record in enumerate(records):
            logger.info(f"Evaluating record {i + 1}/{len(records)}")

            result = self.pairwise(
                question=record[question_key],
                answer_a=record[model_a_key],
                answer_b=record[model_b_key],
            )

            if result.verdict == "A":
                wins_a += 1
            elif result.verdict == "B":
                wins_b += 1
            else:
                ties += 1

            for dim in DIMENSIONS:
                if dim in result.scores_a:
                    all_scores_a[dim].append(result.scores_a[dim]["score"])
                if dim in result.scores_b:
                    all_scores_b[dim].append(result.scores_b[dim]["score"])

        total = len(records)

        avg_a = {
            d: (sum(scores) / len(scores) if scores else 0.0)
            for d, scores in all_scores_a.items()
        }

        avg_b = {
            d: (sum(scores) / len(scores) if scores else 0.0)
            for d, scores in all_scores_b.items()
        }

        return {
            "total": total,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "win_rate_a": wins_a / total if total else 0,
            "win_rate_b": wins_b / total if total else 0,
            "tie_rate": ties / total if total else 0,
            "avg_likert_a": avg_a,
            "avg_likert_b": avg_b,
        }