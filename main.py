"""
CLI entry point for PubMed Reasoner.
"""

import argparse
import logging
import sys

from pubmed_reasoner import PubMedReasoner


def main():
    parser = argparse.ArgumentParser(
        description="PubMed Reasoner: Evidence-Grounded Biomedical QA"
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Biomedical question to answer"
    )

    parser.add_argument(
        "--task",
        default="",
        help='Task instruction (e.g., "Answer yes/no with justification")',
    )

    parser.add_argument(
        "--context",
        default="",
        help="Additional context"
    )

    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)"
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max self-critic iterations (default: 3)",
    )

    parser.add_argument(
        "--max-articles",
        type=int,
        default=20,
        help="Max articles to retrieve (default: 20)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for reflective retrieval (default: 5)",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    # Interactive mode if no question provided
    question = args.question
    if not question:
        question = input("Enter your biomedical question: ").strip()
        if not question:
            print("No question provided. Exiting.")
            sys.exit(1)

    reasoner = PubMedReasoner(
        model=args.model,
        max_self_critic_iterations=args.max_iterations,
        max_articles=args.max_articles,
        batch_size=args.batch_size,
    )

    print(f"\nQuestion: {question}")
    print("-" * 60)
    print("Running PubMed Reasoner pipeline...\n")

    result = reasoner.run(
        question=question,
        task_instruction=args.task,
        context=args.context,
    )

    # Display results
    print("-" * 60)
    print("RESULTS")
    print("-" * 60)

    print(f"\nPubMed Query: {result.get('query', 'N/A')}")
    print(f"\nAnswer: {result.get('answer')}")
    print(f"\nRationale:\n{result.get('rationale')}")

    if result.get("summary_of_evidence"):
        print(f"\nSummary of Evidence:\n{result['summary_of_evidence']}")

    if result.get("citations"):
        print("\nCitations:")
        for c in result["citations"]:
            print(f" - PMID:{c['pmid']} ({c['title']})")


if __name__ == "__main__":
    main()