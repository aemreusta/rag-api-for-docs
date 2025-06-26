#!/usr/bin/env python3
"""
Baseline Evaluation Script for Chat Endpoint

This script runs baseline evaluations of the chatbot system to establish
performance metrics for future improvements.
"""

import json
import logging
import time
from dataclasses import dataclass

import requests

from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuestion:
    """Represents a question and its expected answer for evaluation."""

    question: str
    expected_answer: str
    category: str
    difficulty: str = "medium"


@dataclass
class EvaluationResult:
    """Represents the result of evaluating a single question."""

    question: str
    actual_answer: str
    expected_answer: str
    response_time: float
    sources_count: int
    category: str
    success: bool
    error: str = None


class ChatEvaluator:
    """Evaluates the chat endpoint performance."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key or settings.API_KEY
        self.session = requests.Session()
        self.session.headers.update({"X-API-Key": self.api_key})

    def test_single_question(
        self, question: EvaluationQuestion, session_id: str = "eval-session"
    ) -> EvaluationResult:
        """Test a single question and return the evaluation result."""
        start_time = time.time()

        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/chat",
                json={"question": question.question, "session_id": session_id},
            )

            end_time = time.time()
            response_time = end_time - start_time

            if response.status_code == 200:
                data = response.json()
                return EvaluationResult(
                    question=question.question,
                    actual_answer=data.get("answer", ""),
                    expected_answer=question.expected_answer,
                    response_time=response_time,
                    sources_count=len(data.get("sources", [])),
                    category=question.category,
                    success=True,
                )
            else:
                return EvaluationResult(
                    question=question.question,
                    actual_answer="",
                    expected_answer=question.expected_answer,
                    response_time=response_time,
                    sources_count=0,
                    category=question.category,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                )

        except Exception as e:
            end_time = time.time()
            return EvaluationResult(
                question=question.question,
                actual_answer="",
                expected_answer=question.expected_answer,
                response_time=end_time - start_time,
                sources_count=0,
                category=question.category,
                success=False,
                error=str(e),
            )

    def run_evaluation(self, questions: list[EvaluationQuestion]) -> dict:
        """Run evaluation on a list of questions and return aggregated results."""
        logger.info(f"Starting evaluation with {len(questions)} questions...")

        results = []
        for i, question in enumerate(questions, 1):
            logger.info(f"Evaluating question {i}/{len(questions)}: {question.question[:50]}...")
            result = self.test_single_question(question)
            results.append(result)

            # Log result
            if result.success:
                logger.info(
                    f"✅ Success - Response time: {result.response_time:.2f}s, "
                    f"Sources: {result.sources_count}"
                )
            else:
                logger.error(f"❌ Failed - Error: {result.error}")

        # Calculate aggregated metrics
        successful_results = [r for r in results if r.success]
        total_questions = len(questions)
        successful_questions = len(successful_results)

        metrics = {
            "total_questions": total_questions,
            "successful_questions": successful_questions,
            "success_rate": successful_questions / total_questions if total_questions > 0 else 0,
            "average_response_time": sum(r.response_time for r in successful_results)
            / len(successful_results)
            if successful_results
            else 0,
            "average_sources_per_response": sum(r.sources_count for r in successful_results)
            / len(successful_results)
            if successful_results
            else 0,
            "results": [
                {
                    "question": r.question,
                    "actual_answer": r.actual_answer,
                    "expected_answer": r.expected_answer,
                    "response_time": r.response_time,
                    "sources_count": r.sources_count,
                    "category": r.category,
                    "success": r.success,
                    "error": r.error,
                }
                for r in results
            ],
        }

        return metrics


def get_baseline_questions() -> list[EvaluationQuestion]:
    """Get the baseline evaluation questions."""
    return [
        # Simple fact-finding questions
        EvaluationQuestion(
            question="What is the organization's policy on volunteer background checks?",
            expected_answer=(
                "All volunteers must complete a background check before starting. "
                "This includes criminal history and reference verification."
            ),
            category="fact-finding",
        ),
        EvaluationQuestion(
            question="How many hours per week are volunteers expected to commit?",
            expected_answer=(
                "Volunteers are expected to commit a minimum of 4 hours per week, "
                "with flexibility for scheduling based on program needs."
            ),
            category="fact-finding",
        ),
        EvaluationQuestion(
            question="What is the dress code for volunteers?",
            expected_answer=(
                "Volunteers should dress professionally and wear their provided ID badge "
                "at all times. Specific attire guidelines vary by department."
            ),
            category="fact-finding",
        ),
        # Process questions
        EvaluationQuestion(
            question="How do I request time off as a volunteer?",
            expected_answer=(
                "Volunteers should submit time-off requests at least 2 weeks in advance "
                "through the volunteer coordinator. Use the online portal or email form "
                "provided during orientation."
            ),
            category="process",
        ),
        EvaluationQuestion(
            question="What should I do if I witness inappropriate behavior?",
            expected_answer=(
                "Report any inappropriate behavior immediately to your supervisor or use "
                "the anonymous reporting hotline. All reports are taken seriously and "
                "investigated promptly."
            ),
            category="process",
        ),
        # Edge cases (should return "I don't know" responses)
        EvaluationQuestion(
            question="What is the organization's policy on cryptocurrency donations?",
            expected_answer=(
                "I don't have information about cryptocurrency donation policies in the "
                "available documents. Please contact our finance department for guidance."
            ),
            category="edge-case",
        ),
        EvaluationQuestion(
            question="Can volunteers bring their pets to work?",
            expected_answer=(
                "I don't find specific information about pet policies for volunteers in "
                "the available documents. Please check with your supervisor or HR."
            ),
            category="edge-case",
        ),
    ]


def main():
    """Run the baseline evaluation."""
    logger.info("🚀 Starting Baseline Evaluation")

    # Check if API is accessible
    try:
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            logger.info("✅ API health check passed")
        else:
            logger.error("❌ API health check failed")
            return
    except Exception as e:
        logger.error(f"❌ Cannot connect to API: {e}")
        return

    # Initialize evaluator
    evaluator = ChatEvaluator()

    # Get evaluation questions
    questions = get_baseline_questions()
    logger.info(f"📝 Loaded {len(questions)} evaluation questions")

    # Run evaluation
    results = evaluator.run_evaluation(questions)

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 BASELINE EVALUATION RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total Questions: {results['total_questions']}")
    logger.info(f"Successful: {results['successful_questions']}")
    logger.info(f"Success Rate: {results['success_rate']:.1%}")
    logger.info(f"Average Response Time: {results['average_response_time']:.2f}s")
    logger.info(f"Average Sources per Response: {results['average_sources_per_response']:.1f}")

    # Print detailed results by category
    categories = {}
    for result in results["results"]:
        category = result["category"]
        if category not in categories:
            categories[category] = []
        categories[category].append(result)

    logger.info("\n📋 Results by Category:")
    for category, cat_results in categories.items():
        successful = sum(1 for r in cat_results if r["success"])
        total = len(cat_results)
        logger.info(f"  {category.title()}: {successful}/{total} ({successful / total:.1%})")

    # Save results to file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"baseline_evaluation_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"💾 Detailed results saved to: {filename}")

    # Recommendations
    logger.info("\n💡 Recommendations:")
    if results["success_rate"] < 0.8:
        logger.info("  - Success rate below 80%. Check API configuration and error logs.")
    if results["average_response_time"] > 3.0:
        logger.info("  - Response time above 3s. Consider performance optimization.")
    if results["average_sources_per_response"] < 1.0:
        logger.info("  - Low source retrieval. Check document ingestion and embedding quality.")

    logger.info("🏁 Baseline evaluation complete!")


if __name__ == "__main__":
    main()
