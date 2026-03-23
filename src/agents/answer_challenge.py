"""
answer_challenge.py
-------------------
CLI entry point for the Drilling Intelligence System.
Uses the lean orchestrator (1-2 LLM calls) instead of CrewAI (10+ LLM calls).
"""
import sys
import logging
from pathlib import Path
from src.agents.orchestrator import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main(question: str):
    print("\n" + "=" * 70)
    print("⛽  DRILLING INTELLIGENCE SYSTEM")
    print("=" * 70)
    print(f"\nQuestion: {question}\n")
    print("-" * 70)

    answer, needs, evidence, steps = run_pipeline(question)

    print("\n" + "=" * 70)
    print("📄 FINAL REPORT")
    print("=" * 70)
    print(answer)

    # Save to file
    out_path = Path("challenge_output.md")
    out_path.write_text(answer, encoding="utf-8")
    print(f"\n💾 Report saved to {out_path.absolute()}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/agents/answer_challenge.py \"<Your Question>\"")
        print('Example: python src/agents/answer_challenge.py "What is rate of penetration?"')
        sys.exit(1)

    main(sys.argv[1])
