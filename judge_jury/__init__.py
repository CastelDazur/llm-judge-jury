"""llm-judge-jury: score instruction/response pairs with a jury of independent LLM judges."""

from judge_jury.protocol import Verdict, parse_verdict, VERDICT_INSTRUCTIONS

__version__ = "0.1.0"
__all__ = ["Verdict", "parse_verdict", "VERDICT_INSTRUCTIONS"]
