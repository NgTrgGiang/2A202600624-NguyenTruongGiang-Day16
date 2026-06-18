from __future__ import annotations
from .schemas import CallUsage, JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer

# Các câu cố tình sai ở lần thử đầu để minh hoạ lợi ích của Reflexion.
FIRST_ATTEMPT_WRONG = {"hp2": "London", "hp4": "Atlantic Ocean", "hp6": "Red Sea", "hp8": "Andes"}
FAILURE_MODE_BY_QID = {"hp2": "incomplete_multi_hop", "hp4": "wrong_final_answer", "hp6": "entity_drift", "hp8": "entity_drift"}


class MockRuntime:
    """Runtime giả lập — deterministic, không gọi LLM. Dùng cho mock mode / autograding."""

    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, CallUsage]:
        usage = CallUsage(token_estimate=220 + attempt_id * 40, latency_ms=120 + attempt_id * 30)
        if example.qid not in FIRST_ATTEMPT_WRONG:
            return example.gold_answer, usage
        if agent_type == "react":
            return FIRST_ATTEMPT_WRONG[example.qid], usage
        if attempt_id == 1 and not reflection_memory:
            return FIRST_ATTEMPT_WRONG[example.qid], usage
        return example.gold_answer, usage

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, CallUsage]:
        usage = CallUsage(token_estimate=90, latency_ms=40)
        if normalize_answer(example.gold_answer) == normalize_answer(answer):
            return JudgeResult(score=1, reason="Final answer matches the gold answer after normalization."), usage
        if normalize_answer(answer) == "london":
            return JudgeResult(score=0, reason="The answer stopped at the birthplace city and never completed the second hop to the river.", missing_evidence=["Need to identify the river that flows through London."], spurious_claims=[]), usage
        return JudgeResult(score=0, reason="The final answer selected the wrong second-hop entity.", missing_evidence=["Need to ground the answer in the second paragraph."], spurious_claims=[answer]), usage

    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, CallUsage]:
        usage = CallUsage(token_estimate=120, latency_ms=90)
        strategy = "Do the second hop explicitly: birthplace city -> river through that city." if example.qid == "hp2" else "Verify the final entity against the second paragraph before answering."
        entry = ReflectionEntry(attempt_id=attempt_id, failure_reason=judge.reason, lesson="A partial first-hop answer is not enough; the final answer must complete all hops.", next_strategy=strategy)
        return entry, usage
