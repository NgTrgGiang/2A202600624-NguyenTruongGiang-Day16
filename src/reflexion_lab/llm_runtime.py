from __future__ import annotations
import json
import time

from .prompts import ACTOR_SYSTEM, EVALUATOR_SYSTEM, REFLECTOR_SYSTEM
from .schemas import CallUsage, JudgeResult, QAExample, ReflectionEntry

_VALID_FAILURE_MODES = {"entity_drift", "incomplete_multi_hop", "wrong_final_answer"}


def _format_context(example: QAExample) -> str:
    return "\n\n".join(f"[{c.title}] {c.text}" for c in example.context)


class OpenAIRuntime:
    """Runtime gọi LLM thật qua OpenAI API. Cần biến môi trường OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv()  # đọc OPENAI_API_KEY từ file .env nếu có
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    def _chat(self, system: str, user: str, json_mode: bool = False) -> tuple[str, CallUsage]:
        kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        tokens = resp.usage.total_tokens if resp.usage else 0
        content = (resp.choices[0].message.content or "").strip()
        return content, CallUsage(token_estimate=tokens, latency_ms=latency_ms)

    def actor_answer(self, example: QAExample, attempt_id: int, agent_type: str, reflection_memory: list[str]) -> tuple[str, CallUsage]:
        parts = [f"Question: {example.question}", "", "Context:", _format_context(example)]
        if reflection_memory:
            notes = "\n".join(f"- {n}" for n in reflection_memory)
            parts += ["", "Reflection notes from previous failed attempts (follow these):", notes]
        parts += ["", "Final answer:"]
        answer, usage = self._chat(ACTOR_SYSTEM, "\n".join(parts))
        return answer, usage

    def evaluator(self, example: QAExample, answer: str) -> tuple[JudgeResult, CallUsage]:
        user = (
            f"Question: {example.question}\n"
            f"Gold answer: {example.gold_answer}\n"
            f"Predicted answer: {answer}\n"
        )
        content, usage = self._chat(EVALUATOR_SYSTEM, user, json_mode=True)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {"score": 0, "reason": f"Could not parse evaluator output: {content[:200]}"}
        score = 1 if int(data.get("score", 0)) == 1 else 0
        fm = data.get("failure_mode")
        if score == 1:
            fm = None
        elif fm not in _VALID_FAILURE_MODES:
            fm = "wrong_final_answer"
        judge = JudgeResult(
            score=score,
            reason=str(data.get("reason", "")),
            missing_evidence=list(data.get("missing_evidence") or []),
            spurious_claims=list(data.get("spurious_claims") or []),
            failure_mode=fm,
        )
        return judge, usage

    def reflector(self, example: QAExample, attempt_id: int, judge: JudgeResult) -> tuple[ReflectionEntry, CallUsage]:
        user = (
            f"Question: {example.question}\n"
            f"Grader feedback: {judge.reason}\n"
            f"Missing evidence: {judge.missing_evidence}\n"
        )
        content, usage = self._chat(REFLECTOR_SYSTEM, user, json_mode=True)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            data = {}
        entry = ReflectionEntry(
            attempt_id=attempt_id,
            failure_reason=str(data.get("failure_reason", judge.reason)),
            lesson=str(data.get("lesson", "")),
            next_strategy=str(data.get("next_strategy", "")),
        )
        return entry, usage
