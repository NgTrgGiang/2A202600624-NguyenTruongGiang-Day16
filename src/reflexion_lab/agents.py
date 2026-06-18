from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
from .mock_runtime import FAILURE_MODE_BY_QID
from .schemas import AttemptTrace, JudgeResult, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer


@dataclass
class BaseAgent:
    runtime: object  # MockRuntime hoặc OpenAIRuntime (cùng interface)
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        last_judge: JudgeResult | None = None
        seen_wrong: set[str] = set()
        looping = False

        for attempt_id in range(1, self.max_attempts + 1):
            # 1) Actor trả lời (có thể tham khảo reflection_memory từ các lần trước)
            answer, actor_usage = self.runtime.actor_answer(example, attempt_id, self.agent_type, reflection_memory)
            # 2) Evaluator chấm điểm
            judge, eval_usage = self.runtime.evaluator(example, answer)
            # Token/latency THẬT lấy từ phản hồi LLM (mock cũng cung cấp giá trị tương ứng)
            token_estimate = actor_usage.token_estimate + eval_usage.token_estimate
            latency_ms = actor_usage.latency_ms + eval_usage.latency_ms
            trace = AttemptTrace(attempt_id=attempt_id, answer=answer, score=judge.score, reason=judge.reason, token_estimate=token_estimate, latency_ms=latency_ms)
            final_answer = answer
            final_score = judge.score
            last_judge = judge

            if judge.score == 1:
                traces.append(trace)
                break

            # Phát hiện looping: lặp lại cùng một câu trả lời sai
            norm = normalize_answer(answer)
            if norm in seen_wrong:
                looping = True
            seen_wrong.add(norm)

            # 3) Reflexion: nếu là agent reflexion và còn lượt thử -> phản chiếu rồi ghi vào memory
            if self.agent_type == "reflexion" and attempt_id < self.max_attempts:
                reflection, ref_usage = self.runtime.reflector(example, attempt_id, judge)
                trace.token_estimate += ref_usage.token_estimate
                trace.latency_ms += ref_usage.latency_ms
                trace.reflection = reflection
                reflections.append(reflection)
                reflection_memory.append(reflection.next_strategy)

            traces.append(trace)

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        failure_mode = self._failure_mode(example, final_score, last_judge, looping)
        return RunRecord(qid=example.qid, question=example.question, gold_answer=example.gold_answer, agent_type=self.agent_type, predicted_answer=final_answer, is_correct=bool(final_score), attempts=len(traces), token_estimate=total_tokens, latency_ms=total_latency, failure_mode=failure_mode, reflections=reflections, traces=traces)

    def _failure_mode(self, example: QAExample, final_score: int, last_judge: JudgeResult | None, looping: bool) -> str:
        if final_score == 1:
            return "none"
        if looping:
            return "looping"
        if last_judge is not None and last_judge.failure_mode:
            return last_judge.failure_mode
        return FAILURE_MODE_BY_QID.get(example.qid, "wrong_final_answer")


class ReActAgent(BaseAgent):
    def __init__(self, runtime: object) -> None:
        super().__init__(runtime=runtime, agent_type="react", max_attempts=1)


class ReflexionAgent(BaseAgent):
    def __init__(self, runtime: object, max_attempts: int = 3) -> None:
        super().__init__(runtime=runtime, agent_type="reflexion", max_attempts=max_attempts)
