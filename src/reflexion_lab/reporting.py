from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from .schemas import ReportPayload, RunRecord

def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {"count": len(rows), "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4), "avg_attempts": round(mean(r.attempts for r in rows), 4), "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2), "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2)}
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {"em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4), "attempts_abs": round(summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"], 4), "tokens_abs": round(summary["reflexion"]["avg_token_estimate"] - summary["react"]["avg_token_estimate"], 2), "latency_abs": round(summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"], 2)}
    return summary

def failure_breakdown(records: list[RunRecord]) -> dict:
    # Keyed theo failure_mode (mỗi mode -> số lượng theo từng agent), để phân tích theo loại lỗi.
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.failure_mode][record.agent_type] += 1
    return {mode: dict(counter) for mode, counter in grouped.items()}

def build_discussion(summary: dict, failures: dict) -> str:
    """Sinh phần discussion theo SỐ LIỆU THẬT của lần chạy hiện tại."""
    react = summary.get("react", {})
    reflexion = summary.get("reflexion", {})
    delta = summary.get("delta_reflexion_minus_react", {})
    # Các loại lỗi còn sót lại của Reflexion (đã loại 'none')
    remaining = {mode: counts.get("reflexion", 0) for mode, counts in failures.items() if mode != "none" and counts.get("reflexion", 0) > 0}
    remaining_txt = ", ".join(f"{m} ({n})" for m, n in sorted(remaining.items(), key=lambda x: -x[1])) or "khong dang ke"
    em_r = react.get("em", 0); em_rx = reflexion.get("em", 0); d_em = delta.get("em_abs", 0)
    tok_r = react.get("avg_token_estimate", 0) or 1; tok_rx = reflexion.get("avg_token_estimate", 0)
    lat_r = react.get("avg_latency_ms", 0) or 1; lat_rx = reflexion.get("avg_latency_ms", 0)
    return (
        f"Reflexion nang Exact Match tu {em_r} (ReAct) len {em_rx}, tuc +{round(d_em*100, 2)} diem phan tram. "
        f"Co che self-reflection giup agent hoan thanh not cac hop con thieu va sua loi chon sai thuc the o lan thu sau, "
        f"the hien qua so attempt trung binh tang tu {react.get('avg_attempts', 0)} len {reflexion.get('avg_attempts', 0)}. "
        f"Doi lai la chi phi: token trung binh tang {round(tok_rx/tok_r, 2)}x ({tok_r} -> {tok_rx}) va do tre tang "
        f"{round(lat_rx/lat_r, 2)}x ({lat_r}ms -> {lat_rx}ms). "
        f"Cac failure mode con sot lai o Reflexion: {remaining_txt} -- cho thay them luot thu khong phai luc nao cung cuu duoc cau sai, "
        f"dac biet khi agent lap lai cung mot dap an (looping) hoac khi evaluator/context khong du de tim ra huong dung. "
        f"Ket luan: Reflexion dang gia khi do chinh xac quan trong hon chi phi, nhung can gioi han so attempt va co co che thoat looping."
    )


def build_report(records: list[RunRecord], dataset_name: str, mode: str = "mock") -> ReportPayload:
    examples = [{"qid": r.qid, "agent_type": r.agent_type, "gold_answer": r.gold_answer, "predicted_answer": r.predicted_answer, "is_correct": r.is_correct, "attempts": r.attempts, "failure_mode": r.failure_mode, "reflection_count": len(r.reflections)} for r in records]
    summary = summarize(records)
    failures = failure_breakdown(records)
    return ReportPayload(meta={"dataset": dataset_name, "mode": mode, "num_records": len(records), "agents": sorted({r.agent_type for r in records})}, summary=summary, failure_modes=failures, examples=examples, extensions=["structured_evaluator", "reflection_memory", "benchmark_report_json", "mock_mode_for_autograding"], discussion=build_discussion(summary, failures))

def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
