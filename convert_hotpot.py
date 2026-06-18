from __future__ import annotations
import argparse
import json
import random
from pathlib import Path


def convert(item: dict, gold_only: bool = True) -> dict:
    """Chuyển 1 item HotpotQA gốc sang format QAExample của lab."""
    # Tên các đoạn văn chứa bằng chứng (supporting facts)
    gold_titles = {t for t, _ in item.get("supporting_facts", [])}
    context = []
    for title, sentences in item["context"]:
        if gold_only and title not in gold_titles:
            continue  # chỉ giữ đoạn vàng, bỏ đoạn nhiễu (distractor)
        context.append({"title": title, "text": " ".join(sentences).strip()})
    return {
        "qid": item["_id"],
        "difficulty": item.get("level", "hard"),  # easy/medium/hard
        "question": item["question"],
        "gold_answer": item["answer"],
        "context": context,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert HotpotQA -> QAExample format")
    ap.add_argument("--src", required=True, help="Đường dẫn file hotpot_dev_distractor_v1.json")
    ap.add_argument("--out", default="data/hotpot_test_set.json", help="File output")
    ap.add_argument("--n", type=int, default=120, help="Số câu muốn lấy")
    ap.add_argument("--seed", type=int, default=42, help="Seed để tái lập")
    ap.add_argument("--all-context", action="store_true", help="Giữ cả đoạn nhiễu (distractor)")
    args = ap.parse_args()

    raw = json.loads(Path(args.src).read_text(encoding="utf-8"))
    random.seed(args.seed)
    sample = random.sample(raw, min(args.n, len(raw)))
    converted = [convert(it, gold_only=not args.all_context) for it in sample]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã ghi {len(converted)} câu -> {out}")


if __name__ == "__main__":
    main()
