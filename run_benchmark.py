from __future__ import annotations
import json
from pathlib import Path
import typer
from rich import print
from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl
app = typer.Typer(add_completion=False)

@app.command()
def main(dataset: str = "data/hotpot_mini.json", out_dir: str = "outputs/sample_run", reflexion_attempts: int = 3, mode: str = "mock", model: str = "gpt-4o-mini") -> None:
    examples = load_dataset(dataset)
    if mode == "openai":
        from src.reflexion_lab.llm_runtime import OpenAIRuntime
        runtime = OpenAIRuntime(model=model)
    else:
        from src.reflexion_lab.mock_runtime import MockRuntime
        runtime = MockRuntime()
    react = ReActAgent(runtime)
    reflexion = ReflexionAgent(runtime, max_attempts=reflexion_attempts)
    from rich.progress import track
    react_records = [react.run(example) for example in track(examples, description="ReAct    ")]
    reflexion_records = [reflexion.run(example) for example in track(examples, description="Reflexion")]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(all_records, dataset_name=Path(dataset).name, mode=mode)
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))

if __name__ == "__main__":
    app()
