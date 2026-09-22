from pathlib import Path

class PromptLoader:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path("prompts")

    def load_prompt(self, name: str) -> str:
        # e.g. name="inconsistency-check-prompt" -> prompts/inconsistency-check-prompt.txt
        path = self.base_dir / f"{name}.txt"
        with path.open("r", encoding="utf-8") as f:
            return f.read()