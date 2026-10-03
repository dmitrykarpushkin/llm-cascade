"""LLM backends for the filter and writer stages, plus tolerant JSON parsing of model output."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Protocol


class LLM(Protocol):
    def complete(self, prompt: str, model: str) -> str: ...


class LLMError(RuntimeError):
    pass


class ClaudeCLI:
    """Runs `claude -p` (Claude Code headless): uses the CLI's own login, no API key in this process.

    All tools are disabled, so the model can only answer from the prompt.
    """

    DISALLOWED = "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,Agent"

    def __init__(self, binary: str = "claude", timeout: float = 1500, cwd: str | None = None, run=subprocess.run):
        self.binary, self.timeout, self.cwd, self.run = binary, timeout, cwd, run

    def complete(self, prompt: str, model: str) -> str:
        argv = [self.binary, "-p", "--model", model, "--disallowedTools", self.DISALLOWED, "--max-turns", "2"]
        r = self.run(argv, input=prompt, text=True, capture_output=True, cwd=self.cwd, timeout=self.timeout)
        if r.returncode != 0 or not r.stdout.strip():
            raise LLMError(f"claude -p ({model}) failed ({r.returncode}): {(r.stderr or r.stdout)[-300:]}")
        return r.stdout.strip()


def parse_json_array(text: str, required: str = "n") -> list[dict]:
    """The first [...] in the text as a list of objects that have `required`; [] when there is none."""
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    return [x for x in data if isinstance(x, dict) and required in x] if isinstance(data, list) else []
