"""A scripted FoldingAgent backend for offline, key-free pipeline tests.

``ScriptedBackend`` replays a fixed list of turns, each a list of ``(tool_name, input)``
pairs; an empty list means a plain text turn. It satisfies ``LLMBackend`` so the real
``run_agent`` loop, controller, simulator, checkpoints and logs all run exactly as in a
model-driven run. Use it to test manifests, run logging and evaluation end to end.
"""

from __future__ import annotations

import itertools
from typing import Any

from foldingagent.backends.base import ExecutedToolResult, LLMBackend, LLMResponse, ToolCallRequest

Turn = list[tuple[str, dict[str, Any]]]


class ScriptedBackend(LLMBackend):
    model = "scripted-test"

    def __init__(self, turns: list[Turn]) -> None:
        self._turns = list(turns)
        self._ids = itertools.count(1)
        self.calls = 0

    def chat(self, messages: list[Any], system_prompt: str) -> LLMResponse:
        self.calls += 1
        turn = self._turns.pop(0) if self._turns else []
        tool_calls = [
            ToolCallRequest(id=f"call_{next(self._ids)}", name=name, input=dict(inp))
            for name, inp in turn
        ]
        return LLMResponse(
            raw=None,
            text=None if tool_calls else "done",
            tool_calls=tool_calls,
            stop_reason="tool_use" if tool_calls else "end_turn",
            ordered_content=list(tool_calls),
        )

    def init_messages(self, task_description: str) -> list[Any]:
        return [{"role": "user", "content": task_description}]

    def append_turn(
        self, messages: list[Any], response: LLMResponse, results: list[ExecutedToolResult]
    ) -> list[Any]:
        messages.append({"role": "assistant", "tool_calls": [c.name for c in response.tool_calls]})
        messages.append({"role": "user", "tool_results": [r.result.data for r in results]})
        return messages

    def append_user_text(self, messages: list[Any], response: LLMResponse, text: str) -> list[Any]:
        messages.append({"role": "assistant", "content": response.text or ""})
        messages.append({"role": "user", "content": text})
        return messages


def diagonal_fold_script(num_frames: int) -> list[Turn]:
    """Frame 1 = unfolded square; frame 2 = one diagonal fold; later frames unchanged.

    Deliberately simple: it exercises every stage of the pipeline, and its metrics
    against a real sequence are expected to be poor after step 2.
    """
    turns: list[Turn] = [[("save_checkpoint", {"frame_n": 1})]]
    if num_frames >= 2:
        turns.append(
            [("fold", {"edge": [1, 3], "direction": 1}), ("save_checkpoint", {"frame_n": 2})]
        )
    for n in range(3, num_frames + 1):
        turns.append([("save_checkpoint", {"frame_n": n})])
    return turns
