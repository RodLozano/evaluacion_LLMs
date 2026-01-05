# src/memory.py
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


Message = Dict[str, Any]  # {"role": "...", "content": "..."}


def _msg_len(m: Message) -> int:
    # Simple proxy: character count (good enough for basic length control)
    content = m.get("content", "")
    if isinstance(content, str):
        return len(content)
    # If content is structured, approximate by stringified length
    return len(str(content))


@dataclass
class ChatMemory:
    """
    Minimal conversation memory.
    Stores messages in OpenAI-style role format and trims by:
    - max_turns (user+assistant pairs)
    - max_chars (approx token control)
    """
    system_prompt: str = "Eres un asistente turístico útil y preciso."
    max_turns: int = 8
    max_chars: int = 12000  # approx context budget
    messages: List[Message] = field(default_factory=list)
    summary: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.messages:
            self.messages = [{"role": "system", "content": self.system_prompt}]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.summary = None

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self._trim()

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def add_tool_result(self, text: str) -> None:
        """
        For tool outputs you want in the conversation context.
        Keep it as assistant message for simplicity.
        """
        self.messages.append({"role": "assistant", "content": text})
        self._trim()

    def get_messages(self) -> List[Message]:
        """
        Returns messages ready to send to the LLM.
        If a summary exists, it gets injected after the system prompt.
        """
        if not self.summary:
            return list(self.messages)

        out: List[Message] = []
        out.append(self.messages[0])  # system
        out.append(
            {
                "role": "assistant",
                "content": f"Resumen de la conversación hasta ahora:\n{self.summary}",
            }
        )
        out.extend(self.messages[1:])
        return out

    # ---------------------------
    # Trimming logic
    # ---------------------------

    def _trim(self) -> None:
        """
        Enforce max_turns and max_chars.
        Keeps system prompt always.
        """
        self._trim_by_turns()
        self._trim_by_chars()

    def _trim_by_turns(self) -> None:
        """
        Keep only last max_turns user+assistant pairs (approx).
        """
        if self.max_turns <= 0:
            return

        # Keep system + last N*2 messages (user+assistant pairs)
        # This is a simplification; conversations may include tool results.
        keep_tail = self.max_turns * 2
        if len(self.messages) <= 1 + keep_tail:
            return

        system = self.messages[0:1]
        tail = self.messages[-keep_tail:]
        self.messages = system + tail

    def _trim_by_chars(self) -> None:
        if self.max_chars <= 0:
            return

        total = sum(_msg_len(m) for m in self.messages)
        if total <= self.max_chars:
            return

        # If we exceed, drop oldest non-system messages until within budget
        # and optionally create/update a simple summary placeholder.
        system = self.messages[0]
        rest = self.messages[1:]

        # OPTIONAL: create a very simple "summary" placeholder when trimming.
        # You can later replace this with an LLM summarization step.
        dropped: List[Message] = []

        while rest and (sum(_msg_len(m) for m in [system] + rest) > self.max_chars):
            dropped.append(rest.pop(0))

        if dropped:
            self.summary = self._naive_summary(dropped, prior=self.summary)

        self.messages = [system] + rest

    def _naive_summary(self, dropped: List[Message], prior: Optional[str] = None) -> str:
        """
        Ultra-simple summary: concatenates first lines of dropped messages.
        Replace with LLM summarization later if you want.
        """
        lines: List[str] = []
        if prior:
            lines.append(prior.strip())

        for m in dropped:
            role = m.get("role", "unknown")
            content = m.get("content", "")
            if isinstance(content, str):
                snippet = content.strip().replace("\n", " ")
            else:
                snippet = str(content).replace("\n", " ")
            if len(snippet) > 180:
                snippet = snippet[:180] + "…"
            lines.append(f"- ({role}) {snippet}")

        # Limit summary length to avoid it growing forever
        joined = "\n".join(lines).strip()
        max_summary_chars = int(os.getenv("MEMORY_MAX_SUMMARY_CHARS", "2000"))
        if len(joined) > max_summary_chars:
            joined = joined[-max_summary_chars:]
        return joined
    


'''
COMO USARLO EN EL NOTEBOOK

from src.memory import ChatMemory

mem = ChatMemory(
    system_prompt="Eres un asistente turístico que cita fuentes.",
    max_turns=6,
    max_chars=8000,
)

mem.add_user("Quiero un plan para 2 días.")
mem.add_assistant("Claro, ¿qué ciudad y qué fechas?")
print(mem.get_messages())

'''