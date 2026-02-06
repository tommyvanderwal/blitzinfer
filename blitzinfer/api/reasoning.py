"""Reasoning content extraction for thinking models.

Thinking models (Kimi-VL, Qwen3, DeepSeek R1, GLM, etc.) wrap internal
reasoning in think tags. This module extracts reasoning from content,
matching vLLM's reasoning parser behavior for the OpenAI API:
  - reasoning goes in message.reasoning_content
  - final answer goes in message.content

Supports multiple tag formats:
  - <think>...</think>     (Qwen3, DeepSeek R1, GLM)
  - ◁think▷...◁/think▷     (Kimi-VL, Unicode U+25C1/U+25B7)
"""

from typing import Optional, Tuple

THINK_TAG_PAIRS = [
    ("<think>", "</think>"),       # Qwen3, DeepSeek R1, GLM
    ("◁think▷", "◁/think▷"),       # Kimi-VL
]


def extract_reasoning_content(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract reasoning from thinking model output.

    Returns (reasoning, content) where:
      - reasoning: text between think tags (or None if no tags / empty)
      - content: text after closing think tag (or None if truncated mid-reasoning)
      - If no think tags found, returns (None, original_text)
    """
    for start_tag, end_tag in THINK_TAG_PAIRS:
        if start_tag in text:
            # Split on start tag
            _before, _, after_start = text.partition(start_tag)
            if end_tag in after_start:
                reasoning, _, content = after_start.partition(end_tag)
                return reasoning.strip() or None, content.strip() or None
            else:
                # Start tag but no end tag - entire output is reasoning (truncated)
                return after_start.strip() or None, None
    return None, text
