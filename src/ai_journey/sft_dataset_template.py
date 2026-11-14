"""Unambiguous ChatML rendering with explicit assistant-only loss spans."""

from dataclasses import dataclass
from hashlib import sha256
import json

from .sft_dataset_format import ChatExample
from .sft_dataset_record import SFTDatasetError


@dataclass(frozen=True)
class RenderedChat:
    """ChatML text plus the character spans that carry training loss.

    Construction requires the <|end|> terminator and ordered, nonempty,
    in-bounds spans; render_chatml spans exactly the quoted assistant reply.
    """

    example_id: str
    text: str
    assistant_spans: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.endswith("<|end|>\n"):
            raise SFTDatasetError("rendered chat must end with the ChatML terminator")
        previous_end = 0
        for span in self.assistant_spans:
            if (
                not isinstance(span, tuple)
                or len(span) != 2
                or any(type(value) is not int for value in span)
                or not previous_end <= span[0] < span[1] <= len(self.text)
            ):
                raise SFTDatasetError(
                    "assistant spans must be ordered nonempty intervals"
                )
            previous_end = span[1]

    @property
    def sha256(self) -> str:
        """Digest of the exact rendered text a model would be trained on."""
        return sha256(self.text.encode("utf-8")).hexdigest()


def _quoted_content(content: str) -> str:
    return json.dumps(content, ensure_ascii=False, separators=(",", ":"))


def render_chatml(example: ChatExample) -> RenderedChat:
    """Render role markers while JSON-escaping marker-like source text."""
    if not isinstance(example, ChatExample):
        raise TypeError("example must be ChatExample")
    user, assistant = example.messages
    prefix = f"<|user|>\n{_quoted_content(user.content)}\n<|assistant|>\n"
    target = _quoted_content(assistant.content)
    suffix = "\n<|end|>\n"
    start = len(prefix)
    return RenderedChat(
        example_id=example.example_id,
        text=prefix + target + suffix,
        assistant_spans=((start, start + len(target)),),
    )


def assistant_character_mask(rendered: RenderedChat) -> tuple[bool, ...]:
    """Per-character mask, True only in assistant spans, so prompts get no loss."""
    if not isinstance(rendered, RenderedChat):
        raise TypeError("rendered must be RenderedChat")
    mask = [False] * len(rendered.text)
    for start, end in rendered.assistant_spans:
        mask[start:end] = [True] * (end - start)
    return tuple(mask)
