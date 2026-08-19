"""Unit tests for ChatCerebras streaming."""

from unittest.mock import AsyncMock

from langchain_core.messages import HumanMessage
from pydantic import SecretStr

from langchain_cerebras import ChatCerebras

_REASONING_CHUNKS = [
    {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "reasoning": "let me think"},
                "finish_reason": None,
            }
        ],
    },
    {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-oss-120b",
        "choices": [
            {"index": 0, "delta": {"content": "hello"}, "finish_reason": "stop"}
        ],
    },
]


class _FakeAsyncStream:
    async def __aenter__(self) -> "_FakeAsyncStream":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def __aiter__(self):  # type: ignore[no-untyped-def]
        for chunk in _REASONING_CHUNKS:
            yield chunk


async def test_astream_surfaces_reasoning() -> None:
    """Async streaming must surface Cerebras reasoning deltas like sync does."""
    llm = ChatCerebras(
        model="gpt-oss-120b", api_key=SecretStr("test"), reasoning_effort="high"
    )
    mock_client = AsyncMock()
    mock_client.create = AsyncMock(return_value=_FakeAsyncStream())
    object.__setattr__(llm, "async_client", mock_client)

    reasoning = ""
    async for chunk in llm._astream([HumanMessage("hi")]):
        reasoning += chunk.message.additional_kwargs.get("reasoning", "")

    assert reasoning == "let me think"
