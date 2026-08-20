import pytest

from langchain_cerebras import ChatCerebras


def test_max_completion_tokens_is_preserved_without_alias():
    model = ChatCerebras(
        model="gpt-oss-120b",
        api_key="test-key",
        max_tokens=32,
        max_completion_tokens=64,
    )

    params = model._default_params

    assert params["max_completion_tokens"] == 64
    assert "max_tokens" not in params


def test_reasoning_effort_none_is_publicly_supported():
    model = ChatCerebras(
        model="gemma-4-31b",
        api_key="test-key",
        reasoning_effort="none",
    )

    assert model._default_params["reasoning_effort"] == "none"


def test_disable_reasoning_maps_to_reasoning_effort_none():
    model = ChatCerebras(
        model="gemma-4-31b",
        api_key="test-key",
        disable_reasoning=True,
    )

    with pytest.warns(DeprecationWarning, match="reasoning_effort='none'"):
        params = model._default_params

    assert params["reasoning_effort"] == "none"
    assert "disable_reasoning" not in params.get("extra_body", {})
