"""Unit tests for reasoning content extraction in ChatCerebras."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langchain_cerebras import ChatCerebras


class TestReasoningContentExtraction:
    """Test that reasoning content is properly extracted and formatted."""

    def test_should_extract_reasoning_gpt_oss_with_effort(self):
        """Test _should_extract_reasoning returns True for gpt-oss-120b with reasoning_effort."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )
        assert llm._should_extract_reasoning() is True

    def test_should_extract_reasoning_gpt_oss_without_effort(self):
        """Test _should_extract_reasoning returns False for gpt-oss-120b without reasoning_effort."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            api_key="test-key"
        )
        assert llm._should_extract_reasoning() is False

    def test_should_extract_reasoning_zai_glm_enabled(self):
        """Test _should_extract_reasoning returns True for zai-glm-4.6 with reasoning enabled."""
        llm = ChatCerebras(
            model="zai-glm-4.6",
            disable_reasoning=False,
            api_key="test-key"
        )
        assert llm._should_extract_reasoning() is True

    def test_should_extract_reasoning_zai_glm_disabled(self):
        """Test _should_extract_reasoning returns False for zai-glm-4.6 with reasoning disabled."""
        llm = ChatCerebras(
            model="zai-glm-4.6",
            disable_reasoning=True,
            api_key="test-key"
        )
        assert llm._should_extract_reasoning() is False

    def test_should_extract_reasoning_zai_glm_not_set(self):
        """Test _should_extract_reasoning returns False for zai-glm-4.6 without disable_reasoning set."""
        llm = ChatCerebras(
            model="zai-glm-4.6",
            api_key="test-key"
        )
        assert llm._should_extract_reasoning() is False

    def test_should_extract_reasoning_other_model(self):
        """Test _should_extract_reasoning returns False for non-reasoning models."""
        llm = ChatCerebras(
            model="llama-3.3-70b",
            api_key="test-key"
        )
        assert llm._should_extract_reasoning() is False

    def test_extract_reasoning_from_response(self):
        """Test extracting reasoning field from API response."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        response_dict = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42",
                        "reasoning": "To find the answer, I analyzed the question and determined that 42 is the answer to life, the universe, and everything."
                    }
                }
            ]
        }

        reasoning = llm._extract_reasoning_from_response(response_dict)
        assert reasoning == "To find the answer, I analyzed the question and determined that 42 is the answer to life, the universe, and everything."

    def test_extract_reasoning_from_response_no_reasoning(self):
        """Test extracting reasoning when not present in response."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        response_dict = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 42"
                    }
                }
            ]
        }

        reasoning = llm._extract_reasoning_from_response(response_dict)
        assert reasoning is None

    def test_extract_reasoning_from_malformed_response(self):
        """Test extracting reasoning from malformed response."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        # Empty response
        assert llm._extract_reasoning_from_response({}) is None

        # No choices
        assert llm._extract_reasoning_from_response({"choices": []}) is None

        # No message
        assert llm._extract_reasoning_from_response({"choices": [{}]}) is None

    def test_add_reasoning_to_message_with_string_content(self):
        """Test adding reasoning block to message with string content."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        original_message = AIMessage(content="The answer is 42")
        reasoning = "I analyzed the question carefully."

        updated_message = llm._add_reasoning_to_message(original_message, reasoning)

        # Check that content is now a list
        assert isinstance(updated_message.content, list)
        assert len(updated_message.content) == 2

        # Check reasoning block comes first
        assert updated_message.content[0]["type"] == "reasoning_content"
        assert updated_message.content[0]["reasoning_content"]["text"] == reasoning

        # Check text block comes second
        assert updated_message.content[1]["type"] == "text"
        assert updated_message.content[1]["text"] == "The answer is 42"

    def test_add_reasoning_to_message_with_empty_content(self):
        """Test adding reasoning block to message with empty string content."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        original_message = AIMessage(content="")
        reasoning = "I analyzed the question carefully."

        updated_message = llm._add_reasoning_to_message(original_message, reasoning)

        # Should only have reasoning block, no empty text block
        assert isinstance(updated_message.content, list)
        assert len(updated_message.content) == 1
        assert updated_message.content[0]["type"] == "reasoning_content"

    def test_add_reasoning_to_message_with_list_content(self):
        """Test adding reasoning block to message with list content."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        original_message = AIMessage(
            content=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"}
            ]
        )
        reasoning = "I analyzed the question carefully."

        updated_message = llm._add_reasoning_to_message(original_message, reasoning)

        # Check that content has reasoning first, then original blocks
        assert isinstance(updated_message.content, list)
        assert len(updated_message.content) == 3

        # Reasoning block first
        assert updated_message.content[0]["type"] == "reasoning_content"
        assert updated_message.content[0]["reasoning_content"]["text"] == reasoning

        # Original blocks follow
        assert updated_message.content[1] == {"type": "text", "text": "Part 1"}
        assert updated_message.content[2] == {"type": "text", "text": "Part 2"}

    def test_add_reasoning_to_message_empty_reasoning(self):
        """Test adding empty reasoning doesn't modify message."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        original_message = AIMessage(content="The answer is 42")
        updated_message = llm._add_reasoning_to_message(original_message, "")

        # Message should be unchanged
        assert updated_message.content == "The answer is 42"

    def test_reasoning_content_structure(self):
        """Test the structure of reasoning content blocks."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        message = AIMessage(content="Answer")
        reasoning = "Reasoning text"

        updated = llm._add_reasoning_to_message(message, reasoning)

        reasoning_block = updated.content[0]

        # Verify structure matches AWS Bedrock pattern
        assert "type" in reasoning_block
        assert reasoning_block["type"] == "reasoning_content"
        assert "reasoning_content" in reasoning_block
        assert "text" in reasoning_block["reasoning_content"]
        assert reasoning_block["reasoning_content"]["text"] == reasoning


class TestReasoningContentIntegration:
    """Integration-style tests for reasoning content (mocking API responses)."""

    def test_create_chat_result_adds_reasoning_for_gpt_oss(self, monkeypatch):
        """Test that _create_chat_result adds reasoning for gpt-oss-120b."""
        llm = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            api_key="test-key"
        )

        # Mock API response with reasoning
        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The cube root of 50.653 is approximately 3.7.",
                        "reasoning": "To find the cube root of 50.653, I'll estimate: 3^3 = 27, 4^3 = 64. Since 50.653 is between these values, the answer is between 3 and 4, closer to 3.7."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

        result = llm._create_chat_result(mock_response)

        # Check the result
        assert len(result.generations) == 1
        message = result.generations[0].message

        # Content should be a list with reasoning and text blocks
        assert isinstance(message.content, list)
        assert len(message.content) == 2

        # First block should be reasoning
        assert message.content[0]["type"] == "reasoning_content"
        assert "To find the cube root" in message.content[0]["reasoning_content"]["text"]

        # Second block should be text
        assert message.content[1]["type"] == "text"
        assert "approximately 3.7" in message.content[1]["text"]

    def test_create_chat_result_no_reasoning_for_other_models(self, monkeypatch):
        """Test that reasoning is NOT added for non-reasoning models."""
        llm = ChatCerebras(
            model="llama-3.3-70b",  # Non-reasoning model
            api_key="test-key"
        )

        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "llama-3.3-70b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello, how can I help you?",
                        "reasoning": "This should be ignored"  # Even if present, should be ignored
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

        result = llm._create_chat_result(mock_response)

        message = result.generations[0].message

        # Content should remain a string, not converted to blocks
        assert isinstance(message.content, str)
        assert message.content == "Hello, how can I help you?"

    def test_create_chat_result_no_reasoning_when_param_not_set(self):
        """Test that reasoning is NOT added when reasoning_effort is not set."""
        llm = ChatCerebras(
            model="gpt-oss-120b",  # Reasoning model but param not set
            api_key="test-key"
        )

        mock_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "gpt-oss-120b",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer",
                        "reasoning": "Should be ignored"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }

        result = llm._create_chat_result(mock_response)
        message = result.generations[0].message

        # Should remain string since reasoning_effort not set
        assert isinstance(message.content, str)
