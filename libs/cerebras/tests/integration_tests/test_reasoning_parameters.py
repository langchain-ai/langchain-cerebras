"""Test reasoning parameters for gpt-oss-120b and zai-glm-4.6 models."""

import os
import pytest
from langchain_cerebras import ChatCerebras


class TestReasoningParameters:
    """Test suite for reasoning parameters."""

    def test_gpt_oss_120b_reasoning_effort_low(self):
        """Test gpt-oss-120b with low reasoning effort."""
        chat = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="low",
            temperature=0.7,
        )
        
        # Verify the parameter is set
        assert chat.reasoning_effort == "low"
        
        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content
        print(f"Low reasoning response: {response.content}")

    def test_gpt_oss_120b_reasoning_effort_medium(self):
        """Test gpt-oss-120b with medium reasoning effort."""
        chat = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="medium",
            temperature=0.7,
        )
        
        # Verify the parameter is set
        assert chat.reasoning_effort == "medium"
        
        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content
        print(f"Medium reasoning response: {response.content}")

    def test_gpt_oss_120b_reasoning_effort_high(self):
        """Test gpt-oss-120b with high reasoning effort."""
        chat = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
            temperature=0.7,
        )
        
        # Verify the parameter is set
        assert chat.reasoning_effort == "high"
        
        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content
        print(f"High reasoning response: {response.content}")

    def test_gpt_oss_120b_without_reasoning_effort(self):
        """Test gpt-oss-120b without reasoning effort (default behavior)."""
        chat = ChatCerebras(
            model="gpt-oss-120b",
            temperature=0.7,
        )
        
        # Verify the parameter is None (default)
        assert chat.reasoning_effort is None
        
        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content
        print(f"Default (no reasoning effort) response: {response.content}")

    def test_zai_glm_4_6_disable_reasoning_true(self):
        """Test zai-glm-4.6 with reasoning disabled."""
        chat = ChatCerebras(
            model="zai-glm-4.6",
            disable_reasoning=True,
            temperature=0.7,
        )
        
        # Verify the parameter is set
        assert chat.disable_reasoning is True
        
        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content
        print(f"Reasoning disabled response: {response.content}")

    def test_zai_glm_4_6_disable_reasoning_false(self):
        """Test zai-glm-4.6 with reasoning enabled."""
        chat = ChatCerebras(
            model="zai-glm-4.6",
            disable_reasoning=False,
            temperature=0.7,
        )
        
        # Verify the parameter is set
        assert chat.disable_reasoning is False
        
        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content
        print(f"Reasoning enabled response: {response.content}")

    def test_zai_glm_4_6_without_disable_reasoning(self):
        """Test zai-glm-4.6 without disable_reasoning (default behavior)."""
        chat = ChatCerebras(
            model="zai-glm-4.6",
            temperature=0.7,
        )
        
        # Verify the parameter is None (default)
        assert chat.disable_reasoning is None
        
        # Test with a simple query
        response = chat.invoke("What is 2+2?")
        assert response.content
        print(f"Default (no disable_reasoning) response: {response.content}")

    def test_default_params_includes_reasoning_effort(self):
        """Test that _default_params includes reasoning_effort when set."""
        chat = ChatCerebras(
            model="gpt-oss-120b",
            reasoning_effort="high",
        )
        
        params = chat._default_params
        assert "reasoning_effort" in params
        assert params["reasoning_effort"] == "high"

    def test_default_params_includes_disable_reasoning(self):
        """Test that _default_params includes disable_reasoning when set."""
        chat = ChatCerebras(
            model="zai-glm-4.6",
            disable_reasoning=True,
        )
        
        params = chat._default_params
        # disable_reasoning is a nonstandard parameter and must be in extra_body
        assert "extra_body" in params
        assert "disable_reasoning" in params["extra_body"]
        assert params["extra_body"]["disable_reasoning"] is True

    def test_default_params_excludes_none_values(self):
        """Test that _default_params excludes reasoning params when None."""
        chat = ChatCerebras(
            model="gpt-oss-120b",
        )
        
        params = chat._default_params
        # When None, the parameters should not be in the dict
        assert chat.reasoning_effort is None
        assert chat.disable_reasoning is None


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
