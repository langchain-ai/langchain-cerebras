import logging
import os

from dotenv import load_dotenv

from langchain_cerebras import ChatCerebras

# Load environment variables from root .env
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
load_dotenv(os.path.join(root_path, ".env"), override=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_gpt_oss_reasoning_content() -> None:
    """Test gpt-oss-120b reasoning content structure with a challenging problem."""
    llm = ChatCerebras(
        model="gpt-oss-120b",
        reasoning_effort="high",
        temperature=0.7,
        max_tokens=500,
    )

    # Invoke with a harder question that requires reasoning
    response = llm.invoke(
        "If a train leaves Station A at 60 mph heading east, and another train leaves "
        "Station B (120 miles east of A) at 40 mph heading west, at what time "
        "will they meet if the first train left at 2:00 PM?"
    )

    # Check if response.content is a list (structured content)
    assert isinstance(response.content, list)
    logger.info(f"Response content: {response.content}")

    has_reasoning = False
    reasoning_text = ""

    for block in response.content:
        if isinstance(block, dict):
            if block.get("type") == "reasoning_content":
                has_reasoning = True
                reasoning_text = block["reasoning_content"]["text"]
                logger.info(f"Reasoning: {reasoning_text[:200]}...")
            elif block.get("type") == "text":
                answer_text = block["text"]
                logger.info(f"Answer: {answer_text}")

    assert has_reasoning, "Expected reasoning content block"
    assert len(reasoning_text) > 20, "Reasoning should be substantial"
    # Note: With high reasoning effort, models may return only reasoning without
    # separate text content. The reasoning itself contains the answer.
    assert "reasoning" not in response.additional_kwargs


def test_gpt_oss_reasoning_streaming() -> None:
    """Test gpt-oss-120b reasoning content streaming with a challenging problem."""
    llm = ChatCerebras(
        model="gpt-oss-120b",
        reasoning_effort="medium",
        temperature=0.7,
        max_tokens=500,
    )

    full_reasoning = ""
    full_text = ""

    for chunk in llm.stream("What is the cube root of 50.653? Show your reasoning."):
        if "reasoning" in chunk.additional_kwargs:
            full_reasoning += chunk.additional_kwargs["reasoning"]

        if isinstance(chunk.content, str):
            full_text += chunk.content

    assert len(full_reasoning) > 20, "Reasoning should be substantial"
    assert len(full_text) > 0
    logger.info(f"Streamed Reasoning: {full_reasoning}")
    logger.info(f"Streamed Text: {full_text}")


def test_zai_glm_reasoning_content() -> None:
    """Test zai-glm-4.6 reasoning content structure with a challenging problem."""
    llm = ChatCerebras(
        model="zai-glm-4.6",
        disable_reasoning=False,
        temperature=0.7,
        max_tokens=500,
    )

    # Invoke with a harder question
    response = llm.invoke(
        "Explain the key differences between quantum computing and classical "
        "computing, focusing on how information is processed."
    )

    # Check if response.content is a list (structured content)
    assert isinstance(response.content, list)

    has_reasoning = False
    reasoning_text = ""

    for block in response.content:
        if isinstance(block, dict):
            if block.get("type") == "reasoning_content":
                has_reasoning = True
                reasoning_text = block["reasoning_content"]["text"]
                logger.info(f"Reasoning: {reasoning_text[:200]}...")
            elif block.get("type") == "text":
                answer_text = block["text"]
                logger.info(f"Answer: {answer_text}")

    assert has_reasoning, "Expected reasoning content block"
    assert len(reasoning_text) > 20, "Reasoning should be substantial"
    # Note: zai-glm-4.6 may return reasoning without text content
    # This is valid behavior for reasoning models
    assert "reasoning" not in response.additional_kwargs


def test_zai_glm_reasoning_streaming() -> None:
    """Test zai-glm-4.6 reasoning content streaming with a challenging problem."""
    llm = ChatCerebras(
        model="zai-glm-4.6",
        disable_reasoning=False,
        temperature=0.7,
        max_tokens=500,
    )

    full_reasoning = ""
    full_text = ""

    for chunk in llm.stream(
        "What are the ethical implications of artificial general intelligence?"
    ):
        if "reasoning" in chunk.additional_kwargs:
            full_reasoning += chunk.additional_kwargs["reasoning"]

        if isinstance(chunk.content, str):
            full_text += chunk.content

    assert len(full_reasoning) > 20, "Reasoning should be substantial"
    # Note: zai-glm-4.6 may return only reasoning without text content
    # This is valid behavior for reasoning models
    logger.info(f"Streamed Reasoning: {full_reasoning[:200]}...")
    if full_text:
        logger.info(f"Streamed Text: {full_text}")


def test_llama_no_reasoning_content() -> None:
    """Test llama3.3-70b (non-reasoning model) doesn't break."""
    llm = ChatCerebras(
        model="llama3.3-70b",
        temperature=0.7,
        max_tokens=200,
    )

    # Invoke
    response = llm.invoke("What is the capital of France?")

    # For non-reasoning models, content should be a string
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert "Paris" in response.content
    assert "reasoning" not in response.additional_kwargs
    logger.info(f"Response: {response.content}")


def test_llama_no_reasoning_streaming() -> None:
    """Test llama3.3-70b (non-reasoning model) streaming doesn't break."""
    llm = ChatCerebras(
        model="llama3.3-70b",
        temperature=0.7,
        max_tokens=200,
    )

    full_text = ""

    for chunk in llm.stream("What is the capital of France?"):
        # Should not have reasoning in additional_kwargs
        assert "reasoning" not in chunk.additional_kwargs

        if isinstance(chunk.content, str):
            full_text += chunk.content

    assert len(full_text) > 0
    assert "Paris" in full_text
    logger.info(f"Streamed Text: {full_text}")
