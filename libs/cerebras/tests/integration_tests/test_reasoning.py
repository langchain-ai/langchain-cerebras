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
