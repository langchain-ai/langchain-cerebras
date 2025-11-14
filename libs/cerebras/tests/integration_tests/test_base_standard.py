"""Standard LangChain interface tests"""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests

from langchain_cerebras import ChatCerebras


class TestCerebrasStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatCerebras

    @property
    def chat_model_params(self) -> dict:
        return {"model": "gpt-oss-120b"}

    @pytest.mark.xfail(reason=("Array input not supported"))
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)

    @pytest.mark.xfail(
        reason=(
            "Cerebras API does not support tool_choice='required', "
            "only 'auto' or 'none' are supported"
        )
    )
    def test_structured_few_shot_examples(self, model: BaseChatModel) -> None:
        super().test_structured_few_shot_examples(model)

    @pytest.mark.xfail(
        reason=(
            "Cerebras API does not support tool_choice with specific function, "
            "only 'auto' or 'none' are supported"
        )
    )
    def test_structured_output(self, model: BaseChatModel) -> None:
        super().test_structured_output(model)

    @pytest.mark.xfail(
        reason=(
            "Cerebras API does not support tool_choice with specific function, "
            "only 'auto' or 'none' are supported"
        )
    )
    async def test_structured_output_async(self, model: BaseChatModel) -> None:
        await super().test_structured_output_async(model)

    @pytest.mark.xfail(
        reason=(
            "Cerebras API does not support tool_choice with specific function, "
            "only 'auto' or 'none' are supported"
        )
    )
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(
        reason=(
            "Cerebras API does not support tool_choice with specific function, "
            "only 'auto' or 'none' are supported"
        )
    )
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)

    @pytest.mark.xfail(
        reason="Tool calling with no arguments returns {'': {}} instead of {}"
    )
    def test_tool_calling_with_no_arguments(self, model: BaseChatModel) -> None:
        super().test_tool_calling_with_no_arguments(model)
