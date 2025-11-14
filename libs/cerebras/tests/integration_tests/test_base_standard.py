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
        return {"model": "llama-3.1-8b"}

    @pytest.mark.xfail(reason=("Array input not supported"))
    def test_tool_message_histories_list_content(self, model: BaseChatModel) -> None:
        super().test_tool_message_histories_list_content(model)
