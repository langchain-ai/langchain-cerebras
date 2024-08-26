"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_standard_tests.unit_tests import ChatModelUnitTests

from langchain_cerebras import ChatCerebras


class TestCerebrasStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatCerebras

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama3.1-8b", "stream_usage": True}
