"""Standard LangChain interface tests"""

import os
from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_cerebras import ChatCerebras

os.environ["CEREBRAS_API_KEY"] = "foo"


class TestCerebrasStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatCerebras

    @property
    def chat_model_params(self) -> dict:
        return {"model": "llama3.1-8b"}
