"""Wrapper around Cerebras' Chat Completions API."""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, cast

import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.utils import (
    from_env,
    secret_from_env,
)

# We ignore the "unused imports" here since we want to reexport these from this package.
from langchain_openai.chat_models.base import (
    BaseChatOpenAI,
)
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/"


class ChatCerebras(BaseChatOpenAI):
    r"""ChatCerebras chat model.

    Setup:
        Install ``langchain-cerebras`` and set environment variable ``CEREBRAS_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-cerebras
            export CEREBRAS_API_KEY="your-api-key"


    Key init args — completion params:
        model: str
            Name of model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        timeout: Union[float, Tuple[float, float], Any, None]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Cerebras API key. If not passed in will be read from env var CEREBRAS_API_KEY.

    Instantiate:
        .. code-block:: python

            from langchain_cerebras import ChatCerebras

            llm = ChatCerebras(
                model="llama3.1-70b",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                (
                    "system",
                    "You are a helpful translator. Translate the user sentence to French.",
                ),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='The translation of "I love programming" to French is:\n\n"J\'adore programmer."',
                response_metadata={
                    'token_usage': {'completion_tokens': 20, 'prompt_tokens': 32, 'total_tokens': 52},
                    'model_name': 'llama3.1-70b',
                    'system_fingerprint': 'fp_679dff74c0',
                    'finish_reason': 'stop',
                },
                id='run-377c2887-30ef-417e-b0f5-83efc8844f12-0',
                usage_metadata={'input_tokens': 32, 'output_tokens': 20, 'total_tokens': 52})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            content='' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='The' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' translation' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' of' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' "' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='I' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' love' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' programming' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='"' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' to' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' French' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' is' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=':\n\n' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='"' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='J' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content="'" id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='ad' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='ore' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content=' programmer' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='."' id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
            content='' response_metadata={'finish_reason': 'stop', 'model_name': 'llama3.1-70b', 'system_fingerprint': 'fp_679dff74c0'} id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(
                content='The translation of "I love programming" to French is:\n\n"J\'adore programmer."',
                response_metadata={
                    'token_usage': {'completion_tokens': 20, 'prompt_tokens': 32, 'total_tokens': 52},
                    'model_name': 'llama3.1-70b',
                    'system_fingerprint': 'fp_679dff74c0',
                    'finish_reason': 'stop',
                },
                id='run-377c2887-30ef-417e-b0f5-83efc8844f12-0',
                usage_metadata={'input_tokens': 32, 'output_tokens': 20, 'total_tokens': 52})

    Tool calling:
        .. code-block:: python

            from langchain_core.pydantic_v1 import BaseModel, Field

            llm = ChatCerebras(model="llama3.1-70b")

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke(
                "Which city is bigger: LA or NY?"
            )
            ai_msg.tool_calls


        .. code-block:: python

            [
                {
                    'name': 'GetPopulation',
                    'args': {'location': 'NY'},
                    'id': 'call_m5tstyn2004pre9bfuxvom8x',
                    'type': 'tool_call'
                },
                {
                    'name': 'GetPopulation',
                    'args': {'location': 'LA'},
                    'id': 'call_0vjgq455gq1av5sp9eb1pw6a',
                    'type': 'tool_call'
                }
            ]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from langchain_core.pydantic_v1 import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(
                setup='Why was the cat sitting on the computer?',
                punchline='To keep an eye on the mouse!',
                rating=7
            )

    JSON mode:
        .. code-block:: python

            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke(
                "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
            )
            ai_msg.content

        .. code-block:: python

            ' {\\n"random_ints": [\\n13,\\n54,\\n78,\\n45,\\n67,\\n90,\\n11,\\n29,\\n84,\\n33\\n]\\n}'

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 37, 'output_tokens': 6, 'total_tokens': 43}

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'token_usage': {
                    'completion_tokens': 4,
                    'prompt_tokens': 19,
                    'total_tokens': 23
                    },
                'model_name': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                'system_fingerprint': None,
                'finish_reason': 'eos',
                'logprobs': None
            }

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example,
            {"cerebras_api_key": "CEREBRAS_API_KEY"}
        """
        return {"cerebras_api_key": "CEREBRAS_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "cerebras"]

    @property
    def lc_attributes(self) -> Dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        attributes: Dict[str, Any] = {}

        if self.cerebras_api_base:
            attributes["cerebras_api_base"] = self.cerebras_api_base

        if self.cerebras_proxy:
            attributes["cerebras_proxy"] = self.cerebras_proxy

        return attributes

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "cerebras-chat"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "cerebras"
        return params

    model_name: str = Field(alias="model")
    """Model name to use."""
    cerebras_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("CEREBRAS_API_KEY", default=None),
    )
    """Automatically inferred from env are `CEREBRAS_API_KEY` if not provided."""
    cerebras_api_base: str = Field(
        default_factory=from_env("CEREBRAS_API_BASE", default=CEREBRAS_BASE_URL),
        alias="base_url",
    )

    cerebras_proxy: str = Field(default_factory=from_env("CEREBRAS_PROXY", default=""))

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params = {
            "api_key": (
                self.cerebras_api_key.get_secret_value()
                if self.cerebras_api_key
                else None
            ),
            # Ensure we always fallback to the Cerebras API url.
            "base_url": self.cerebras_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        if self.cerebras_proxy and (self.http_client or self.http_async_client):
            raise ValueError(
                "Cannot specify 'cerebras_proxy' if one of "
                "'http_client'/'http_async_client' is already specified. Received:\n"
                f"{self.cerebras_proxy=}\n{self.http_client=}\n{self.http_async_client=}"
            )
        if not self.client:
            if self.cerebras_proxy and not self.http_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_client = httpx.Client(proxy=self.cerebras_proxy)
            sync_specific = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)  # type: ignore
            self.client = self.root_client.chat.completions
        if not self.async_client:
            if self.cerebras_proxy and not self.http_async_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_async_client = httpx.AsyncClient(proxy=self.cerebras_proxy)
            async_specific = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,  # type: ignore
                **async_specific,  # type: ignore
            )
            self.async_client = self.root_async_client.chat.completions
        return self

    # Patch tool calling w/ streaming.
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if kwargs.get("tools"):
            yield cast(
                ChatGenerationChunk,
                super()._generate(messages, stop, run_manager, **kwargs).generations[0],
            )
        else:
            yield from super()._stream(messages, stop, run_manager, **kwargs)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        if kwargs.get("tools"):
            generation = await super()._agenerate(messages, stop, run_manager, **kwargs)
            yield (
                cast(
                    ChatGenerationChunk,
                    generation.generations[0],
                )
            )
        else:
            async for msg in super()._astream(messages, stop, run_manager, **kwargs):
                yield msg
