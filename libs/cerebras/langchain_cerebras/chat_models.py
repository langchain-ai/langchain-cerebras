"""Wrapper around Cerebras' Chat Completions API."""

import warnings
from typing import Any, Dict, Iterator, List, Literal, Optional, Type, Union

import openai
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import (
    from_env,
    secret_from_env,
)

# We ignore the "unused imports" here since we want to reexport these from this package.
from langchain_openai.chat_models.base import (
    BaseChatOpenAI,
    _convert_chunk_to_generation_chunk,
    _handle_openai_bad_request,
)
from pydantic import Field, SecretStr, model_validator
from typing_extensions import Self

CEREBRAS_BASE_URL = "https://api.cerebras.ai/v1/"


class ChatCerebras(BaseChatOpenAI):
    r"""ChatCerebras chat model.

    Setup:
        Install `langchain-cerebras` and set environment variable `CEREBRAS_API_KEY`.

        ```bash
        pip install -U langchain-cerebras
        export CEREBRAS_API_KEY="your-api-key"
        ```


    Key init args — completion params:
        model: str
            Name of model to use.
        temperature: Optional[float]
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.
        reasoning_effort: Optional[Literal["low", "medium", "high"]]
            Level of reasoning effort for gpt-oss-120b model.
        disable_reasoning: Optional[bool]
            Whether to disable reasoning for zai-glm-4.6 model.

    Key init args — client params:
        timeout: Union[float, Tuple[float, float], Any, None]
            Timeout for requests.
        max_retries: Optional[int]
            Max number of retries.
        api_key: Optional[str]
            Cerebras API key. If not passed in will be read from env var `CEREBRAS_API_KEY`.

    Instantiate:
        ```python
        from langchain_cerebras import ChatCerebras

        llm = ChatCerebras(
            model="llama-3.3-70b",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        llm.invoke(messages)
        ```

        ```python
        AIMessage(
            content='The translation of "I love programming" to French is:\n\n"J\'adore programmer."',
            response_metadata={
                'token_usage': {'completion_tokens': 20, 'prompt_tokens': 32, 'total_tokens': 52},
                'model_name': 'llama-3.3-70b',
                'system_fingerprint': 'fp_679dff74c0',
                'finish_reason': 'stop',
            },
            id='run-377c2887-30ef-417e-b0f5-83efc8844f12-0',
            usage_metadata={'input_tokens': 32, 'output_tokens': 20, 'total_tokens': 52})
        ```

    Stream:
        ```python
        for chunk in llm.stream(messages):
            print(chunk)
        ```

        ```python
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
        content='' response_metadata={'finish_reason': 'stop', 'model_name': 'llama-3.3-70b', 'system_fingerprint': 'fp_679dff74c0'} id='run-3f9dc84e-208f-48da-b15d-e552b6759c24'
        ```

    Async:
        ```python
        await llm.ainvoke(messages)

        # stream:
        # async for chunk in (await llm.astream(messages))

        # batch:
        # await llm.abatch([messages])
        ```

        ```python
        AIMessage(
            content='The translation of "I love programming" to French is:\n\n"J\'adore programmer."',
            response_metadata={
                'token_usage': {'completion_tokens': 20, 'prompt_tokens': 32, 'total_tokens': 52},
                'model_name': 'llama-3.3-70b',
                'system_fingerprint': 'fp_679dff74c0',
                'finish_reason': 'stop',
            },
            id='run-377c2887-30ef-417e-b0f5-83efc8844f12-0',
            usage_metadata={'input_tokens': 32, 'output_tokens': 20, 'total_tokens': 52})
        ```

    Tool calling:
        ```python
        from langchain_core.pydantic_v1 import BaseModel, Field

        llm = ChatCerebras(model="llama-3.3-70b")

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
        ```

        ```python
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
        ```

    Structured output:
        ```python
        from typing import Optional

        from langchain_core.pydantic_v1 import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


        structured_llm = llm.with_structured_output(Joke)
        structured_llm.invoke("Tell me a joke about cats")
        ```

        ```python
        Joke(
            setup='Why was the cat sitting on the computer?',
            punchline='To keep an eye on the mouse!',
            rating=7
        )
        ```

    JSON mode:
        ```python
        json_llm = llm.bind(response_format={"type": "json_object"})
        ai_msg = json_llm.invoke(
            "Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]"
        )
        ai_msg.content
        ```

        ```python
        ' {\\n"random_ints": [\\n13,\\n54,\\n78,\\n45,\\n67,\\n90,\\n11,\\n29,\\n84,\\n33\\n]\\n}'
        ```

    Token usage:
        ```python
        ai_msg = llm.invoke(messages)
        ai_msg.usage_metadata
        ```

        ```python
        {'input_tokens': 37, 'output_tokens': 6, 'total_tokens': 43}
        ```

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

    Reasoning with gpt-oss-120b:
        .. code-block:: python

            llm = ChatCerebras(
                model="gpt-oss-120b",
                reasoning_effort="high"  # "low", "medium", or "high"
            )
            response = llm.invoke("What is the cube root of 50.653?")

            # Reasoning is exposed as structured content blocks
            for block in response.content:
                if isinstance(block, dict):
                    if block["type"] == "reasoning_content":
                        reasoning_text = block["reasoning_content"]["text"]
                        print(f"Reasoning: {reasoning_text}")
                    elif block["type"] == "text":
                        print(f"Answer: {block['text']}")

    Reasoning with zai-glm-4.6:
        .. code-block:: python

            llm = ChatCerebras(
                model="zai-glm-4.6",
                disable_reasoning=False  # Enable reasoning
            )
            response = llm.invoke("Explain quantum computing")

            # Same access pattern for reasoning content
            for block in response.content:
                if isinstance(block, dict):
                    if block["type"] == "reasoning_content":
                        print(f"Reasoning: {block['reasoning_content']['text']}")
                    elif block["type"] == "text":
                        print(f"Answer: {block['text']}")

    Reasoning with streaming:
        .. code-block:: python

            llm = ChatCerebras(
                model="gpt-oss-120b",
                reasoning_effort="medium"
            )

            full_reasoning = ""
            full_text = ""

            for chunk in llm.stream("What is 2+2?"):
                # Reasoning tokens are in additional_kwargs during streaming
                if "reasoning" in chunk.additional_kwargs:
                    full_reasoning += chunk.additional_kwargs["reasoning"]
                if isinstance(chunk.content, str):
                    full_text += chunk.content

            print(f"Reasoning: {full_reasoning}")
            print(f"Answer: {full_text}")

    """  # noqa: E501

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example, `{"cerebras_api_key": "CEREBRAS_API_KEY"}`
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

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling the Cerebras API."""
        params = super()._default_params
        # Add Cerebras-specific reasoning parameters if set
        # Note: reasoning_effort is already handled by BaseChatOpenAI
        # disable_reasoning is a nonstandard parameter for zai-glm-4.6
        # and must be passed via extra_body
        if self.disable_reasoning is not None:
            extra_body = params.get("extra_body") or {}
            extra_body["disable_reasoning"] = self.disable_reasoning
            params["extra_body"] = extra_body
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

    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(
        default=None,
        description=(
            "Level of reasoning effort for the gpt-oss-120b model. "
            "Options: 'low' (minimal reasoning, faster), "
            "'medium' (moderate reasoning), "
            "or 'high' (extensive reasoning, more thorough analysis)."
        ),
    )
    """Reasoning effort level for gpt-oss-120b model."""

    disable_reasoning: Optional[bool] = Field(
        default=None,
        description=(
            "Whether to disable reasoning for the zai-glm-4.6 model. "
            "Set to True to disable reasoning, False (default) to enable."
        ),
    )
    """Disable reasoning for zai-glm-4.6 model."""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        kwargs["stream"] = True
        payload = self._get_request_payload(messages, stop=stop, **kwargs)
        default_chunk_class: Type[BaseMessageChunk] = AIMessageChunk
        base_generation_info = {}

        if "response_format" in payload:
            if self.include_response_headers:
                warnings.warn(
                    "Cannot currently include response headers when response_format is "
                    "specified."
                )
            payload.pop("stream")
            response_stream = self.root_client.beta.chat.completions.stream(**payload)
            context_manager = response_stream
        else:
            if self.include_response_headers:
                raw_response = self.client.with_raw_response.create(**payload)
                response = raw_response.parse()
                base_generation_info = {"headers": dict(raw_response.headers)}
            else:
                response = self.client.create(**payload)
            context_manager = response
        try:
            with context_manager as response:
                is_first_chunk = True
                for chunk in response:
                    if not isinstance(chunk, dict):
                        chunk = chunk.model_dump()
                    generation_chunk = _convert_chunk_to_generation_chunk(
                        chunk,
                        default_chunk_class,
                        base_generation_info if is_first_chunk else {},
                    )
                    if generation_chunk is None:
                        continue

                    # Cerebras specific: extract reasoning from delta
                    choices = chunk.get("choices", [])
                    if choices:
                        choice = choices[0]
                        delta = choice.get("delta", {})
                        reasoning = delta.get("reasoning")
                        if reasoning:
                            generation_chunk.message.additional_kwargs[
                                "reasoning"
                            ] = reasoning

                    default_chunk_class = generation_chunk.message.__class__
                    logprobs = (generation_chunk.generation_info or {}).get("logprobs")
                    if run_manager:
                        run_manager.on_llm_new_token(
                            generation_chunk.text,
                            chunk=generation_chunk,
                            logprobs=logprobs,
                        )
                    is_first_chunk = False
                    yield generation_chunk
        except openai.BadRequestError as e:
            _handle_openai_bad_request(e)
        if hasattr(response, "get_final_completion") and "response_format" in payload:
            final_completion = response.get_final_completion()
            generation_chunk = self._get_generation_chunk_from_completion(
                final_completion
            )
            if run_manager:
                run_manager.on_llm_new_token(
                    generation_chunk.text, chunk=generation_chunk
                )
            yield generation_chunk

    def _create_chat_result(
        self,
        response: Union[dict, Any],
        generation_info: Optional[Dict] = None,
    ) -> ChatResult:
        result = super()._create_chat_result(response, generation_info)
        response_dict = (
            response if isinstance(response, dict) else response.model_dump()
        )

        for i, res in enumerate(response_dict["choices"]):
            message = res.get("message", {})
            reasoning = message.get("reasoning")
            if reasoning:
                result.generations[i].message.additional_kwargs["reasoning"] = reasoning
        return result

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = super()._generate(messages, stop, run_manager, **kwargs)

        for generation in result.generations:
            msg = generation.message
            reasoning = msg.additional_kwargs.get("reasoning")
            # Only structure content if user explicitly requested reasoning
            # via parameters, to maintain compatibility with standard tests
            # that expect string content by default.
            should_structure = (
                self.reasoning_effort is not None or self.disable_reasoning is False
            )

            if reasoning and should_structure:
                reasoning_block = {
                    "type": "reasoning_content",
                    "reasoning_content": {"text": reasoning},
                }

                if isinstance(msg.content, str):
                    text_block = {"type": "text", "text": msg.content}
                    if msg.content:
                        msg.content = [reasoning_block, text_block]
                    else:
                        msg.content = [reasoning_block]
                elif isinstance(msg.content, list):
                    msg.content.insert(0, reasoning_block)

                msg.additional_kwargs.pop("reasoning", None)

        return result

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
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
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

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
