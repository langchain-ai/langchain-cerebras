# Langchain-Cerebras

This package contains the LangChain integration with Cerebras.

At Cerebras, we've developed the world's largest and fastest AI processor, the Wafer-Scale Engine-3 (WSE-3). The Cerebras CS-3 system, powered by the WSE-3, represents a new class of AI supercomputer that sets the standard for generative AI training and inference with unparalleled performance and scalability.

With Cerebras as your inference provider, you can:
- Achieve unprecedented speed for AI inference workloads
- Build commercially with high throughput
- Effortlessly scale your AI workloads with our seamless clustering technology

Our CS-3 systems can be quickly and easily clustered to create the largest AI supercomputers in the world, making it simple to place and run the largest models. Leading corporations, research institutions, and governments are already using Cerebras solutions to develop proprietary models and train popular open-source models.

Want to experience the power of Cerebras? Check out our [website](https://cerebras.ai) for more resources and explore options for accessing our technology through the Cerebras Cloud or on-premise deployments!

For more information about Cerebras Cloud, visit [cloud.cerebras.ai](https://cloud.cerebras.ai/). Our API reference is available at [inference-docs.cerebras.ai](https://inference-docs.cerebras.ai/).

## Installation

```bash
pip install langchain-cerebras
```

## API Key
Get an API Key from [cloud.cerebras.ai](https://cloud.cerebras.ai/) and add it to your environment variables:
```
export CEREBRAS_API_KEY="your-api-key-here"
```

## Chat model

### More examples
See more examples [here](http://python.langchain.com/docs/integrations/chat/cerebras).

### Example

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_cerebras import ChatCerebras

chat = ChatCerebras(
    model="llama3.1-70b",
)

system = "You are an expert on animals who must answer questions in a manner that a 5 year old can understand."
human = "I want to learn more about this animal: {text}"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", human)
    ]
)

chain = prompt | chat
chain.invoke({"text": "Lion"})
```

Result content:
```
OH BOY! Let me tell you all about LIONS!

Lions are the kings of the jungle! They're really big and have beautiful, fluffy manes around their necks. The mane is like a big, golden crown!

Lions live in groups called prides. A pride is like a big family, and the lionesses (that's what we call the female lions) take care of the babies. The lionesses are like the mommies, and they teach the babies how to hunt and play.

Lions are very good at hunting. They work together to catch their food, like zebras and antelopes. They're super fast and can run really, really fast!

But lions are also very sleepy. They like to take long naps in the sun, and they can sleep for up to 20 hours a day! Can you imagine sleeping that much?

Lions are also very loud. They roar really loudly to talk to each other. It's like they're saying, "ROAR! I'm the king of the jungle!"

And guess what? Lions are very social. They like to play and cuddle with each other. They're like big, furry teddy bears!

So, that's lions! Aren't they just the coolest?
```

## Development + Contributing

For more information, visit [LangChain's contribution guide](https://python.langchain.com/v0.1/docs/contributing/code/).

### Install Dependencies
```
poetry install --with test,lint,codespell
```

### Build
```
poetry build
```

### Unit Test
Unit tests are completely offline and do not require an API key.
```
make test
```

### Integration Test
Integration tests require the environment variable `CEREBRAS_API_KEY` to be set to a valid API key along with a connection to the Cerebras Cloud servers.

```
make integration_test
```

### Linting and Formatting
```
make lint spell_check check_imports
```