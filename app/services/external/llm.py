from typing import Dict, Tuple

from openai import AsyncOpenAI, AsyncStream, OpenAI
from openai.types import CreateEmbeddingResponse
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from singleton_decorator import singleton

from app.settings.config import config


@singleton
class LLMService:
    openai_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    a_openai_client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

    # https://platform.openai.com/docs/api-reference/chat/create
    # a_oai_completion: Callable[..., ChatCompletion | AsyncStream[ChatCompletionChunk]] = partial(
    #     a_openai_client.chat.completions.create)
    # oai_completion: Callable[..., ChatCompletion | AsyncStream[ChatCompletionChunk]] = partial(
    #     openai_client.chat.completions.create)

    # https://platform.openai.com/docs/api-reference/embeddings/create
    # a_oai_embed: Callable[..., CreateEmbeddingResponse] = partial(a_openai_client.embeddings.create)
    # oai_embed: Callable[..., CreateEmbeddingResponse] = partial(openai_client.embeddings.create)
    def get_client(self, api_key=None, base_url=None, is_async=False, **kwargs) -> Tuple[OpenAI | AsyncOpenAI, Dict]:
        if api_key is not None:
            if is_async:
                return AsyncOpenAI(api_key=api_key, base_url=base_url), kwargs
            else:
                return OpenAI(api_key=api_key, base_url=base_url), kwargs
        if is_async:
            return self.a_openai_client, kwargs
        else:
            return self.openai_client, kwargs

    async def a_oai_completion(self, **kwargs) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        async_client, kw = self.get_client(is_async=True, **kwargs)
        return await async_client.chat.completions.create(**kw)

    def oai_completion(self, **kwargs) -> ChatCompletion | AsyncStream[ChatCompletionChunk]:
        client, kw = self.get_client(**kwargs)
        return client.chat.completions.create(**kw)

    async def a_oai_embed(self, **kwargs) -> CreateEmbeddingResponse:
        async_client, kw = self.get_client(is_async=True, **kwargs)
        return await async_client.embeddings.create(**kw)

    def oai_embed(self, **kwargs) -> CreateEmbeddingResponse:
        client, kw = self.get_client(**kwargs)
        return client.embeddings.create(**kw)
