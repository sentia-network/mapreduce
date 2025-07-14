import traceback
from typing import List, Optional, Tuple

import cohere
import httpx
from cohere import RerankResponse
from loguru import logger
from singleton_decorator import singleton

from app.middlewares.exception.common import CommonException
from app.schemas.chunk import ChunkIndexed
from app.schemas.public import (CohereRerankUsage, ContentType, LLMUsage,
                                RerankConfig)
from app.schemas.response.query import RerankChunk
from app.settings.config import config


@singleton
class RerankService:
    # cohere reference: https://docs.cohere.com/reference/rerank-1

    def __init__(self):
        cohere_api_key = config.COHERE_API_KEY
        cohere_proxy = config.COHERE_PROXY
        self.httpx_client = httpx.AsyncClient(proxies=cohere_proxy)
        self.default_client = cohere.AsyncClient(api_key=cohere_api_key, httpx_client=self.httpx_client)

    @staticmethod
    def log_rerank(indices: List[int]):
        rank_changes = [f'{j + 1} -> {i + 1}' for i, j in enumerate(indices) if i != j]
        if rank_changes:
            logger.info(f'Rank changes: {rank_changes}')
            return
        logger.info(f'No changes with rerank.')

    def get_client(self, api_key: Optional[str] = None) -> cohere.AsyncClient:
        if api_key:
            return cohere.AsyncClient(api_key, httpx_client=self.httpx_client)
        return self.default_client

    async def rerank_strings(self, query: str, docs: List[str], top_n: Optional[int] = None,
                             model: Optional[str] = 'rerank-multilingual-v3.0') -> RerankResponse:
        client = self.get_client()
        return await client.rerank(model=model, query=query, documents=docs, top_n=top_n)

    async def rerank(
            self,
            query: str,
            chunks: List[ChunkIndexed],
            config: RerankConfig,
            api_key: Optional[str] = None
    ) -> Tuple[List[RerankChunk], LLMUsage]:
        if not chunks:
            return [], LLMUsage.null_rerank_usage(model=config.model, supplier=config.supplier)

        client = self.get_client(api_key)
        docs = [c.raw_content if config.content_type == ContentType.RAW else c.query_content for c in chunks]
        try:
            resp: RerankResponse = await client.rerank(model=config.model.value,
                                                       query=query,
                                                       documents=docs,
                                                       top_n=config.top_k)
        except Exception as e:
            raise CommonException.service_error(detail=f'Rerank service error: {e}, trace: \n{traceback.format_exc()}')
        indices = [ret.index for ret in resp.results]
        self.log_rerank(indices)
        reranked_chunks = [RerankChunk(relevance_score=ret.relevance_score,
                                       id=chunks[ret.index].id,
                                       raw_content=chunks[ret.index].raw_content,
                                       query_content=chunks[ret.index].query_content)
                           for ret in resp.results]
        usage = LLMUsage(model=config.model, supplier=config.supplier, usage=CohereRerankUsage(count=1))
        return reranked_chunks, usage
