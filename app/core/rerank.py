import json
from typing import List, Optional, Tuple

import cohere
import httpx
from loguru import logger

from app.models import RerankSettings, Segment
from app.settings.config import config


def log_rerank(indices: List[int]):
    rank_changes = [f'{j + 1} -> {i + 1}' for i, j in enumerate(indices) if i != j]
    if rank_changes:
        logger.info(f'Rank changes: {rank_changes}')
        return
    logger.info(f'No changes with rerank.')


class Rerank:
    # cohere reference: https://docs.cohere.com/reference/rerank-1
    def __init__(self, cohere_api_key: Optional[str], cohere_proxy: Optional[str]):
        cohere_api_key = cohere_api_key if cohere_api_key else config.COHERE_API_KEY
        cohere_proxy = cohere_proxy if cohere_proxy else config.COHERE_PROXY
        httpx_client = httpx.Client(proxies=cohere_proxy)
        self.client = cohere.Client(api_key=cohere_api_key, httpx_client=httpx_client)

    def rerank(self, query: str, docs: List[str], top_k: int = 4,
               settings: RerankSettings = RerankSettings()
               ) -> Tuple[List[int], List[float]]:
        response = self.client.rerank(
            model=settings.model,
            query=query,
            documents=docs,
            top_n=top_k)
        indices = [ret.index for ret in response.results]
        logger.info(f'Indices: {indices}')
        relevance_scores = [ret.relevance_score for ret in response.results]
        return indices, relevance_scores

    def rerank_segments(self, query: str, segments: List[Segment],
                        scores: List[float], top_k: int = 4,
                        settings: RerankSettings = RerankSettings()
                        ) -> Tuple[List[Segment], List[Optional[float]]]:
        if not settings:
            # return segments[:top_k], [None] * top_k
            return [], []

        if settings.trigger_threshold and scores[0] > settings.trigger_threshold:
            logger.info(f'Rerank not triggered with best score {scores[0]:4.f} given settings {settings}')
            return segments[:top_k], [None] * top_k
        else:
            logger.info(f"Rerank triggered with settings {settings}")
            try:
                indices, rerank_scores = self.rerank(
                    query,
                    [f'{segment.text}\n{json.loads(segment.metadata)[config.CHUNK_METADATA_QA_ANSWER]}'
                     if json.loads(segment.metadata).get(config.CHUNK_METADATA_QA_ANSWER, None) else segment.text
                     for segment in segments],
                    top_k=top_k,
                    settings=settings)
                ranked_segments = [segments[i] for i in indices]
                log_rerank(indices)
                return ranked_segments, rerank_scores
            except Exception as e:
                logger.error(f"Rerank failed with exception {e}")
                return [], []


# rerank_module = Rerank()
