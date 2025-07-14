from typing import List

from pydantic import BaseModel, Field

from app.schemas.chunk import ChunkIndexed
from app.schemas.public import FacadeExtraMixin, LLMUsage
from app.schemas.response.base import SuccessResponse


class RerankChunk(ChunkIndexed):
    relevance_score: float = Field(0, title='关联度分数')


class SimilarChunk(ChunkIndexed):
    similarity_score: float = Field(0, title='相似度分数')


class RetrievalQueryResult(BaseModel):
    query_content: str = Field(..., title='查询文本')
    vdb_indices: List[str] = Field(..., title='vdb索引')
    similar_chunks: List[SimilarChunk] = Field(..., title='召回切片列表')
    rerank_chunks: List[RerankChunk] | None = Field(None, title='重排序切片列表')


class RetrievalQueryResults(FacadeExtraMixin):
    items: List[RetrievalQueryResult] = Field(..., title='召回结果列表')
    # llm_usage: LLMUsage | None = Field(None, title='模型使用量')


class RetrievalQueryData(SuccessResponse):
    data: RetrievalQueryResults = Field(..., title='向量查询结果')
