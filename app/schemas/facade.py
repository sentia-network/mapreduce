from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.chunk import (ChunkContent, ChunkIndexed, ChunkSet,
                               ChunkWithAssetID)
from app.schemas.public import FacadeExtraMixin, LLMUsage
from app.schemas.response.query import SimilarChunk
from app.schemas.vector import QueryResponse as ScoredVector
from app.schemas.vector import Vector


class Chunk(BaseModel):
    id: str
    raw_content: str
    chunk_metadata: Dict
    enabled: bool
    query_content: str
    embedded: bool
    sort_id: int
    next_id: str

    @classmethod
    def resp_to_chunk_set(cls, resp: Dict) -> ChunkSet:
        chunks = [cls(**chunk) for chunk in resp['data']['chunks']]
        chunks = [ChunkWithAssetID(id=chunk.id,
                                   raw_content=chunk.raw_content,
                                   query_content=chunk.query_content,
                                   metadata=chunk.chunk_metadata,
                                   next_id=chunk.next_id,
                                   sort_id=chunk.sort_id)
                  for chunk in chunks]
        return ChunkSet(chunks=chunks)

    @classmethod
    def resp_to_chunks(cls, resp: Dict) -> List[ChunkIndexed]:
        chunks = [cls(**chunk) for chunk in resp['data']['chunks']]
        return [ChunkIndexed(id=chunk.id,
                             raw_content=chunk.raw_content,
                             query_content=chunk.query_content,
                             metadata=chunk.chunk_metadata)
                for chunk in chunks]

    @staticmethod
    def vector_to_chunk(
            vector: Vector | ScoredVector,
            content: Optional[ChunkContent | ChunkIndexed] = None
    ) -> ChunkIndexed | SimilarChunk:
        if isinstance(vector, ScoredVector):
            return SimilarChunk(
                id=vector.id,
                raw_content=content and content.raw_content or vector.metadata['raw_content'],
                query_content=content and content.query_content or vector.metadata['query_content'],
                similarity_score=vector.score)
        return ChunkIndexed(id=vector.id, raw_content=vector.metadata['raw_content'],
                            query_content=vector.metadata['query_content'],
                            metadata=vector.metadata)


class ChunkCreate(FacadeExtraMixin):
    chunks: List[ChunkContent]

    @classmethod
    def parse_from(cls, chunk_set: ChunkSet, extra: dict):
        chunks = [ChunkContent(raw_content=chunk.raw_content, query_content=chunk.query_content)
                  for chunk in chunk_set.chunks]
        return cls(chunks=chunks, extra=extra)


class DatasetJobStatus(str, Enum):
    SUBMIT = 'submit'  # 提交
    RECEIVED = 'received'  # 接收
    START = 'start'  # 开始
    PROCESSING = 'processing'  # 进行
    FINISHED = 'finished'  # 完成
    EXCEPT = 'except'  # 异常
    CANCEL = 'cancel'  # 取消

    @classmethod
    def comment(cls):
        return f'任务状态：{"|".join(cls.__members__.values())}'


class DatasetJobCategory(str, Enum):
    DATASET_CREATE = 'dataset_create'
    EMBED_CREATE = 'embed_create'
    EMBED_UPDATE = 'embed_update'


class DatasetJobUpdate(FacadeExtraMixin):
    usage: LLMUsage | None = Field(None, description='模型使用量')
    category: DatasetJobCategory = Field(title='任务种类')
    status: DatasetJobStatus = Field(..., title='任务状态', description=DatasetJobStatus.comment())
    detail: str = Field('', title='任务状态描述')

    @field_validator('extra')
    @classmethod
    def check_extra(cls, v):
        if 'job_id' not in v:
            raise ValueError('job_id 不在 extra 中！')
        if 'user_id' not in v:
            raise ValueError('user_id 不在 extra 中！')
        return v

    @property
    def id(self):
        return self.extra.get('job_id')


class FacadeChunkModel(BaseModel):
    """Facade返回的切片结构"""
    id: str = Field(..., title='切片ID')
    enabled: bool = Field(True, title='是否启用切片')
    raw_content: str = Field(..., title='原始文本内容，QA类型中的回答')
    query_content: str | None = Field(None, title='查询文本, QA类型中的问题')
    tenant_id: int = Field(..., title='租户ID')
    knowledge_base_id: int = Field(..., title='知识库ID')
    dataset_id: int = Field(..., title='数据集ID')
    next_id: str = Field('', title='下一个切片ID')
    asset_ids: List[int] = Field(default_factory=list, title='素材ID列表')
    job_id: int = Field(0, title='任务ID')
    chunk_metadata: dict = Field(default_factory=dict, title='切片元数据')
    sort_id: int = Field(..., title='序号')
    embedded: bool = Field(..., title='是否完成向量化')
