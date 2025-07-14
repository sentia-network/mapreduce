from typing import TYPE_CHECKING, Dict, List
from uuid import uuid4

from pydantic import UUID4, BaseModel, Field, model_validator

if TYPE_CHECKING:
    from app.schemas.facade import FacadeChunkModel


class RawContent(BaseModel):
    raw_content: str = Field(..., title='原始文本内容')


class QueryContent(BaseModel):
    query_content: str = Field(..., title='查询文本')


class ChunkContent(RawContent, QueryContent):
    pass


class RawContentIndexed(RawContent):
    id: str = Field(..., title='切片ID')


class QueryContentAnnotated(QueryContent):
    metadata: Dict = Field(default_factory=dict, title='meta数据')


class ChunkContentAnnotated(RawContent, QueryContentAnnotated):
    metadata: Dict = Field(default_factory=dict, title='meta数据')


class QueryContentIndexed(QueryContentAnnotated):
    id: str = Field(default_factory=lambda: str(uuid4()), title='切片ID')


class QueryContentIndexedOptional(QueryContentIndexed):
    query_content: str | None = Field(None, title='查询文本')
    metadata: Dict | None = Field(None, title='meta数据')

    @model_validator(mode='after')
    def check(self):
        if self.query_content is None and self.metadata is None:
            raise ValueError('查询文本和meta数据不能同时为空')
        return self


class ChunkIndexed(RawContent, QueryContentIndexed):
    pass


class ChunkLinked(RawContent, QueryContentIndexed):
    next_id: UUID4 | str = Field('', description='下一个切片索引ID')
    sort_id: int = Field(description='排序ID')


class ChunkWithAssetID(ChunkLinked):
    asset_ids: List[int] = Field(default_factory=list, description='素材列表')


class ChunkSet(BaseModel):
    chunks: List[ChunkWithAssetID] = Field(default_factory=list, description='切分数据内容')

    @property
    def ids(self) -> List[str]:
        return [str(chunk.id) for chunk in self.chunks]

    @property
    def contents(self) -> List[ChunkContent]:
        return [ChunkContent(raw_content=chunk.raw_content, query_content=chunk.query_content) for chunk in self.chunks]

    def to_content_list(self) -> List[QueryContentIndexed]:
        return [QueryContentIndexed(query_content=chunk.query_content, metadata=chunk.metadata, id=chunk.id)
                for chunk in self.chunks]

    def align_with_facade_chunks(self, facade_chunks: List['FacadeChunkModel']):
        assert len(self.chunks) == len(facade_chunks), '切片数量不一致'
        for chunk, facade_chunk in zip(self.chunks, facade_chunks):
            chunk.id = facade_chunk.id
            chunk.metadata = facade_chunk.chunk_metadata


class ChunkConfig(BaseModel):
    chunk_size: int = Field(500, description='切片大小', ge=1)
    chunk_overlap: int = Field(0, description='切片重叠大小', ge=0)
    splitter: str | None = Field(None, description='切片分割符号')
    strip_splitter: bool = Field(False, description='是否去除分割符号')


class ChunkSplitCreate(BaseModel):
    content: str = Field(..., description='原始文本内容')
    chunk_config: ChunkConfig = Field(..., description='切片配置')


class ChunkOutlineCreate(BaseModel):
    file_name: str
    items: List[RawContentIndexed] = Field(..., description='切片内容列表')
