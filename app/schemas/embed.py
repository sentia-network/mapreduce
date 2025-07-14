from typing import List, Literal

from openai.types.embedding import Embedding as OAIEmbedding
from pydantic import BaseModel, Field, HttpUrl

from app.schemas.chunk import QueryContentIndexed
from app.schemas.public import (EmbModelConfMixin, FacadeAPIKeyTag,
                                FacadeAPIKeyTagMixin, FacadeExtraMixin,
                                LLMUsage)


# todo: put embed model config here
class Embedding(OAIEmbedding):
    index: int | None = Field(None, title='索引')
    object: Literal["embedding"] = Field('embedding', title='对象类型')


class EmbeddingBatch(BaseModel):
    items: List[Embedding] = Field(..., title='嵌入向量列表')


class EmbeddingResponse(Embedding):
    usage: LLMUsage | None = Field(None, title='模型使用量')


class EmbeddingBatchResponse(EmbeddingBatch):
    items: List[Embedding | OAIEmbedding] = Field(..., title='嵌入向量列表')
    usage: LLMUsage | None = Field(None, title='模型使用量')


class EmbedCreate(FacadeExtraMixin, EmbModelConfMixin, FacadeAPIKeyTagMixin):
    items: List[QueryContentIndexed] = Field(..., title='切片列表')


class EmbedCreateWithFileUrl(EmbedCreate):
    items: List[QueryContentIndexed] | None = Field(..., title='切片列表')
    file_url: HttpUrl | None = Field(None, title='切片列表json文件 url')
