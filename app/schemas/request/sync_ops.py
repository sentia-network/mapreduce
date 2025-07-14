from typing import Dict, List

from pydantic import Field, model_validator

from app.schemas.chunk import QueryContentIndexed, QueryContentIndexedOptional
from app.schemas.public import (ChunkUpdateMixin, EmbModelConf,
                                EmbModelConfMixin, FacadeExtraMixin)
from app.schemas.vector import Vector, VectorBatch


class VectorCreate(FacadeExtraMixin, EmbModelConfMixin):
    items: List[QueryContentIndexed] = Field(description='切分数据内容')

    @classmethod
    def example(cls):
        return cls(
            items=[
                QueryContentIndexed(query_content='这是查询文本',
                                    metadata={'_tenant_id': 1},
                                    id='123e4567-e89b-12d3-a456-426614174000')],
            embed_config=EmbModelConf.default()
        )


class VectorMigrate(VectorBatch, EmbModelConfMixin):
    vectors: List[Vector] = Field(description='向量数据内容', min_items=1)

    @classmethod
    def example(cls):
        return cls(vectors=[Vector(id='123e4567-e89b-12d3-a456-426614174000',
                                   vector=[0.1] * 1536,
                                   metadata={'_tenant_id': 1})],
                   embed_config=EmbModelConf.default())


class VectorDelete(FacadeExtraMixin):
    ids: List[str] | None = Field(None, title='切片ID列表')
    filter: Dict | None = Field(None, title='过滤条件')
    embed_config: EmbModelConf | None = Field(None, title='向量化模型配置')

    @model_validator(mode='after')
    def check(self):
        if (self.ids is None and self.filter is None) or (isinstance(self.ids, List) and isinstance(self.filter, Dict)):
            raise ValueError('ids和filter不能同时为空或同时存在')
        if self.ids is not None:
            assert len(self.ids) > 0, 'ids不能为空'
        if self.filter is not None:
            assert 'tenant_id' in self.filter, 'filter中必须包含tenant_id'
            assert 'knowledge_base_id' in self.filter, 'filter中必须包含knowledge_base_id'
        return self

    @classmethod
    def example(cls):
        return cls(ids=['123e4567-e89b-12d3-a456-426614174000',
                        '123e4567-e89b-12d3-a456-426614174001'],
                   embed_config=EmbModelConf.default())

    @classmethod
    def example1(cls):
        return cls(ids=None,
                   filter={'tenant_id': 1, 'knowledge_base_id': '123e4567-e89b-12d3-a456-426614174000'},
                   embed_config=EmbModelConf.default())


class VectorFetch(VectorDelete):
    pass


class VectorUpdate(ChunkUpdateMixin, FacadeExtraMixin, EmbModelConfMixin):

    @classmethod
    def example(cls):
        return cls(
            items=[QueryContentIndexedOptional(
                query_content='这是新的查询文本',
                metadata={'_tenant_id': 1},
                id='123e4567-e89b-12d3-a456-426614174000')],
            embed_config=EmbModelConf.default()
        )
