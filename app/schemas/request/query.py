from typing import Dict, List

from pydantic import Field

from app.schemas.chunk import QueryContent
from app.schemas.public import (CohereRerankModels, EmbModelConf,
                                FacadeExtraMixin, MultiEmbModelConfMixin,
                                RerankConfig, RerankModelSuppliers)


class RetrievalQuery(QueryContent, MultiEmbModelConfMixin):
    top_k: int = Field(10, title='top k')
    filter: Dict = Field(default_factory=dict,
                         title='过滤参数，语法参考：https://docs.pinecone.io/docs/metadata-filtering')
    rerank_config: RerankConfig | None = Field(None, title='rerank配置')

    @classmethod
    def example(cls):
        return cls(
            query_content='这是查询文本',
            filter={'_tenant_id': 1},
            rerank_config=RerankConfig(
                top_k=5, model=CohereRerankModels.RERANK_MULTILINGUAL_V3_0,
                supplier=RerankModelSuppliers.COHERE),
            embed_configs=[EmbModelConf.default()]
        )


class BatchRetrievalQuery(FacadeExtraMixin):
    query_specs: List[RetrievalQuery] = Field(..., title='查询文本列表')

    @classmethod
    def example(cls):
        return cls(query_specs=[RetrievalQuery.example()])
