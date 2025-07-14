from singleton_decorator import singleton

from app.schemas.request.query import BatchRetrievalQuery
from app.schemas.response.query import (RetrievalQueryResult,
                                        RetrievalQueryResults)
from app.schemas.vector import VectorQuery
from app.services.embed import EmbedService
from app.services.external.facade import ServiceFacade
from app.services.external.rerank import RerankService
from app.services.vdb import VDBService


@singleton
class RetrivalService:
    embed_service = EmbedService()
    rerank_service = RerankService()
    facade = ServiceFacade()
    vdb_service = VDBService()

    def __init__(self):
        pass

    async def a_retrieve(
            self, batch_query: BatchRetrievalQuery
    ) -> RetrievalQueryResults:
        results = []
        # llm_usages = []
        for query_spec in batch_query.query_specs:
            vdb_indices = []
            similar_collection = []
            rerank_collection = []
            for embed_config in query_spec.embed_configs:
                emb_resp = await self.embed_service.a_create_embedding(
                    query_spec.query_content, embed_config)
                vdb = self.vdb_service.get_vdb(embed_config)
                query = VectorQuery(vector=emb_resp.embedding, top_k=query_spec.top_k, filter=query_spec.filter)
                query_result = await vdb.a_query(query, include_metadata=True)
                # rerank_chunks = None
                chunks = await self.facade.a_fill_content(query_result.items)
                api_key_tag = self.facade.get_api_key()
                if query_spec.rerank_config:
                    rerank_chunks, rerank_usage = await self.rerank_service.rerank(
                        query_spec.query_content, chunks, query_spec.rerank_config, api_key=api_key_tag.cohere_api_key)
                    rerank_collection = rerank_chunks.extend(rerank_chunks)
                    if not api_key_tag.has_cohere_api_key() or api_key_tag.is_platform_cohere_api_key():
                        await self.facade.send_usage(rerank_usage)
                vdb_indices.append(vdb.index_name)
                similar_collection.extend(chunks)
            results.append(RetrievalQueryResult(query_content=query_spec.query_content,
                                                vdb_indices=vdb_indices,
                                                similar_chunks=similar_collection,
                                                rerank_chunks=rerank_collection if rerank_collection else None))
            # llm_usages.append(emb_resp.usage)
        # llm_usage = LLMUsage.sum(llm_usages)
        # return RetrievalQueryResults(items=results, llm_usage=llm_usage)
        return RetrievalQueryResults(items=results)

# if __name__ == '__main__':
#     # from app.services.retrieval import RetrivalService
#     from app.schemas.public import FacadeData, FacadeAPIKeyTag
#     from app.services.external.facade import ServiceFacade, FacadeExtraContext
#
#     api_key_tag = FacadeAPIKeyTag(openai_api_key='sk-8CoIqeTWcqFfjM0n52FcAb78023c4052B83b5e916c9476Bf',
#                                   openai_api_base='https://key.agthub.ai/v1')
#     q = BatchRetrievalQuery.example()
#     service = RetrivalService()
#     with FacadeExtraContext(FacadeData(extra={}, facade_api_key=api_key_tag)):
#         # out = await service.a_retrieve(q)
#         ...
