import traceback
from functools import partial
from typing import List, Optional, Tuple

from loguru import logger
from openai import NOT_GIVEN
from singleton_decorator import singleton

from app.schemas.chunk import QueryContentIndexed, QueryContentIndexedOptional
from app.schemas.embed import EmbeddingBatchResponse, EmbeddingResponse
from app.schemas.public import (ChunkUpdateMode, EmbModelConf, LLMUsage,
                                OpenAIEmbModels)
from app.schemas.vector import (Vector, VectorBatch, VectorBatchOptional,
                                VectorOptional)
from app.services.external.facade import ServiceFacade
from app.services.external.llm import LLMService
from app.services.vdb import VDBService
from app.settings.config import config
from app.utils import make_chunks


@singleton
class EmbedService:
    llm_service = LLMService()
    facade = ServiceFacade()

    async def a_create_embeddings(
            self, inputs: str | List[str], config: EmbModelConf
    ) -> EmbeddingBatchResponse:
        if isinstance(inputs, List) and len(inputs) == 0:
            return EmbeddingBatchResponse(
                items=[], usage=LLMUsage.null_embed_usage(model=config.model, supplier=config.supplier))

        dim = int(config.dim) if config.model != OpenAIEmbModels.ADA002 else NOT_GIVEN
        api_key_tag = self.facade.get_api_key()
        resp = await self.llm_service.a_oai_embed(
            input=inputs, model=config.model.value, dimensions=dim,
            api_key=api_key_tag.openai_api_key, base_url=api_key_tag.openai_api_base)
        usage = LLMUsage.from_llm_resp(model=config.model, supplier=config.supplier, usage=resp.usage)
        if not api_key_tag.has_openai_api_key() or api_key_tag.is_platform_openai_api_key():
            await self.facade.send_usage(usage)
        return EmbeddingBatchResponse(items=resp.data, usage=usage)

    async def a_create_embedding(self, input: str, config: EmbModelConf) -> EmbeddingResponse:
        resp = await self.a_create_embeddings([input], config)
        return EmbeddingResponse(embedding=resp.items[0].embedding,
                                 index=resp.items[0].index,
                                 object=resp.items[0].object,
                                 usage=resp.usage)

    async def a_embed_by_chunk(
            self, content_list: List[QueryContentIndexed | QueryContentIndexedOptional],
            embed_config: EmbModelConf, update_mode: Optional[ChunkUpdateMode] = None
    ) -> Tuple[LLMUsage, Optional[Exception]]:
        if isinstance(content_list[0], QueryContentIndexedOptional):
            embed_method = partial(self.a_embed_update, update_mode=update_mode)
        else:
            embed_method = self.a_embed

        llm_usages = [LLMUsage.null_embed_usage(model=embed_config.model, supplier=embed_config.supplier)]
        try:
            for content_batch in make_chunks(content_list, config.EMBED_BATCH_SIZE):
                llm_usage = await embed_method(content_batch, embed_config)
                llm_usages.append(llm_usage)

        except Exception as e:
            logger.error(f'Error in a_embed_by_chunk: {e}')
            logger.error(f'Traceback: {traceback.format_exc()}')
            return LLMUsage.sum(llm_usages), e

        return LLMUsage.sum(llm_usages), None

    async def a_embed(self, content_list: List[QueryContentIndexed], embed_config: EmbModelConf) -> LLMUsage:
        vdb = VDBService().get_vdb(embed_config)
        embed_batch = await self.a_create_embeddings([c.query_content for c in content_list], embed_config)
        await vdb.a_upsert(VectorBatch(vectors=[
            Vector(vector=item.embedding, id=content.id, metadata=content.metadata)
            for content, item in zip(content_list, embed_batch.items)]))
        return embed_batch.usage

    async def a_embed_update(
            self, content_list: List[QueryContentIndexedOptional],
            embed_config: EmbModelConf, update_mode: ChunkUpdateMode
    ) -> LLMUsage:
        vdb = VDBService().get_vdb(embed_config)
        query_contents = [c for c in content_list if c.query_content]
        embed_batch = await self.a_create_embeddings([c.query_content for c in query_contents], embed_config)
        vector_id_map = {c.id: i.embedding for c, i in zip(query_contents, embed_batch.items)}

        if update_mode == ChunkUpdateMode.REPLACE:
            empty_contents = [c for c in content_list if c.query_content is None]
            if empty_contents:
                ret_vector_batch = await vdb.a_fetch([c.id for c in empty_contents])
                old_vector_id_map = {c.id: i.vector for c, i in zip(empty_contents, ret_vector_batch.vectors)}
                vector_id_map = old_vector_id_map | vector_id_map
            vector_batch = VectorBatch(vectors=[
                Vector(id=content.id, vector=vector_id_map[content.id], metadata=content.metadata)
                for content in content_list])
            await vdb.a_upsert(vector_batch)
            return embed_batch.usage

        elif update_mode == ChunkUpdateMode.MERGE:
            vector_batch = VectorBatchOptional(vectors=[
                VectorOptional(id=content.id, metadata=content.metadata, vector=vector_id_map.get(content.id))
                for content in content_list])
            await vdb.a_update(vector_batch)
            return embed_batch.usage

        else:
            raise ValueError(f'Unsupported update mode: {update_mode}')
