import io
import json
from contextvars import ContextVar
from typing import Dict, List

from loguru import logger
from singleton_decorator import singleton

from app.core.kafka import MQManager
from app.schemas.billing import UsagePayload, UsageType
from app.schemas.chunk import ChunkIndexed
from app.schemas.facade import (Chunk, ChunkCreate, DatasetJobCategory,
                                DatasetJobStatus, DatasetJobUpdate,
                                FacadeChunkModel)
from app.schemas.public import (AssetType, CohereRerankUsage, FacadeAPIKeyTag,
                                FacadeData, FacadeExtraMixin, KafkaBaseMessage,
                                LLMUsage, OAICompletionUsage,
                                OAIEmbeddingUsage, OrcUsage)
from app.schemas.response.query import SimilarChunk
from app.schemas.vector import QueryResponse as ScoredVector
from app.schemas.vector import Vector
from app.services.external.base import ServiceClient
from app.settings.config import config

ctx_var = ContextVar('context_var', default=None)


class FacadeExtraContext:
    CTX_VAR = ctx_var

    def __init__(self, facade_data: FacadeData):
        self.facade_data = facade_data.copy(deep=True)
        self.token = None

    def __enter__(self):
        self.token = self.CTX_VAR.set(self.facade_data)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.CTX_VAR.reset(self.token)
            raise exc_val
        else:
            self.CTX_VAR.reset(self.token)

    @classmethod
    def get_extra_obj(cls) -> FacadeExtraMixin | None:
        facade_data: FacadeData = cls.CTX_VAR.get()
        if facade_data:
            return FacadeExtraMixin(extra=facade_data.extra)

    @classmethod
    def get_api_key_tag(cls) -> FacadeAPIKeyTag | None:
        facade_data: FacadeData = cls.CTX_VAR.get()
        if facade_data:
            return facade_data.facade_api_key


@singleton
class ServiceFacade:
    client: ServiceClient
    OUTLINE_IDS: str = 'outline_ids'

    def __init__(self, additional_headers: dict = None):
        self.client = ServiceClient(host=config.SERVICE_FACADE_HOST, additional_headers=additional_headers)

    async def a_update_job_status(
            self,
            status: DatasetJobStatus,
            category: DatasetJobCategory,
            detail: str,
            extra: Dict = None,
            usage: LLMUsage = None
    ) -> Dict:
        """更新任务状态"""
        update = DatasetJobUpdate(status=status, category=category, detail=detail, extra=extra, usage=usage)
        endpoint = '/admin/job/update'

        resp = await self.client.async_post(endpoint, body=update.model_dump(mode='json'))
        return resp.json()

    def get_job_status(self):
        """获取任务状态"""
        pass

    def put_assets(self, file_bytes: io.BytesIO, atype: AssetType, name: str, suffix: str, extra: dict) -> dict:
        """上传素材"""
        endpoint = '/admin/knowledge/assets/create'

        body = {
            'asset_create': json.dumps({
                'atype': atype.value,
                'name': name,
                'suffix': suffix,
                'knowledge_base_id': extra['knowledge_base_id']
            }, ensure_ascii=False),
            'extra': json.dumps(extra, ensure_ascii=False)
        }

        files = {
            'file': file_bytes
        }

        response = self.client.post(endpoint, data=body, files=files, timeout=60)
        return response.json().get('data')

    def get_assets(self):
        """获取素材"""
        pass

    async def a_put_chunks(self, chunk_create: ChunkCreate, md: str) -> List[FacadeChunkModel]:
        """上传分片"""
        endpoint = '/admin/knowledge/chunks/create'

        body = {
            **chunk_create.model_dump(mode='json'),
            'md': md
        }
        resp = await self.client.async_post(endpoint=endpoint, body=body)
        return [FacadeChunkModel.model_validate(c) for c in resp.json()['data']['chunks']]

    async def a_chunks_flag_embedded(self, chunk_ids: List[str]):
        """更新切分数据"""
        resp = await self.client.async_post(
            '/admin/knowledge/chunks/flag/embedded',
            body={'ids': chunk_ids})
        return resp.json()

    async def a_fill_content(self, vectors: List[Vector | ScoredVector]) -> List[ChunkIndexed | SimilarChunk]:
        vector_map = {v.id: v for v in vectors}
        failed_ids = []
        indexed_chunks = {}
        for vector in vectors:
            try:
                chunk = Chunk.vector_to_chunk(vector)
                indexed_chunks[vector.id] = chunk
            except Exception:
                failed_ids.append(vector.id)
        if failed_ids:
            external_chunks = await self.a_get_chunks(failed_ids)
            chunks = [Chunk.vector_to_chunk(vector_map[id], chunk) for id, chunk in
                      zip(failed_ids, external_chunks)]
            indexed_chunks = indexed_chunks | {chunk.id: chunk for chunk in chunks}
        return [indexed_chunks[v.id] for v in vectors]

    async def a_get_chunks(self, ids: List[str]) -> List[ChunkIndexed]:
        """获取分片"""
        resp = await self.client.async_post(
            '/admin/knowledge/chunks/get',
            body={'chunk_ids': ids, 'page': 1, 'limit': len(ids)})
        return Chunk.resp_to_chunks(resp.json())

    async def a_update_chunks_metadata(
            self, extra: Dict, ids: List[str], metadata_lst: List[Dict]
    ) -> List[FacadeChunkModel]:
        """更新切片元数据"""
        resp = await self.client.async_post(
            '/admin/knowledge/chunks/update/metadata',
            body={'chunks': [{'id': i, 'metadata': m} for i, m in zip(ids, metadata_lst)], 'extra': extra})
        return [FacadeChunkModel.model_validate(c) for c in resp.json()['data']['chunks']]

    async def a_update_dataset(self, extra: Dict, outline: dict) -> Dict:
        """更新数据集"""
        resp = await self.client.async_post('/admin/knowledge/dataset/update',
                                            {'outline': outline, 'extra': extra})
        return resp.json()

    def get_api_key(self) -> FacadeAPIKeyTag | None:
        return FacadeExtraContext.get_api_key_tag()

    @classmethod
    async def send_usage(cls, usage: LLMUsage | OrcUsage):
        extra = FacadeExtraContext.get_extra_obj()
        if not extra:
            logger.error('FacadeExtraMixin not found in context')
            return
        data = cls.usage_to_payload(usage=usage, extra=extra)
        logger.info(f'send usage: {data.model_dump(mode="json")}')
        kafka_msg = KafkaBaseMessage(data=data.to_body(), topic=config.KAFKA_TOPIC_USAGE)
        await MQManager().produce_message(message=kafka_msg, reraise=False, raw=True)

    @staticmethod
    def usage_to_payload(usage: LLMUsage | OrcUsage, extra: FacadeExtraMixin) -> UsagePayload:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        used_count = 0
        usage_type = None
        if isinstance(usage, OrcUsage):
            used_count = usage.used_count
            usage_type = UsageType.OCR
        elif isinstance(usage.usage, OAIEmbeddingUsage):
            prompt_tokens = usage.usage.prompt_tokens
            total_tokens = usage.usage.total_tokens
            usage_type = UsageType.EMBEDDING
        elif isinstance(usage.usage, OAICompletionUsage):
            prompt_tokens = usage.usage.prompt_tokens
            completion_tokens = usage.usage.completion_tokens
            total_tokens = usage.usage.total_tokens
            usage_type = UsageType.CHAT
        elif isinstance(usage.usage, CohereRerankUsage):
            used_count = usage.usage.count
            usage_type = UsageType.RERANK

        return UsagePayload(
            extra=extra.extra,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model=str(usage.model.value),
            used_count=used_count,
            type=usage_type,
        )

# if __name__ == '__main__':
#     from services.external.facade import ServiceFacade
#
#     s = ServiceFacade()
#     # await s.get_chunks(["0442495f-1cd2-4a91-a813-0a9346e0e6ae"])
#     chunk_create = ChunkCreate(
#         extra={"tenant_id": 1,
#                "knowledge_base_id": 2,
#                "dataset_id": 3},
#         chunks=[ChunkWithAsset(
#             query_content='这是查询文本',
#             metadata={},
#             raw_content='这是一个切片',
#             sort_id=1
#         )])
#     # chunk_ids = await s.put_chunks(chunk_create)
