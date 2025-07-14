from typing import List

from fastapi import APIRouter, Body, Depends, Query
from loguru import logger

from app.schemas.public import (EmbModelConf, EmbModelSuppliers,
                                FacadeAPIKeyTag, FacadeData, OpenAIEmbModels)
from app.schemas.request.base import CommonHeaders
from app.schemas.request.sync_ops import (VectorCreate, VectorDelete,
                                          VectorMigrate, VectorUpdate)
from app.schemas.response.base import SuccessResponse
from app.schemas.response.sync_ops import (UsageData, VectorCreateResponse,
                                           VectorData, VectorUpdateResponse)
from app.services.embed import EmbedService
from app.services.external.facade import FacadeExtraContext
from app.services.vdb import VDBService

router = APIRouter(dependencies=[Depends(CommonHeaders.from_request)])


@router.get("/sync_ops/vector/get", summary='向量数据查询')
async def get_vectors(
        ids: List[str] = Query(
            default=[],
            example=['123e4567-e89b-12d3-a456-426614174000',
                     '123e4567-e89b-12d3-a456-426614174001'],
            description='切片ID列表'),
        model: OpenAIEmbModels | None = Query(
            OpenAIEmbModels.ADA002,
            example=OpenAIEmbModels.ADA002,
            description='向量化模型名称，无则默认为ada002'),
        supplier: EmbModelSuppliers | None = Query(
            EmbModelSuppliers.OPENAI,
            example=EmbModelSuppliers.OPENAI,
            description='向量化模型供应商，无则默认为openai'),
        dim: int | None = Query(
            None,
            example=1536,
            description='向量维度，无则默认为模型最大维度'),
        include_vector: bool | None = Query(
            False,
            example=True,
            description='是否包含向量数据'),
) -> VectorData:
    logger.info(f'params: ids={ids}, model={model}, supplier={supplier}, dim={dim}, include_vector={include_vector}')
    embed_config = EmbModelConf(supplier=supplier, model=model, dim=dim)
    vdb = VDBService().get_vdb(embed_config)
    vector_batch = await vdb.a_fetch(ids)
    return VectorData(data=vector_batch)


@router.post("/sync_ops/vector/create", summary='向量数据新增')
async def create_vectors(
        body: VectorCreate = Body(example=VectorCreate.example()),
        api_key_tag: FacadeAPIKeyTag = Depends(FacadeAPIKeyTag.get_api_key_header),
) -> VectorCreateResponse:
    with FacadeExtraContext(FacadeData(extra=body.extra, facade_api_key=api_key_tag)):
        llm_usage = await EmbedService().a_embed(body.items, body.embed_config)
    return VectorCreateResponse(data=UsageData(usage=llm_usage, extra=body.extra))


@router.post("/sync_ops/vector/upload", summary='向量数据入库')
async def upload_vectors(
        body: VectorMigrate = Body(example=VectorMigrate.example())
) -> SuccessResponse:
    vdb = VDBService().get_vdb(body.embed_config)
    await vdb.a_upsert(body)
    return SuccessResponse()


@router.post("/sync_ops/vector/delete", summary='向量数据删除')
async def delete_vectors(
        body: VectorDelete = Body(examples=[VectorDelete.example1(), VectorDelete.example()])
) -> SuccessResponse:
    if body.embed_config:
        vdb = VDBService().get_vdb(body.embed_config)
        await vdb.a_delete(body.ids, body.filter)
    else:
        await VDBService().global_vdb_delete(body.ids, body.filter)
    return SuccessResponse()


@router.post("/sync_ops/vector/update", summary='向量数据更新')
async def update_vectors(
        body: VectorUpdate = Body(example=VectorUpdate.example()),
        api_key_tag: FacadeAPIKeyTag = Depends(FacadeAPIKeyTag.get_api_key_header),
) -> VectorUpdateResponse:
    with FacadeExtraContext(FacadeData(extra=body.extra, facade_api_key=api_key_tag)):
        llm_usage = await EmbedService().a_embed_update(body.items, body.embed_config, body.update_mode)
    return VectorUpdateResponse(data=UsageData(usage=llm_usage))
