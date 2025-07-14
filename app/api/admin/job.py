from fastapi import APIRouter, Body, Depends

from app.schemas.public import FacadeAPIKeyTag
from app.schemas.request.base import CommonHeaders
from app.schemas.request.job import JobDatasetBody, JobEmbedChunksBody
from app.schemas.response.base import SuccessResponse
from app.services.job import JobService

router = APIRouter(dependencies=[Depends(CommonHeaders.from_request)])


@router.post("/job/dataset", summary='数据集解析、提取、切分、向量化的完整流程处理任务')
async def job_dateset(
        body: JobDatasetBody = Body(openapi_examples=JobDatasetBody.examples()),
        api_key_tag: FacadeAPIKeyTag = Depends(FacadeAPIKeyTag.get_api_key_header),
) -> SuccessResponse:
    body.facade_api_key = api_key_tag
    await JobService().add_dataset_job(body)
    return SuccessResponse()


@router.post("/job/embed", summary='向量化任务')
async def job_embed(
        body: JobEmbedChunksBody = Body(openapi_examples=JobEmbedChunksBody.examples()),
        api_key_tag: FacadeAPIKeyTag = Depends(FacadeAPIKeyTag.get_api_key_header),
) -> SuccessResponse:
    body.facade_api_key = api_key_tag
    await JobService().add_embed_create_job(body)
    return SuccessResponse()
