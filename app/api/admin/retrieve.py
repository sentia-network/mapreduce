from fastapi import APIRouter, Body, Depends

from app.schemas.public import FacadeAPIKeyTag, FacadeData
from app.schemas.request.base import CommonHeaders
from app.schemas.request.query import BatchRetrievalQuery
from app.schemas.response.query import RetrievalQueryData
from app.services.external.facade import FacadeExtraContext
from app.services.retrieval import RetrivalService

router = APIRouter(dependencies=[Depends(CommonHeaders.from_request)])


@router.post("/vector/retrieve", summary='向量数据召回')
async def vector_retrieve(
        batch_query: BatchRetrievalQuery = Body(example=BatchRetrievalQuery.example()),
        api_key_tag: FacadeAPIKeyTag = Depends(FacadeAPIKeyTag.get_api_key_header),
) -> RetrievalQueryData:
    with FacadeExtraContext(FacadeData(extra=batch_query.extra, facade_api_key=api_key_tag)):
        results = await RetrivalService().a_retrieve(batch_query)
        results.extra = batch_query.extra
    return RetrievalQueryData(data=results)
