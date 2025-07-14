from fastapi import APIRouter, Body, Depends

from app.core.llm_ops import a_propose_outline_sections
from app.schemas.public import FacadeAPIKeyTag, FacadeData
from app.schemas.request.base import CommonHeaders
from app.schemas.request.llm_ops import OutlineQuery
from app.schemas.response.llm_ops import OutlineRefine, OutlineRefineRespond
from app.services.external.facade import FacadeExtraContext

router = APIRouter(dependencies=[Depends(CommonHeaders.from_request)])


@router.post("/llm_ops/outline_refine", summary='提取相关大纲目录id')
async def outline_refine(
        query: OutlineQuery = Body(example=OutlineQuery.example()),
        api_key_tag: FacadeAPIKeyTag = Depends(FacadeAPIKeyTag.get_api_key_header),
) -> OutlineRefineRespond:
    with FacadeExtraContext(FacadeData(extra=query.extra, facade_api_key=api_key_tag)):
        skeleton = query.combine()
        light_headings, full_headings = skeleton.light_headings, skeleton.full_headings
        section_ids = await a_propose_outline_sections(query.query_content, full_headings)
    return OutlineRefineRespond(
        data=OutlineRefine(
            section_ids=section_ids.section_ids,
            full_headings=full_headings,
            light_headings=light_headings))
