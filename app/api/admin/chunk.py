from fastapi import APIRouter, Depends

from app.schemas.chunk import ChunkOutlineCreate, ChunkSplitCreate
from app.schemas.request.base import CommonHeaders
from app.schemas.response.chunk import (ChunkOutline, ChunkOutlineResponse,
                                        ChunkSplitResponse)
from app.services.chunk import ChunkService

router = APIRouter(dependencies=[Depends(CommonHeaders.from_request)])


@router.post('/chunk/split', summary='切分文本')
async def chunk_split(body: ChunkSplitCreate) -> ChunkSplitResponse:
    data = ChunkService().split_text(body=body)
    return ChunkSplitResponse(data=data)


@router.post('/chunk/outline', summary='提取大纲')
async def chunk_outline(body: ChunkOutlineCreate) -> ChunkOutlineResponse:
    chunk_id_to_heading_ids, outline = ChunkService().extract_outline(body=body)
    return ChunkOutlineResponse(
        data=ChunkOutline(chunk_id_to_heading_ids=chunk_id_to_heading_ids, outline=outline))
