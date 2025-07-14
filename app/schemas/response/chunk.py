from typing import Dict, List

from pydantic import BaseModel, Field

from app.schemas.response.base import BaseResponse


class ChunkSplitResponse(BaseResponse):
    data: List[str] = Field(title='切片列表')


class ChunkOutline(BaseModel):
    chunk_id_to_heading_ids: Dict[str, List[str]] = Field(title='切片ID到标题的映射')
    outline: Dict = Field(title='大纲JSON')


class ChunkOutlineResponse(BaseResponse):
    data: ChunkOutline = Field(title='大纲信息')
