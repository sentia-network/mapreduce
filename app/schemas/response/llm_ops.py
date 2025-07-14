from pydantic import BaseModel, Field

from app.schemas.response.base import SuccessResponse


class OutlineRefine(BaseModel):
    section_ids: list[str] = []
    full_headings: str
    light_headings: str


class OutlineRefineRespond(SuccessResponse):
    data: OutlineRefine = Field(..., title='向量查询结果')
