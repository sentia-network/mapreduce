from pydantic import Field

from app.schemas.public import FacadeExtraMixin, LLMUsage
from app.schemas.response.base import SuccessResponse
from app.schemas.vector import VectorBatch


class VectorData(SuccessResponse):
    data: VectorBatch = Field(..., title='向量数据')


class UsageData(FacadeExtraMixin):
    usage: LLMUsage | None = Field(None, title='模型使用量')


class VectorCreateResponse(SuccessResponse):
    data: UsageData = Field(None, title='模型使用量')


class VectorUpdateResponse(VectorCreateResponse):
    pass
