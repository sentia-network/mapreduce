from pydantic import BaseModel, Field

from app.middlewares.request_ctx import RequestContextualize


class BaseResponse(BaseModel):
    x_request_id: str = Field(default_factory=lambda: RequestContextualize.get_request_id(), description='追踪ID')
    errcode: int = Field(0, description='错误码')
    errmsg: str = Field('', description='错误类型信息')
    detail: str = Field('', description='错误详细信息')
    data: dict | BaseModel = Field(default_factory=dict, description='返回数据')


class SuccessResponse(BaseResponse):
    data: dict = Field({'state': True}, description='成功')


class FailResponse(BaseResponse):
    data: dict = Field({'state': False}, description='失败')
