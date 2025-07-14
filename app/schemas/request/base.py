import uuid
from typing import List

from fastapi import Header
from pydantic import UUID4, BaseModel, Field


class CommonHeaders(BaseModel):
    x_request_id: str | None = Field(None, description='日志追踪ID')

    @staticmethod
    def from_request(x_request_id: str = Header(None, description='日志追踪ID', example=uuid.uuid4().hex)
                     ) -> "CommonHeaders":
        return CommonHeaders(x_request_id=x_request_id)


class Example(BaseModel):
    summary: str = Field(description='摘要')
    description: str = Field('', description='描述')
    data: BaseModel = Field(description='值')

    def to_openapi(self):
        return {'summary': self.summary, 'description': self.description, 'value': self.data.model_dump(mode='json')}


class ExampleSet(BaseModel):
    examples: List[Example] = Field(description='OpenAPI示例')

    def to_openapi_examples(self):
        return {example.summary: example.to_openapi() for example in self.examples}


def to_openapi_examples(example: BaseModel, summary: str, description: str) -> dict:
    return ExampleSet(examples=[Example(summary=summary, description=description, data=example)]).to_openapi_examples()
