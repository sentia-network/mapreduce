from typing import List, Union

from pydantic import BaseModel, Field, field_validator

from app.core.chunker import MarkdownSkeleton
from app.schemas.chunk import QueryContent
from app.schemas.public import FacadeExtraMixin


class Section(BaseModel):
    name: str = Field(..., title='大纲节点名称')
    id: str = Field(..., title='大纲节点ID', coerce_numbers_to_str=True)
    sub_secs: List[Union['Section', MarkdownSkeleton]] = Field(default_factory=list, title='子节点')

    def combine(self) -> MarkdownSkeleton:
        return MarkdownSkeleton(
            heading=self.name, id=self.id, sub_sections=[
                s.combine() if isinstance(s, Section) else s for s in self.sub_secs
            ])


class OutlineQuery(QueryContent, FacadeExtraMixin):
    outline_root_name: str = Field(..., title='大纲根节点名称')
    secs: List[Section] = Field(..., title='大纲节点列表')

    @classmethod
    def example(cls):
        return cls(
            query_content='这是查询文本',
            outline_root_name='某助理的知识库',
            secs=[Section(name='A知识库',
                          id='21365468',
                          sub_secs=[MarkdownSkeleton(id='4abc4e29', heading='# A')]),
                  Section(name='B知识库',
                          id='34857598',
                          sub_secs=[
                              Section(
                                  name='C数据集',
                                  id='87539349',
                                  sub_secs=[MarkdownSkeleton(id='4aec4e29', heading='# B')])
                          ])
                  ])

    def combine(self) -> MarkdownSkeleton:
        return MarkdownSkeleton(heading=self.outline_root_name, id='00000000',
                                sub_sections=[s.combine() for s in self.secs])
