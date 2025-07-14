from typing import List

from pydantic import Field, HttpUrl, model_validator

from app.schemas.chunk import QueryContentIndexed, QueryContentIndexedOptional
from app.schemas.embed import EmbedCreate
from app.schemas.file import DatasetCreate
from app.schemas.public import (ChunkUpdateMixin, EmbModelConf,
                                EmbModelConfMixin, FacadeExtraMixin)
from app.schemas.request.base import Example, ExampleSet


class JobDatasetBody(DatasetCreate):

    @classmethod
    def examples(cls):
        extra = {'user_id': 1, 'tenant_id': 20,
                 'dataset_id': '7224357675757490177',
                 'job_id': '7224357820620361728',
                 'knowledge_base_id': '7224248015815454749'}
        embed_config = EmbModelConf.default()

        return ExampleSet(examples=[
            Example(
                summary='.docx文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/5fa4dedc-da95-4a41-976f-dde9c405cfa3.dox',
                         file_suffix='.docx',
                         file_name='测试文件.docx',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.doc文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/2c2d105e-05f8-49e4-9711-e21bccd6c485.doc',
                         file_suffix='.doc',
                         file_name='测试文件.doc',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.xlsx文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/a5b86fb3-e84a-4fb2-8b3c-93eb2d8d3293.xlsx',
                         file_suffix='.xlsx',
                         file_name='测试文件.xlsx',
                         is_qa=True,
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.csv文件（GB2312编码）',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/41e87503-b519-4a0c-82fe-8c9887b2700d.csv',
                         file_suffix='.csv',
                         file_name='测试文件.csv',
                         is_qa=True,
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.csv文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/d7eda69c-e937-46bf-bf60-098d24a2dae1.csv',
                         file_suffix='.csv',
                         file_name='测试文件.csv',
                         is_qa=True,
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.xlsx文件（文档类型解析）',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/83f45920-f3cd-48f2-b940-b6e0c71a42bf.xlsx',
                         file_suffix='.xlsx',
                         file_name='测试文件.xlsx',
                         is_qa=False,
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.pdf文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/e0ebabed-d1cf-41cc-87dc-626172b23225.pdf',
                         file_suffix='.pdf',
                         file_name='测试文件.pdf',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.pdf(ppt形式)',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/ba0ba647-3168-49be-bbc8-49a9bcfe7995.pdf',
                         file_suffix='.pdf',
                         file_name='测试文件.pdf',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.pptx文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/00776c1b-01ec-41c4-add1-685e06c1cbb2.pptx',
                         file_suffix='.pptx',
                         file_name='测试文件.pptx',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.txt文件（GB2312编码）',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/82d1a27c-7413-4014-a88b-aa2831592785.txt',
                         file_suffix='.txt',
                         file_name='测试文件.txt',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.txt文件（utf-8编码）',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/e17155a5-4301-4f4a-9a4c-31ec503686fb.txt',
                         file_suffix='.txt',
                         file_name='测试文件.txt',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.txt文件（GB18030编码）',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/6f7c548f-f446-4837-a786-f3ec3b84f041.txt',
                         file_suffix='.txt',
                         file_name='测试文件.txt',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.txt文件（大文本）',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/990c3032-e247-4e06-a5aa-446a5d6cb14f.txt',
                         file_suffix='.txt',
                         file_name='测试文件.txt',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
              summary='.xlsx文件（空白）',
              data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                '/176f5a84-b59c-45e6-a8ee-4a8fcbd23207.xlsx',
                       file_suffix='.xlsx',
                       file_name='测试文件.xlsx',
                       embed_config=embed_config,
                       extra=extra)
            ),
            Example(
                summary='.xlsx文件（内容自动识别类型错误）',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/k1n7p6aifhj8fjmjks3cmm90zvm9k96h.xlsx',
                         file_suffix='.xlsx',
                         file_name='测试文件.xlsx',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.epub文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/26ec4113-61fc-4e78-b62c-c9e8953e6f44.epub',
                         file_suffix='.epub',
                         file_name='测试文件.epub',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.png文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/bcf74bdb-41c8-4b8f-a7aa-6874d5939c5c.png',
                         file_suffix='.png',
                         file_name='测试文件.png',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.zip文件',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/b434ab4a-c353-4282-8d5e-926aafab3740.zip',
                         file_suffix='.zip',
                         file_name='markdown.zip测试',
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='.zip文件(缺.md 文件)',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/6608dfbf-48b2-4082-bc6c-b5e21ec8aa86.zip',
                         file_suffix='.zip',
                         file_name='markdown.zip测试',
                         embed_config=embed_config,
                         extra=extra)
            )
            # Example(
            #     summary='.gif文件',
            #     data=cls(file_url='/f45a21ad-e98c-4aab-8f0c-b8926e674092.gif',
            #              file_suffix='.gif',
            #              embed_config=embed_config,
            #              extra=extra)
            # ),
            # Example(
            #     summary='.jpg文件',
            #     data=cls(file_url='/53b04727-d1fc-445b-917a-48b3ced198c9.jpg',
            #              file_suffix='.jpg',
            #              embed_config=embed_config,
            #              extra=extra)
            # ),
            # Example(
            #     summary='.jpeg文件',
            #     data=cls(file_url='/84493661-25da-479f-bc82-5f469cb19534.jpeg',
            #              file_suffix='.jpeg',
            #              embed_config=embed_config,
            #              extra=extra)
            # ),
            # Example(
            #     summary='.webp文件',
            #     data=cls(file_url='/1ff48de6-3b22-4f26-b0c0-d9b028c4dbfe.webp',
            #              file_suffix='.webp',
            #              embed_config=embed_config,
            #              extra=extra)
            # ),
            # Example(
            #     summary='.bmp文件',
            #     data=cls(file_url='/9ba48cda-7129-480d-86dd-070e34828750.bmp',
            #              file_suffix='.bmp',
            #              embed_config=embed_config,
            #              extra=extra)
            # ),
            # Example(
            #     summary='.tiff文件',
            #     data=cls(file_url='/1d76ef0d-5507-4007-851f-bbe4c3a265af.tiff',
            #              file_suffix='.tiff',
            #              embed_config=embed_config,
            #              extra=extra)
            # ),
            # Example(
            #     summary='.wav文件',
            #     data=cls(file_url='/e566531b-6e2e-449c-b88a-b39b783aafb8.wav',
            #              file_suffix='.wav',
            #              embed_config=embed_config,
            #              extra=extra)
            # ),
            # Example(
            #     summary='.sql文件',
            #     data=cls(file_url='/89c003d7-3788-47ea-a820-cb58bf935636.sql',
            #              file_suffix='.sql',
            #              embed_config=embed_config,
            #              extra=extra)
            # )
        ]).to_openapi_examples()


class JobEmbedChunksBody(EmbedCreate):
    items: List[QueryContentIndexed] | None = Field(None, title='切片列表，与 file_url 二选一')
    file_url: HttpUrl | None = Field(None, title='切片列表json文件 url，与 items 二选一')

    @classmethod
    def examples(cls):
        embed_config = EmbModelConf.default()
        extra = FacadeExtraMixin.extra_examples()
        return ExampleSet(examples=[
            Example(
                summary='列表形式',
                data=cls(items=[QueryContentIndexed(query_content='这是一个问题')],
                         embed_config=embed_config,
                         extra=extra)
            ),
            Example(
                summary='json文件形式',
                data=cls(file_url='https://agent-circle-pub-test-1257687450.cos.ap-beijing.myqcloud.com'
                                  '/d86756eb-082c-431f-a39d-b4b4684de3ae.json',
                         embed_config=embed_config,
                         extra=extra)
            ),
        ]).to_openapi_examples()

    @model_validator(mode='after')
    def check(self):
        if self.items is None and self.file_url is None:
            raise ValueError('切片列表和文件 url 不能同时为空')

        return self


class JobUpdateChunksBody(ChunkUpdateMixin, FacadeExtraMixin, EmbModelConfMixin):

    @classmethod
    def examples(cls):
        return cls(
            items=[QueryContentIndexedOptional(query_content='这是一个问题', metadata={'new_field': 'some value'})],
            embed_config=EmbModelConf.default(),
            extra=FacadeExtraMixin.extra_examples())
