import json
from enum import Enum, IntEnum
from html import escape, unescape
from typing import List, Literal, Optional
from uuid import UUID, uuid4

from core.openai_adapter import openai_client
from fastapi import Header
from loguru import logger
from pydantic import BaseModel as PDModel
from pydantic import Field
from pydantic import Field as PydanticField
from pydantic import field_validator, model_validator
from tortoise.contrib.pydantic import pydantic_model_creator
from tortoise.fields import (CharEnumField, CharField, DatetimeField,
                             IntEnumField, IntField, TextField, UUIDField)
from tortoise.models import Model as TTModel
from utils import num_tokens_from_string

from app.settings.config import config

PINECONE_HASH_KEY = '_hash'


def get_openai_model_type():
    import re
    from collections import Counter

    model_items = {}
    try:
        model_list_raw = openai_client.models.list().dict()['data']
        # model_list_raw = openai.Model.list(api_key=env.ORIGIN_OPENAI_API_KEYS[0],
        #                                    api_base=env.ORIGIN_OPENAI_API_BASE,
        #                                    proxy=env.ORIGIN_OPENAI_API_PROXY)['data']
        pattern = re.compile(r'[\D]')  # re.compile(r'[a-zA-Z]|[^0-9]')
        to_upper_or_under_score = (lambda match_obj:
                                   match_obj.group(0).upper() if match_obj.group(0).isalpha() else '_')
        model_enums = [(pattern.sub(to_upper_or_under_score, d['id']), d['id']) for d in model_list_raw]
        model_enums_count = Counter([e for e, n in model_enums])

        for model_enum, c in model_enums_count.items():
            filtered_model_enums = [n for e, n in model_enums if e == model_enum]
            if c == 1:
                model_items.update({model_enum: filtered_model_enums[0]})
            if c > 1:
                model_items.update({f'{model_enum}_{i}': n for i, n in enumerate(filtered_model_enums)})

        logger.info(f"Found {len(model_list_raw)} models")

    except Exception as e:
        logger.exception(f"An error occurred during the request: {e}")
    return Enum('OpenAIModelType', model_items)


OpenAIModelType = get_openai_model_type()


class AllowedFileType(str, Enum):
    ARXIV = 'arxiv'
    AUDIO = 'audio'
    CSV = 'csv'
    DOCX = 'docx'
    DOC = 'doc'
    EPUB = 'epub'
    HTML = 'html'
    MARKDOWN = 'markdown'
    NOTEBOOK = 'notebook'
    ODT = 'odt'
    PDF = 'pdf'
    POWERPOINT = 'powerpoint'
    PPTX = 'pptx'
    PPT = 'ppt'
    TXT = 'txt'
    REQUEST_BODY_CHUNKS = 'request_body_chunks'
    WEBPAGE = 'web_url'
    EXCEL = 'xlsx'
    JSON = 'json'
    MD = 'md'


class FileUploadStatus(str, Enum):
    INITIATED = 'initiated'
    QUEUING = 'queuing'
    INGESTING = 'ingesting'
    FAILED = 'failed'
    DONE = 'done'
    # deprecated
    DOWNLOAD_FAILED = 'url_fetch_failed'  # -
    CHUNKING_FAILED = 'chunking_failed'  # -
    # INITIATED = 'initiated'
    # EMBEDDING_FAILED = 'embedding_failed'
    # UPLOAD_FAILED = 'upload_failed'


class RawFileUploadStatus(str, Enum):
    INITIATED = 'initiated'
    INGESTING = 'ingesting'
    FAILED = 'failed'
    DONE = 'done'
    # deprecated
    EMBEDDING_FAILED = 'embedding_failed'  # -
    UPLOAD_FAILED = 'upload_failed'  # -


class SegmentUploadStatus(IntEnum):
    INITIATED = 0
    DONE = 1
    FAILED = -1


class EmbeddingPydantic(PDModel):
    vector: List[float]
    model: str
    prompt_tokens: int


class RawFiles(TTModel):
    """向量化文件信息：Segment文段指向这个表"""
    id = CharField(pk=True, max_length=32, unique=True,
                   description='md5 hash of the file and metadata')
    pinecone_namespace = TextField(default='', description='namespace for pinecone insertion')
    created_at = DatetimeField(auto_now_add=True, description='date time when created')
    chunk_size = IntField(description='num token of chunk', default=500)
    chunk_overlap = IntField(description='num token overlap between chunk', default=50)
    num_chunks = IntField(description='Total number of chunks', null=True)
    status = CharEnumField(RawFileUploadStatus, defualt=RawFileUploadStatus.INITIATED,
                           description='status of file upload')
    size = IntField(description='file size for files but synthetic number for chunks',
                    default=0)

    def __str__(self):
        return f'{str(self.id)}'

    async def get_files(self):
        files = await Files.filter(raw_id=self.id).all()
        return files

    async def get_segments(self):
        segments = await Segment.filter(raw_doc_id=self.id).all()
        return segments


RawFile_Pydantic = pydantic_model_creator(RawFiles, name='RawFile')


class Files(TTModel):
    """向量化文件信息：Files一对应一RawFiles"""
    id = UUIDField(pk=True, description='uuid associated with the file')
    metadata = TextField(default='{}', description='json string of file metadata')
    url = CharField(max_length=512, description='OSS path for the file', null=True)
    title = CharField(default='Missing Title', max_length=200,
                      description='title of the file for display')
    description = TextField(default='', description='description of the file')
    namespace = TextField(default='', description='pinecone partition')
    created_at = DatetimeField(auto_now_add=True, description='date time when created')
    raw_id = CharField(max_length=32, unique=True, null=True)
    status = CharEnumField(FileUploadStatus, defualt=FileUploadStatus.INITIATED,
                           description='status of file upload')

    def __str__(self):
        return f'{str(self.id)}({self.title[:10]}...)'

    async def get_raw_file(self) -> "RawFiles":
        if await RawFiles.exists(id=self.raw_id):
            return await RawFiles.get(id=self.raw_id)


File_Pydantic = pydantic_model_creator(Files, name='File')
FileIn_Pydantic = pydantic_model_creator(
    Files, name='FileIn', exclude=('created_at', 'status', 'raw_id'))


class File(File_Pydantic):
    metadata: dict

    @classmethod
    def from_file_pydantic(cls, obj: File_Pydantic) -> 'File':
        return cls(**{**obj.dict(), "metadata": json.loads(obj.metadata)})


class FilesBatch(PDModel):
    files: List[File_Pydantic]
    batch_size: int
    offset: int
    total: int


class FileStatus(File_Pydantic):
    raw_id: Optional[str] = None
    num_chunks: int
    status: FileUploadStatus
    status_raw: Optional[RawFileUploadStatus]

    @classmethod
    async def from_file(cls, file: Files):
        num_chunks = -1
        status_raw = None
        file_pdc = File_Pydantic.model_validate(file)
        related_raw_file = await file.get_raw_file()
        if related_raw_file:
            num_chunks = related_raw_file.num_chunks
            status_raw = related_raw_file.status
        return cls.model_validate({**file_pdc.model_dump(),
                                   'num_chunks': num_chunks,
                                   'status_raw': status_raw})


class Segment(TTModel):
    id = UUIDField(pk=True)
    raw_doc_id = CharField(max_length=32, unique=False)
    idx = IntField(description='zero indexed segment id of the file')
    text = TextField(null=True)
    metadata = TextField(null=True)
    upload_status = IntEnumField(SegmentUploadStatus, defualt=SegmentUploadStatus.INITIATED)

    class Meta:
        ordering = ('idx',)

    def __str__(self):
        return f'segment@{self.idx}: "{self.text[:50]}..."'

    async def get_raw_file(self) -> "RawFiles":
        if await RawFiles.exists(id=self.raw_doc_id):
            return await RawFiles.get(id=self.raw_doc_id)

    @classmethod
    async def escaped_bulk_create(cls, objs: List['Segment'], **kwargs):
        return await cls.bulk_create([obj.escape() for obj in objs], **kwargs)

    @classmethod
    async def escaped_bulk_update(cls, objs: List['Segment'], **kwargs):
        return await cls.bulk_update([obj.escape() for obj in objs], **kwargs)

    def escape(self):
        self.text = escape(self.text)
        metadata_dict = json.loads(self.metadata)
        if (text := metadata_dict.get(config.CHUNK_METADATA_QA_ANSWER)) is not None:
            metadata_dict[config.CHUNK_METADATA_QA_ANSWER] = escape(text)
        self.metadata = json.dumps(metadata_dict)
        return self


Segment_Pydantic = pydantic_model_creator(Segment, name='Segment')
SegmentIn_Pydantic = pydantic_model_creator(Segment,
                                            name='SegmentIn',
                                            exclude=('upload_status', 'raw_doc_id'),
                                            exclude_readonly=True)


class SegmentInEx(SegmentIn_Pydantic):
    idx: Optional[int]
    metadata: dict = {}
    additional_text: Optional[str] = None

    @classmethod
    def from_str_and_meta(cls, idx: int, chunk: str, metadata: dict, escape_text: bool = False):
        if escape_text:
            chunk = escape(chunk)
        return SegmentInEx(
            idx=idx, text=chunk, metadata=metadata,
            additional_text=metadata.get(config.CHUNK_METADATA_QA_ANSWER, None))


class SegmentEx(Segment_Pydantic):
    metadata: dict = {}
    file: Optional[File] = None
    additional_text: Optional[str] = None
    vdb_enabled: Optional[bool] = None

    @model_validator(mode='after')
    def set_vdb_status(cls, values):
        metadata = values.metadata
        if metadata:
            if config.CHUNK_METADATA_PINECONE_ENABLE not in metadata:
                values.vdb_enabled = True
            else:
                values.vdb_enabled = metadata[config.CHUNK_METADATA_PINECONE_ENABLE]

            if additional_text := metadata.get(config.CHUNK_METADATA_QA_ANSWER):
                values.additional_text = additional_text

        return values

    @field_validator('metadata', mode='before')
    @classmethod
    def validate_metadata(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    @classmethod
    def model_validate_with_file(cls, obj: Segment, file: Optional[File] = None, unescape_text: bool = False):
        obj = cls.model_validate(obj)

        if unescape_text:
            obj.text = unescape(obj.text)
            if obj.additional_text:
                obj.additional_text = unescape(obj.additional_text)

        obj.file = file
        return obj


def get_api_key_header(
        openai_api_key: str = Header(None, alias="openai_api_key"),
        openai_api_base: str = Header(None, alias="openai_api_base"),
        cohere_api_key: str = Header(None, alias="cohere_api_key")
):
    return APIKeyTag(
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        cohere_api_key=cohere_api_key
    )


class APIKeyTag(PDModel):
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    cohere_api_key: Optional[str] = None

    def is_platform_openai_api_key(self):
        return self.openai_api_key == config.ORIGIN_OPENAI_API_KEY

    def is_platform_cohere_api_key(self):
        return self.cohere_api_key == config.COHERE_API_KEY


class ConsumptionStatTag(PDModel):
    tenant_id: int
    agent_id: Optional[int] = None
    agent_appid: Optional[str] = None

    @model_validator(mode='after')
    def validate_agent_id_or_agent_appid(cls, values):
        if values.agent_id is None and values.agent_appid is None:
            raise ValueError('agent_id or agent_appid must be provided')
        return values


class ConsumptionTagged(PDModel):
    consumption_stat_tag: Optional[ConsumptionStatTag] = None


class SegmentAppend(PDModel):
    text: str
    metadata: dict = {}
    additional_text: Optional[str] = None
    token_limit: Optional[int] = 8000

    @model_validator(mode='after')
    def validate_text_token_limit(cls, values):
        text = values.text
        token_limit = values.token_limit
        if num_tokens_from_string(text) > token_limit:
            raise ValueError(f"chunk text token length is over the limit of {token_limit}")
        return values


class SegmentAppendData(ConsumptionTagged):
    chunks: List[SegmentAppend]
    file_id: UUID
    metadata: dict
    # consumption_stat_tag: Optional[ConsumptionStatTag] = None


class SegmentUpdate(SegmentAppend):
    id: str


class SegmentUpdateData(ConsumptionTagged):
    chunks: List[SegmentUpdate]
    metadata: dict
    # consumption_stat_tag: Optional[ConsumptionStatTag] = None


class StructuredFileInfo(FileIn_Pydantic, ConsumptionTagged):
    metadata: dict = {}
    file_type: AllowedFileType
    namespace: Optional[str] = config.PINECONE_DEFAULT_NAMESPACE
    description: str = ''
    title: str = ''
    uid: int = 0
    chunks: Optional[List[SegmentInEx]] = None
    chunk_size: int = 500
    chunk_overlap: int = 0
    disable_hash_check: bool = False
    extract_qa: bool = False


class ChunkBatch(ConsumptionTagged):
    chunk_uuids: List[str]
    chunks: List[str]
    chunk_metas: List[dict]
    namespace: str
    file_str: str
    total: int
    current: int


class QueryResultItem(SegmentEx):
    score: Optional[float]
    metadata: Optional[dict]
    vector: Optional[list[float]]
    rerank_score: Optional[float] = None

    @classmethod
    def from_segment_ex(cls, obj: SegmentEx,
                        score: Optional[float],
                        vector: Optional[list[float]],
                        rerank_score: Optional[float] = None):
        return cls(**obj.model_dump(), score=score, vector=vector, rerank_score=rerank_score)


class QueryResult(PDModel):
    chunks: List[QueryResultItem]
    rerank_chunks: List[QueryResultItem]


class RerankSettings(PDModel):
    model: Literal[
        'rerank-english-v2.0', 'rerank-multilingual-v2.0'
    ] = 'rerank-multilingual-v2.0'
    trigger_threshold: Optional[float] = None
    rerank_from_top_k: int = 20


class QueryBody(ConsumptionTagged):
    query_vector: List[float] = None
    rerank_settings: Optional[RerankSettings] = None


class Messages(PDModel):
    id: Optional[str] = PydanticField(description='message id')
    role: str = PydanticField(
        description='The role of the author of this message. '
                    'One of `system`, `user`, or `assistant`.')
    content: str = PydanticField(
        description='The contents of the message.')
    name: Optional[str] = PydanticField(
        default=None,
        description='The name of the author of this message. '
                    'May contain a-z, A-Z, 0-9, and underscores, '
                    'with a maximum length of 64 characters.')


class ContextMessages(PDModel):
    msgs: List[Messages] = PydanticField(description='a list of messages')

    @classmethod
    def chat_example(cls):
        return cls(
            msgs=[
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174000",
                    role='system',
                    content='You are a helpful assistant.',
                ),
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174003",
                    role='user',
                    content='1+1=?'
                ),
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174004",
                    role='assistant',
                    content="it is 2, you moron!",
                ),
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174005",
                    role='user',
                    content='count from 1 to 6, please'
                ),

            ]
        )

    @classmethod
    def doc_example(cls):
        return cls(
            msgs=[
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174001",
                    role='system',
                    content='You are an helpful assistant.',
                ),
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174003",
                    role='assistant',
                    content="It is about a language model.",
                ),
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174004",
                    role='user',
                    content='When is the paper, published?'
                ),
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174005",
                    role='assistant',
                    content="On November 26th 2019.",
                ),
                Messages(
                    id="123e4567-e89b-12d3-a456-426614174006",
                    role='user',
                    content='What is the methodology of single headed attention?'
                ),
            ]
        )

    def preprocess(self, model_type: OpenAIModelType, max_ctx_token: Optional[int]):
        from utils import num_tokens_from_messages, openai_msg_from

        est_num_input_ctx_token = num_tokens_from_messages(
            [openai_msg_from(msg.dict()) for msg in self.msgs])

        input_ctx_msg_ids = [msg.id for msg in self.msgs]

        input_ctx_openai_msg = [openai_msg_from(msg.dict()) for msg in self.msgs]
        return input_ctx_msg_ids, input_ctx_openai_msg, est_num_input_ctx_token


class ChatResponse(PDModel):
    input_ctx_msg_ids: Optional[List[str]] = None
    input_msgs: Optional[List[Messages]] = None
    output_msg: Optional[Messages] = None
    prompt_tokens: int
    completion_tokens: int
    delta: Optional[dict] = None
    finish_reason: Optional[str] = None


class FileDeleteResponse(PDModel):
    file_id: Optional[str] = None
    raw_file_id: Optional[str] = None
    chunk_ids: Optional[List[str]] = None
    depending_file_ids: Optional[List[str]] = None
    msg: str


class SegmentDeleteResponse(PDModel):
    deleted_ids: Optional[List[str]] = None
    non_existed_ids: Optional[List[str]] = None
    msg: str


class MsgResponse(PDModel):
    msg: str


class TokenConsumptionPayload(ConsumptionStatTag):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str
    vectors: Optional[int] = None
    used_count: int = 0
    need_deduct: bool = Field(True, title='是否需要扣费')
    type: int = Field(description='1: chat, 2:embedding, 3:stt, 4:tts, 5:rerank, 6:image_gen')
    req_uuid: str = Field(default_factory=lambda: str(uuid4()))


class TextResponse(PDModel):
    text: str


class FilePreciseSearchRequest(PDModel):
    keyword: Optional[str] = Field("", description="关键词")
    tenant_id: Optional[int] = Field(0, description="租户ID")
    limit: Optional[int] = Field(10, description="数量限制")


class ChunkFilesAsyncStatus(str, Enum):
    pending = 'pending'
    processing = 'processing'
    success = 'success'
    failure = 'failure'
    notfound = 'notfound'


class ChunkFilesAsyncTask(PDModel):
    data: list[SegmentInEx] = Field(default=[], description="File content preview")
    status: ChunkFilesAsyncStatus = Field(description="task status")
    msg: str = Field(description="status explanation")

    @classmethod
    def from_json(cls, json_data):
        return cls(**json.loads(json_data))

    @classmethod
    def not_found_task(cls):
        return cls(status=ChunkFilesAsyncStatus.notfound, msg='任务已过期或不存在')

    @classmethod
    def pending_task(cls):
        return cls(status=ChunkFilesAsyncStatus.pending, msg='等待开始')

    @classmethod
    def processing_task(cls):
        return cls(status=ChunkFilesAsyncStatus.processing, msg='正在执行中')

    @classmethod
    def success_task(cls, data):
        return cls(status=ChunkFilesAsyncStatus.success, data=data, msg='任务完成')

    @classmethod
    def failure_task(cls, msg=None):
        msg = msg or '执行错误'
        return cls(status=ChunkFilesAsyncStatus.failure, msg=msg)


class ChunkFilesAsyncJob(PDModel):
    task_id: UUID
    url: str
    file_extension: str
    chunk_size: Optional[int]
    chunk_overlap: Optional[int]
    chunk_for_qa_extraction: Optional[bool]

    @classmethod
    def from_json(cls, json_data):
        return cls(**json.loads(json_data))


class FileUpdateMetaParam(PDModel):
    file_id: Optional[str] = Field("", title="文件id")
    out_url: str = Field("", title="文件的外部链接")
