from enum import Enum
from typing import Any, List, Optional

from fastapi import Header
from loguru import logger
from openai.types.completion_usage import CompletionUsage as OAICompletionUsage
from openai.types.create_embedding_response import Usage as OAIEmbeddingUsage
from pydantic import BaseModel, Field, model_validator

from app.middlewares.request_ctx import RequestContextualize
from app.schemas.chunk import QueryContentIndexedOptional
from app.settings.config import config


class FacadeExtraMixin(BaseModel):
    extra: dict = Field(default_factory=dict, description='额外参数')

    @classmethod
    def extra_examples(cls):
        return {'user_id': 1, 'tenant_id': 20,
                'dataset_id': 7219259252294443008,
                'job_id': 7221717360558727168,
                'knowledge_base_id': 7219259166806138880}


class EmbModelSuppliers(str, Enum):
    OPENAI = 'OpenAI'
    MINIMAX = 'MiniMax'


class CplModelSuppliers(str, Enum):
    OPENAI = 'OpenAI'


class RerankModelSuppliers(str, Enum):
    COHERE = 'cohere'


class OpenAICplModels(str, Enum):
    GPT_4O = 'gpt-4o'


class OpenAIEmbModels(str, Enum):
    # ### !!!! DO NOT MODIFY THIS ENUM, VDB INDEX NAME DEPEND ON THIS !!! ###
    # ### !!!! 不要修改此枚举，将影响向量库索引名依赖 !!! ###
    ADA002 = 'text-embedding-ada-002'
    LARGE = 'text-embedding-3-large'
    SMALL = 'text-embedding-3-small'
    MINIMAX_EMBO01 = 'embo-01'

    @classmethod
    def is_member(cls, value):
        try:
            cls(value)
            return True
        except ValueError:
            return False


class OcrModel(str, Enum):
    BAIDU = 'baidu_ocr_accurate'


class OpenAIEmbModelDim(int, Enum):
    # ### !!!! DO NOT MODIFY THIS ENUM, VDB INDEX NAME DEPEND ON THIS !!! ###
    # ### !!!! 不要修改此枚举，将影响向量库索引名依赖 !!! ###
    # https://openai.com/index/new-embedding-models-and-api-updates/
    ADA002 = 1536
    LARGE = 3072
    SMALL = 1536
    MINIMAX_EMBO01 = 1536


class CohereRerankModels(str, Enum):
    # https://docs.cohere.com/reference/rerank
    RERANK_ENGLISH_V3_0 = 'rerank-english-v3.0'
    RERANK_MULTILINGUAL_V3_0 = 'rerank-multilingual-v3.0'

    # Deprecated
    # RERANK_ENGLISH_V2_0 = 'rerank-english-v2.0'
    # RERANK_MULTILINGUAL_V2_0 = 'rerank-multilingual-v2.0'


class CohereRerankUsage(BaseModel):
    count: int = Field(description='使用次数')


class ContentType(str, Enum):
    RAW = 'raw'
    QUERY = 'query'


class RerankConfig(BaseModel):
    top_k: int = Field(10, title='rerank top k')
    model: CohereRerankModels = CohereRerankModels.RERANK_MULTILINGUAL_V3_0
    content_type: ContentType = Field(ContentType.RAW, title='rerank内容类型')
    supplier: RerankModelSuppliers = RerankModelSuppliers.COHERE


class OrcUsage(BaseModel):
    model: OcrModel = Field(description='模型名称')
    used_count: int = Field(description='使用次数', ge=1)


class LLMUsage(BaseModel):
    model: OpenAICplModels | OpenAIEmbModels | CohereRerankModels = Field(description='模型名称')
    supplier: CplModelSuppliers | EmbModelSuppliers | RerankModelSuppliers = Field(description='模型供应商')
    usage: OAICompletionUsage | OAIEmbeddingUsage | CohereRerankUsage = Field(description='模型使用量')

    @classmethod
    def from_llm_resp(
            cls, model: OpenAICplModels | OpenAIEmbModels,
            supplier: CplModelSuppliers | EmbModelSuppliers,
            usage: OAICompletionUsage | OAIEmbeddingUsage
    ) -> 'LLMUsage':
        return cls(model=model, supplier=supplier, usage=usage)

    @classmethod
    def null_embed_usage(
            cls, model: OpenAIEmbModels, supplier: EmbModelSuppliers
    ) -> 'LLMUsage':
        return cls(model=model, supplier=supplier, usage=OAIEmbeddingUsage(prompt_tokens=0, total_tokens=0))

    @classmethod
    def null_rerank_usage(
            cls, model: CohereRerankModels, supplier: RerankModelSuppliers
    ) -> 'LLMUsage':
        return cls(model=model, supplier=supplier, usage=CohereRerankUsage(count=0))

    def __add__(self, other):
        if any([other.supplier != self.supplier,
                other.model != self.model,
                not isinstance(other.usage, type(self.usage))]):
            raise ValueError('不同的模型供应商或模型无法相加')
        if isinstance(self.usage, OAIEmbeddingUsage):
            return LLMUsage(
                model=self.model, supplier=self.supplier,
                usage=OAIEmbeddingUsage(prompt_tokens=self.usage.prompt_tokens + other.usage.prompt_tokens,
                                        total_tokens=self.usage.total_tokens + other.usage.total_tokens))
        return NotImplemented

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    @staticmethod
    def sum(usages: list['LLMUsage']) -> 'LLMUsage':
        return sum(usages)


class CplModelConf(BaseModel):
    supplier: CplModelSuppliers = Field(CplModelSuppliers.OPENAI, title='模型供应商')
    model: OpenAICplModels = Field(OpenAICplModels.GPT_4O, title='模型名称')
    seed: int = Field(0, title='随机种子')
    temperature: float = Field(0.0, title='温度')

    def to_kw(self):
        return self.model_dump(exclude={'supplier'}, mode='json')


class EmbModelConf(BaseModel):
    supplier: EmbModelSuppliers = Field(description='模型供应商')
    model: OpenAIEmbModels = Field(description='模型名称')
    dim: int | None = Field(None, description='向量维度, 无则用模型维度')

    @model_validator(mode='after')
    def check(self):
        match self.supplier:
            case EmbModelSuppliers.OPENAI:
                match self.model:
                    case OpenAIEmbModels.ADA002:
                        match self.dim:
                            case None:
                                self.dim = OpenAIEmbModelDim.ADA002
                            case dim if dim != OpenAIEmbModelDim.ADA002:
                                raise ValueError(f'Invalid dim value for {OpenAIEmbModels.ADA002}')
                    case OpenAIEmbModels():
                        match self.dim:
                            case None:
                                self.dim = getattr(OpenAIEmbModelDim, self.model.name)
                            case dim if dim < 1 or dim > getattr(OpenAIEmbModelDim, self.model.name):
                                raise ValueError(f'Invalid dim value for {self.model}')
                    case _:
                        raise ValueError(f'Invalid model name for {EmbModelSuppliers.OPENAI.value}')
            case EmbModelSuppliers.MINIMAX:
                match self.model:
                    case OpenAIEmbModels.MINIMAX_EMBO01:
                        match self.dim:
                            case None:
                                self.dim = OpenAIEmbModelDim.MINIMAX_EMBO01
                            case dim if dim != OpenAIEmbModelDim.MINIMAX_EMBO01:
                                raise ValueError(f'Invalid dim value for {OpenAIEmbModels.MINIMAX_EMBO01}')
        return self

    @classmethod
    def default(cls):
        return cls(supplier=EmbModelSuppliers.OPENAI, model=OpenAIEmbModels.ADA002)


class ChunkUpdateMode(str, Enum):
    REPLACE = 'replace'
    MERGE = 'merge'


class ChunkUpdateMixin(BaseModel):
    items: List[QueryContentIndexedOptional] = Field(..., title='切片列表')
    update_mode: ChunkUpdateMode = Field(
        'replace', title='元数据更新模式', description='选择replace则替换原有元数据，选择merge则合并元数据')

    @model_validator(mode='after')
    def check_update_mode(self):
        if self.update_mode == ChunkUpdateMode.REPLACE:
            if any([item.metadata is None for item in self.items]):
                raise ValueError('元数据更新模式为replace时，元数据不能为None!')
        return self


class EmbModelConfMixin(BaseModel):
    embed_config: EmbModelConf = Field(..., title='向量化模型配置')


class MultiEmbModelConfMixin(BaseModel):
    embed_configs: List[EmbModelConf] | EmbModelConf = Field(..., title='向量化模型配置')

    @model_validator(mode='after')
    def check_embed_configs(self):
        if isinstance(self.embed_configs, EmbModelConf):
            self.embed_configs = [self.embed_configs]
        return self


class KafkaBaseMessage(BaseModel):
    x_request_id: str = Field(default_factory=lambda: RequestContextualize.get_request_id(), description='追踪ID')
    topic: str = Field(description='消息主题')
    data: Any | None = Field(None, description='任务数据')


class AssetType(str, Enum):
    IMAGE = 'image'


class AssetInfo(BaseModel):
    aid: int = Field(description='素材ID')
    path: str = Field(description='素材路径')
    atype: AssetType = Field(description='素材类型')
    size: int = Field(description='素材大小，单位B', ge=0)
    suffix: str = Field(description='素材后缀')
    title: str = Field('', description='素材标题')
    description: str = Field('', description='素材描述')

    @model_validator(mode='after')
    def check_suffix(self):
        if not self.suffix.startswith('.'):
            raise ValueError('Suffix must start with "."')
        return self

    def to_md_content(self):
        return f'![{self.atype.value}]({self.path} "{self.aid}")'


class FacadeAPIKeyTag(BaseModel):
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    cohere_api_key: Optional[str] = None

    def is_platform_openai_api_key(self):
        is_platform = self.openai_api_key == config.LLM_API_KEY
        logger.info(f'openai api key is platform: {is_platform}')
        return is_platform

    def is_platform_cohere_api_key(self):
        is_platform = self.cohere_api_key == config.COHERE_API_KEY
        logger.info(f'cohere api key is platform: {is_platform}')
        return is_platform

    def has_openai_api_key(self):
        has_key = self.openai_api_key is not None
        logger.info(f'request has openai api key: {has_key}')
        return has_key

    def has_cohere_api_key(self):
        has_key = self.cohere_api_key is not None
        logger.info(f'request has cohere api key: {has_key}')
        return has_key

    @classmethod
    def get_api_key_header(
            cls,
            openai_api_key: str = Header(None, alias="openai_api_key"),
            openai_api_base: str = Header(None, alias="openai_api_base"),
            cohere_api_key: str = Header(None, alias="cohere_api_key")
    ):
        return cls(
            openai_api_key=openai_api_key if openai_api_key else None,
            openai_api_base=openai_api_base if openai_api_base else None,
            cohere_api_key=cohere_api_key if cohere_api_key else None
        )


class FacadeAPIKeyTagMixin(BaseModel):
    facade_api_key: None | FacadeAPIKeyTag = Field(None, title='API Key')


class FacadeData(FacadeExtraMixin, FacadeAPIKeyTagMixin):
    pass
