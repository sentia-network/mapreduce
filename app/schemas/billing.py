from enum import Enum
from typing import Dict, Optional
from uuid import uuid4

from pydantic import Field

from app.schemas.public import FacadeExtraMixin


class UsageType(int, Enum):
    CHAT = 1
    EMBEDDING = 2
    STT = 3
    TTS = 4
    RERANK = 5
    IMAGE_GEN = 6
    VOICE_CLONE = 7
    OCR = 8


class UsagePayload(FacadeExtraMixin):
    prompt_tokens: int = Field(0, title='请求token数')
    completion_tokens: int = Field(0, title='返回token数')
    total_tokens: int = Field(0, title='总token数')
    model: str = Field('', title='模型名称')
    vectors: Optional[int] = None
    used_count: int = Field(0, title='使用次数')
    need_deduct: bool = Field(True, title='是否需要扣费')
    type: UsageType = Field(
        description='1: chat, 2:embedding, 3:stt, 4:tts, 5:rerank, 6:image_gen, 7:voice_clone, 8: ocr')
    req_uuid: str = Field(default_factory=lambda: str(uuid4()))

    def to_body(self) -> Dict:
        body = self.model_dump(mode='json')
        extra = body.pop('extra')
        return extra | body
