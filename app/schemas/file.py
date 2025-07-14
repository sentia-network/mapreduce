import re
from enum import Enum
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, HttpUrl

from app.schemas.chunk import ChunkConfig
from app.schemas.public import (EmbModelConfMixin, FacadeAPIKeyTagMixin,
                                FacadeExtraMixin)


class AllowedDocumentSuffix(str, Enum):
    DOCX = '.docx'
    DOC = '.doc'
    PDF = '.pdf'
    PPTX = '.pptx'
    TXT = '.txt'
    MD = '.md'
    EPUB = '.epub'
    ZIP = '.zip'


class AllowedImageSuffix(str, Enum):
    GIF = '.gif'
    PNG = '.png'
    JPG = '.jpg'
    JPEG = '.jpeg'
    WEBP = '.webp'
    BMP = '.bmp'
    TIFF = '.tiff'
    TIF = '.tif'


class AllowedVideoSuffix(str, Enum):
    MP4 = '.mp4'


class AllowedAudioSuffix(str, Enum):
    MP3 = '.mp3'
    WAV = '.wav'
    PCM = '.pcm'
    OPUS = '.opus'
    FLAC = '.flac'
    OGG = '.ogg'
    M4A = '.m4a'
    AMR = '.amr'
    SPEEX = '.speex'
    LYB = '.lyb'
    AC3 = '.ac3'
    AAC = '.aac'
    APE = '.ape'
    M4R = '.m4r'
    ACC = '.acc'
    WMA = '.wma'


class AllowedDataBaseSuffix(str, Enum):
    SQL = '.sql'


class AllowedTableSuffix(str, Enum):
    XLS = '.xls'
    XLSX = '.xlsx'
    CSV = '.csv'


AllowedFileSuffix = (AllowedDocumentSuffix |
                     AllowedImageSuffix |
                     AllowedVideoSuffix |
                     AllowedAudioSuffix |
                     AllowedDataBaseSuffix |
                     AllowedTableSuffix)


class MarkdownAssetContent(BaseModel):
    raw: str = Field(description='匹配到模式的原文本')
    alt: str | None = Field('', description='alt 文本')
    path: HttpUrl | Path = Field('', description='url连接或者路径')
    title: str | None = Field('', description='可选标题')

    def model_post_init(self, __context):
        if self.title is None:
            self.title = ''
        if self.alt is None:
            self.alt = ''

    @classmethod
    def parse_from_text(cls, text: str) -> List["MarkdownAssetContent"]:
        pattern = r'!\[(.*?)\]\((.*?)(?:\s*"(.*?)")?\)'
        match_ls = re.finditer(pattern, text)
        result = []
        if not match_ls:
            return result
        for m in match_ls:
            try:
                allow = cls(raw=m[0], alt=m[1], path=m[2], title=m[3])
                result.append(allow)
            except Exception as _:
                print(_)

        return result


class DatasetCreate(FacadeExtraMixin, EmbModelConfMixin, FacadeAPIKeyTagMixin):
    file_url: HttpUrl = Field(description='文件url地址')
    file_name: str = Field(description='文件名称')
    file_suffix: AllowedFileSuffix = Field(description='文件后缀')
    chunk_config: ChunkConfig = Field(default_factory=ChunkConfig, description='切片配置')
    is_qa: bool = Field(True, description='是否为问答表格，当后缀为 .xlsx 和 .csv 时有效')
    max_len: int = Field(100000, description='最大文本长度')
    create_outline: bool = Field(False, description='是否生成大纲')
