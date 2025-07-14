from pydantic import Field

from app.schemas.embed import EmbedCreateWithFileUrl
from app.schemas.file import DatasetCreate
from app.schemas.public import KafkaBaseMessage


class KafkaJobDatasetMessage(KafkaBaseMessage):
    data: DatasetCreate = Field(description='数据集处理任务数据')


class KafkaJobEmbedCreateMessage(KafkaBaseMessage):
    data: EmbedCreateWithFileUrl = Field(description='向量化处理任务数据')
