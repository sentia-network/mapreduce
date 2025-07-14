from typing import Optional

from aiokafka import AIOKafkaConsumer

from app.core.kafka import BaseConsumer
from app.core.logger import get_logger
from app.schemas.kafka import KafkaJobEmbedCreateMessage
from app.schemas.public import FacadeData
from app.services.external.facade import FacadeExtraContext
from app.services.job import JobService

logger = get_logger("Embed Create consumer")


class EmbedCreateConsumer(BaseConsumer):

    def __init__(self, serial_no: int = 1) -> None:
        super().__init__(name='DatasetCreate', serial_no=serial_no, msg_pydantic_model=KafkaJobEmbedCreateMessage)

    async def handle_msg(self, msg: KafkaJobEmbedCreateMessage):
        logger.info(f'consumer: {self.name}')
        with FacadeExtraContext(FacadeData(extra=msg.data.extra, facade_api_key=msg.data.facade_api_key)):
            await JobService.process_embed_create_job(msg)

    @staticmethod
    async def consume_dataset_create_task(consumer: AIOKafkaConsumer, serial_no: Optional[int] = 1):
        await EmbedCreateConsumer(serial_no).consume_task(consumer)
