import asyncio
import json
import traceback
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from pydantic import BaseModel

from app.core.logger import get_logger
from app.middlewares.request_ctx import RequestContext, RequestContextualize
from app.schemas.public import KafkaBaseMessage
from app.settings.config import config

logger = get_logger("MQ")


class MQManager:
    _instance: Optional['MQManager'] = None
    _producer: Optional[AIOKafkaProducer] = None
    _consumers: Dict[str, List[AIOKafkaConsumer]] = {}
    _tasks: Optional[List[asyncio.Task]] = []
    _enable: int = 0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._enable = config.KAFKA_ENABLE
        return cls._instance

    def producer(self):
        if self._producer is not None:
            return self._producer
        self._producer = AIOKafkaProducer(
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            acks=config.KAFKA_PRODUCER_ACKS,
            linger_ms=config.KAFKA_PRODUCER_LINGER_MS,
        )
        return self._producer

    @staticmethod
    def consumer(topic: str, group_id: Optional[str] = None, enable_auto_commit: Optional[bool] = True):
        # consumers = self._consumers.get(topic)
        # if consumers:
        #     return consumers[0]

        group_id = group_id if group_id and len(group_id) > 0 else config.KAFKA_CONSUMER_DEFAULT_GROUP
        consumer = AIOKafkaConsumer(
            topic,
            bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS,
            group_id=group_id,
            auto_offset_reset=config.KAFKA_CONSUMER_AUTO_OFFSET_RESET,
            max_poll_records=config.KAFKA_CONSUMER_MAX_POLL_RECORDS,
            session_timeout_ms=config.KAFKA_CONSUMER_SESSION_TIMEOUT_MS,
            heartbeat_interval_ms=config.KAFKA_CONSUMER_HEARTBEAT_INTERVAL_MS,
            consumer_timeout_ms=config.KAFKA_CONSUMER_TIMEOUT_MS,
            enable_auto_commit=enable_auto_commit
        )

        # if topic not in self._consumers:
        #     self._consumers[topic] = []
        #
        # self._consumers[topic].append(consumer)
        return consumer

    async def producer_startup(self):
        if self._enable == 0:
            return
        await self.producer().start()

    async def consumer_startup(self, topic: str, consume_callback: Callable, count: Optional[int] = 1,
                               enable_auto_commit: Optional[bool] = True, group: Optional[str] = None):
        if self._enable == 0:
            return

        serial_no = 1
        while True:
            if count <= 0:
                break

            count = count - 1
            consumer = self.consumer(topic, enable_auto_commit=enable_auto_commit, group_id=group)
            await consumer.start()

            if topic not in self._consumers:
                self._consumers[topic] = []
            self._consumers[topic].append(consumer)

            task = asyncio.create_task(consume_callback(consumer, serial_no))
            self._tasks.append(task)
            serial_no = serial_no + 1

    async def shutdown(self):
        if self._enable == 0:
            return

        if self._producer:
            logger.info("kafka stop producer")
            # stop producer
            await self._producer.stop()

        # stop consumers
        for topic in self._consumers:
            consumers = self._consumers[topic]
            for consumer in consumers:
                await consumer.stop()

        # cancel asyncio celery
        for task in self._tasks:
            task.cancel()

        logger.info("kafka stop consumers")
        # wait for tasks canceled
        await asyncio.gather(*self._tasks, return_exceptions=True)

    async def produce_message(self, message: KafkaBaseMessage, raw=False, reraise: bool = True):
        if self._enable == 0:
            return
        try:
            producer = self.producer()
            if raw:
                value = json.dumps(message.model_dump()['data'], ensure_ascii=False).encode('utf-8')
                value_str = json.dumps(message.model_dump()['data'], ensure_ascii=False, indent=4)
            else:
                value = message.model_dump_json().encode('utf-8')
                value_str = message.model_dump_json(indent=4)
            await producer.send(message.topic, value)
            logger.info(f"Message sent to topic '{message.topic}': \n{value_str}")
        except Exception as e:
            logger.error(f"Error sending message to topic '{message.topic}': {e}")
            logger.info("restart producer")
            try:
                await self._producer.stop()
            except Exception as ignore_e:
                logger.info(f"ignore e self._producer.stop() {ignore_e}")
            self._producer = None
            await self.producer().start()

            if reraise:
                raise e


class BaseConsumer(ABC):

    def __init__(self, name: str, msg_pydantic_model: Type[BaseModel], serial_no: int = 1):
        super().__init__()
        self.name = f"{name}:#{serial_no}"
        self.msg_pydantic_model = msg_pydantic_model
        logger.info(f"<{self.name} init>")

    async def consume_task(self, consumer: AIOKafkaConsumer):
        async for m in consumer:
            msg = self.msg_pydantic_model.model_validate_json(m.value.decode('utf-8'))
            ctx = RequestContext(request_id=msg.x_request_id)

            with RequestContextualize(ctx):
                logger.info(f'{self.name} consume topic {msg.topic}: \n' + msg.model_dump_json(indent=4))

                try:
                    await self.handle_msg(msg)
                    logger.info('finish message consume')

                except Exception as e:
                    logger.error(traceback.format_exc())
                    logger.error(f'except message: {str(e)}')

                finally:
                    await consumer.commit()

    @abstractmethod
    async def handle_msg(self, msg):
        ...
