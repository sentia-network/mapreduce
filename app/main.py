# pylint: disable=E0611,E0401
import asyncio
import contextlib
import json

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError

from app.api.routers import admin_router
from app.core.kafka import MQManager
from app.core.logger import get_logger, init_logging
from app.middlewares.exception.common import CommonException
from app.middlewares.middlewares import (ExceptionMiddleware,
                                         RequestContextMiddleware)
from app.services.consumers.dataset_create import DatasetCreateConsumer
from app.services.consumers.embed import EmbedCreateConsumer
from app.settings.config import config

init_logging()
logger = get_logger('app')
logger.info(f"VDB type: {config.VDB_TYPE}")


async def Launch_consumer():
    await asyncio.sleep(3)

    logger.info(f"init kafka consumer: {config.KAFKA_TOPIC_DATASET_JOB}")
    await MQManager().consumer_startup(topic=config.KAFKA_TOPIC_DATASET_JOB,
                                       consume_callback=DatasetCreateConsumer.consume_dataset_create_task,
                                       count=int(config.KAFKA_TOPIC_DATASET_CONSUMER_COUNT))

    logger.info(f"init kafka consumer: {config.KAFKA_TOPIC_EMBED_CREATE_JOB}")
    await MQManager().consumer_startup(topic=config.KAFKA_TOPIC_EMBED_CREATE_JOB,
                                       consume_callback=EmbedCreateConsumer.consume_dataset_create_task,
                                       count=int(config.KAFKA_TOPIC_EMBED_CREATE_CONSUMER_COUNT))

    while True:
        await asyncio.sleep(5)


def launch_mq_wrapper(loop):
    asyncio.set_event_loop(loop)
    loop.create_task(Launch_consumer())


@contextlib.asynccontextmanager
async def lifespan(_app):
    logger.info(_app)

    logger.info("startup: init kafka producer")
    await MQManager().producer_startup()

    logger.info("startup: threading kafka start consumer")
    import threading
    threading.Thread(target=launch_mq_wrapper, args=(asyncio.get_event_loop(),), daemon=True).start()

    yield

    logger.info("shutdown: shutdown kafka producer and consumer")
    await MQManager().shutdown()


docs_url = '/docs'
redoc_url = '/redoc'
openapi_url = '/openapi.json'

if config.ENV == 'prod':
    docs_url = None
    redoc_url = None
    openapi_url = None

app = FastAPI(title="MapReduce API",
              description='a collection of APIs that provide document upload and conversational interaction',
              version='0.9.8',
              lifespan=lifespan,
              docs_url=docs_url,
              redoc_url=redoc_url,
              openapi_url=openapi_url, )


# 统一输出参数
async def request_info(request: Request):
    logger.info(f'path: {request.method} {request.url.path}')
    logger.info('headers\n' + json.dumps(dict(request.headers), ensure_ascii=False, indent=4))
    if request.headers.get('content-type') == 'application/json':
        body = await request.json()
        logger.info('body\n' + json.dumps(body, indent=4, ensure_ascii=False))


app.include_router(admin_router, dependencies=[Depends(request_info)])

app.add_middleware(ExceptionMiddleware)
app.add_middleware(RequestContextMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    处理请求验证异常。

    当请求数据验证失败时，收集并记录所有验证错误的详细信息，并抛出一个自定义的异常。
    """
    detail = ''
    body = await request.body()
    logger.error(f"invalidate body\n{(body.decode('utf-8'))}")

    for err in exc.errors():
        detail += f"{str(err.get('loc'))}:{err.get('msg')};"
    logger.error(detail)
    raise CommonException.parameter_invalid(detail=detail)


if config.ENV != 'prod':
    openapi_schema = app.openapi().copy()
    openapi_schema['servers'] = [{'url': 'http://127.0.0.1:8000'}]
    with open('openapi.json', 'w') as f:
        json.dump(openapi_schema, f, indent=4, ensure_ascii=False)
