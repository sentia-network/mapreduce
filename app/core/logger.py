import json
import logging
import os
import sys
import traceback
import warnings
from pprint import pformat

from loguru import logger
from loguru._defaults import env

from app.middlewares.request_ctx import RequestContext, RequestContextualize
from app.settings.config import config


class InterceptHandler(logging.Handler):
    """
    Default handler from examples in loguru documentaion.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def json_serialize(record):
    json_format = {
        "timestamp": record["time"].strftime('%Y-%m-%d %H:%M:%S.%f'),
        "pid": str(record["process"].id),
        "level": record["level"].name,
        "xRequestId": ctx.request_id if (ctx := RequestContextualize.get_ctx()) is not None else "",
        "appName": "aigc-mapreduce",
        "threadName": str(record['thread'].id),
        "className": f"{record['name']}",
        "methodName": f"{record['function']}",
        "codeFile": f"{record['file']}",
        "line": f"{record['line']}",
        "message": record["message"],
        "stackTrace": "",
    }

    if record["exception"] is not None:
        json_format["stackTrace"] = "".join(
            traceback.format_exception(
                record["exception"].type,
                record["exception"].value,
                record["exception"].traceback,
            )
        )

    return json.dumps(json_format)


def patching(record):

    record["extra"]["serialized"] = json_serialize(record)


def format_record(record: dict) -> str:
    """
    Custom format for loguru loggers.
    Uses pformat for log any data like request/response body during debug.
    Works with logging if loguru handler it.
    Example:
    >>> payload = [{"users":[{"name": "Nick", "age": 87, "is_active": True}, {"name": "Alex", "age": 27, "is_active": True}], "count": 2}]
    >>> logger.bind(payload=).debug("users payload")
    >>> [   {   'count': 2,
    >>>         'users': [   {'age': 87, 'is_active': True, 'name': 'Nick'},
    >>>                      {'age': 27, 'is_active': True, 'name': 'Alex'}]}]
    """

    LOGURU_FORMAT2 = env(
        "LOGURU_FORMAT",
        str,
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> aigc-mapreduce [PID:{process}] [TID:{thread}] "
        "[<level>{level}</level>] "
        "[<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>] - <level>{message}</level>",
    )

    ctx = RequestContextualize.get_ctx()
    if ctx is not None and isinstance(ctx, RequestContext):
        LOGURU_FORMAT2 = env(
            "LOGURU_FORMAT",
            str,
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> aigc-mapreduce [PID:{process}] [TID:{thread}] "
            "[<level>{x-request-id}</level>] "
            "[<level>{level}</level>] "
            "[<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan>] - <level>{message}</level>",
        )
        record['x-request-id'] = ctx.request_id

    format_string = LOGURU_FORMAT2
    if record["extra"].get("payload") is not None:
        record["extra"]["payload"] = pformat(
            record["extra"]["payload"], indent=4, compact=True, width=88
        )
        format_string += "\n<level>{extra[payload]}</level>"

    format_string += "{exception}\n"
    return format_string


def init_logging():
    """
    Replaces logging handlers with a handler for using the custom handler.

    WARNING!
    if you call the init_logging in startup event function,
    then the first logs before the application start will be in the old format
    >>> app.add_event_handler("startup", init_logging)
    stdout:
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
    INFO:     Started reloader process [11528] using statreload
    INFO:     Started server process [6036]
    INFO:     Waiting for application startup.
    2020-07-25 02:19:21.357 | INFO     | uvicorn.lifespan.on:startup:34 - Application startup complete.

    """

    showwarning_ = warnings.showwarning

    def showwarning(message, *args, **kwargs):
        logger.warning(message)
        showwarning_(message, *args, **kwargs)

    warnings.showwarning = showwarning

    # disable handlers for specific uvicorn loggers
    # to redirect their output to the default uvicorn logger
    # works with uvicorn==0.11.6
    loggers = (
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if name.startswith("uvicorn.")
    )
    for uvicorn_logger in loggers:
        uvicorn_logger.handlers = []

    # change handler for default uvicorn logger
    intercept_handler = InterceptHandler()
    log_name_ls = ['uvicorn', 'qcloud_cos', 'httpx', 'aiokafka', 'elasticsearch']
    for name in log_name_ls:
        logging.getLogger(name).handlers = []
        logging.getLogger(name).handlers = [intercept_handler]
    # logconf = settings.get("LOG")
    logconf = {
        "level": config.get("LOG_LEVEL", "INFO")
    }
    is_json_format = logconf.get("is_json_format")
    setup_logging()
    if is_json_format is not None:
        logconf.pop("is_json_format")
    # set logs output, level and format
    logger.configure(
        patcher=lambda record: patching(record=record),
        handlers=[{
            "sink": sys.stdout,
            "format": "{extra[serialized]}" if config.get('IS_JSON_FORMAT', None) else format_record,
            **logconf
        }]
    )


# -----logging-----
log_format = (
    "%(asctime)s aigc-mapreduce [PID:%(process)d] [TID:%(thread)d] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
)


def setup_logging(log_file=None):
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    _nameToLevel = {
        'CRITICAL': logging.CRITICAL,
        'FATAL': logging.FATAL,
        'ERROR': logging.ERROR,
        'WARN': logging.WARNING,
        'WARNING': logging.WARNING,
        'INFO': logging.INFO,
        'DEBUG': logging.DEBUG,
        'NOTSET': logging.NOTSET,
    }
    logging.basicConfig(format=log_format, level=_nameToLevel[log_level])


def get_logger(name=""):
    return logger
