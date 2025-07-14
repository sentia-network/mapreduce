import uuid

from fastapi import Request
from loguru import logger
from starlette.responses import Response

from app.schemas.response.base import BaseResponse

from .exception.common import CommonException
from .request_ctx import (RequestContext, RequestContextualize,
                          _x_request_id_key)

_request_id_key = _x_request_id_key


class RequestContextMiddleware:

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)

        request = Request(scope, receive=receive)

        x_request_id = dict(request.headers).get(_request_id_key)
        logger.info(f"ingress {_request_id_key}: {x_request_id}")
        if not x_request_id:
            x_request_id = x_request_id if x_request_id else uuid.uuid4().hex
            logger.info(f"gen {_request_id_key}: {x_request_id}")

        ctx = RequestContext(request_id=x_request_id)

        with RequestContextualize(ctx):
            try:
                await self.app(scope, receive, send)
                logger.info(f'{request.method} {request.url.path} success')

            except Exception as e:
                logger.error(f'{request.method} {request.url.path} fail {str(e)}')


class ExceptionMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            await self.app(scope, receive, send)

        except CommonException as e:
            logger.exception(e)
            resp = BaseResponse(errcode=e.errcode, errmsg=e.errmsg, detail=e.detail).model_dump_json()
            modified_response = Response(resp, status_code=200, media_type='application/json')
            await modified_response(scope, receive, send)
            raise e

        except Exception as e:
            logger.exception(e)
            err = CommonException.system_error(detail='unknown error')
            resp = BaseResponse(errcode=err.errcode, errmsg=err.errmsg, detail=err.detail).model_dump_json()
            modified_response = Response(resp, status_code=200, media_type='application/json')
            await modified_response(scope, receive, send)
            raise e
