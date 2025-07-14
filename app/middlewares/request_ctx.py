import contextvars

from pydantic import BaseModel, Field

_request_context = contextvars.ContextVar('request_context')

_x_request_id_key = 'x-request-id'


class RequestContext(BaseModel):
    request_id: str | None = Field(None, description='请求ID')


class RequestContextualize:

    def __init__(self, ctx: RequestContext):
        self.ctx = ctx
        self.token = None

    def __enter__(self):
        self.token = _request_context.set(self.ctx)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _request_context.reset(self.token)
        if exc_type:
            raise exc_val

    @classmethod
    def get_ctx(cls) -> RequestContext | None:
        try:
            return _request_context.get()
        except Exception as _:
            return None

    @classmethod
    def get_request_id(cls) -> str | None:
        if ctx := cls.get_ctx():
            return ctx.request_id
        else:
            return None
