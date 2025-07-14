import httpx
import requests
from loguru import logger
from requests import Response

from app.middlewares.exception.common import CommonException
from app.middlewares.request_ctx import RequestContextualize, _x_request_id_key


class ServiceClient:
    host: str

    def __init__(self, host, additional_headers=None):
        super().__init__()
        self.host = host
        self.additional_headers = additional_headers

    def get(self, endpoint: str, params=None, timeout=15, **kwargs):
        logger.info("=========== service request get ===========")
        logger.info(f"endpoint: {endpoint}")
        logger.info(f"params: {params}")
        response = requests.get(self._api_url(endpoint), headers=self._headers(kwargs.get('headers')), params=params,
                                timeout=timeout, **kwargs)
        return self._parse_response(response)

    def post(self, endpoint: str, body=None, params=None, timeout=15, **kwargs):
        logger.info("=========== service request post ===========")
        logger.info(f"endpoint: {endpoint}")
        response = requests.post(self._api_url(endpoint), headers=self._headers(kwargs.get('headers')),
                                 json=body, params=params, timeout=timeout, **kwargs)
        return self._parse_response(response, params=params, body=body)

    @staticmethod
    def _parse_response(response: Response, params=None, body=None):
        logger.info("=========== service response ===========")
        logger.info(response.status_code)
        if response.status_code != 200 and response.status_code != 201:
            detail = response.content.decode('utf-8')
            logger.info(f"detail: {detail}")
            logger.info(f"params: {params}")
            logger.info(f"body: {body}")
            raise CommonException.service_error(detail)
        if (data := response.json().get('errcode')) != 0:
            raise CommonException.service_error(str(data))
        return response

    async def async_post(self, endpoint: str, body=None, params=None, timeout=300, headers=None):
        async with httpx.AsyncClient() as client:
            logger.info("=========== service async request post ===========")
            logger.info(f"endpoint: {endpoint}")

            response = await client.post(self._api_url(endpoint), json=body, params=params,
                                         headers=self._headers(headers), timeout=timeout)
            return self._parse_response(response, params=params, body=body)

    async def async_get(self, endpoint: str, body=None, params=None, timeout=15, headers=None):
        async with httpx.AsyncClient() as client:
            logger.info("=========== service async request get ===========")
            logger.info(f"endpoint: {endpoint}")

            response = await client.get(self._api_url(endpoint), params=params, headers=self._headers(headers),
                                        timeout=timeout)

            return self._parse_response(response, params=params, body=body)

    async def async_delete(self, endpoint: str, body=None, params=None, timeout=15, headers=None):
        async with httpx.AsyncClient() as client:
            logger.info("=========== service async request delete ===========")
            logger.info(f"endpoint: {endpoint}")
            response = await client.delete(self._api_url(endpoint), params=params, json=body,
                                           headers=self._headers(headers), timeout=timeout)

            return self._parse_response(response, params=params, body=body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        pass

    def _api_url(self, endpoint: str):
        url = f"{self.host}{endpoint}"
        logger.info(url)
        return url

    def _headers(self, headers):
        base = {}
        if x_request_id := RequestContextualize.get_request_id():
            base[_x_request_id_key] = x_request_id

        _additional = self.additional_headers or {}
        _custom = headers or {}
        return base | _additional | _custom
