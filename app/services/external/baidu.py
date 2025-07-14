import json

from aip import AipOcr
from loguru import logger

from app.middlewares.exception.common import CommonException
from app.settings.config import config

'''
对接百度OCR: https://cloud.baidu.com/doc/OCR/index.html
'''


class BaiduOcr:
    appid: str = config.BAIDU_APPID
    apikey: str = config.BAIDU_APIKEY
    secret_key: str = config.BAIDU_SECRET_KEY
    _client: AipOcr = None

    def __init__(self, appid: str = None, apikey: str = None, secret_key: str = None):
        self.appid = appid or self.appid
        self.apikey = apikey or self.apikey
        self.secret_key = secret_key or self.secret_key

    def accurate(self, image: str | bytes, **kwargs):
        if isinstance(image, bytes):
            response = self.client().accurate(image=image, options=kwargs)
        else:
            response = self.client().accurateUrl(url=image, options=kwargs)

        error_code = response.get("error_code", None)
        logger.info("=========== ocr response ===========")
        logger.info(f"ocr ==> error_code: {error_code}")

        if error_code is not None and error_code > 0:
            error_msg = response.get("error_msg", "baidu ocr error")
            detail = f"ocr failed, {error_code}: {error_msg}"
            raise CommonException.system_error(detail=detail)

        logger.info(f"ocr ==> response: {json.dumps(response)}")
        return response

    @staticmethod
    def parse_simple_paragraph(data):
        text = ""
        top = 0
        words_result = data.get("words_result", [])
        for wr in words_result:
            words = wr.get("words")
            location = wr.get("location")
            location_top = location.get('top')
            if len(text) > 0 and location.get('top') > top:
                text = text + "\n"
            text = f"{text}{words}"
            top = location_top

        return text

    def client(self) -> AipOcr:
        if self._client is None:
            self._client = AipOcr(self.appid, self.apikey, self.secret_key)

        return self._client
