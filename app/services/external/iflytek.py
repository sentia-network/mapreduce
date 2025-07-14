import base64
import hashlib
import hmac
import io
import json
import random
import time
import urllib
import uuid

import requests
from loguru import logger

from app.middlewares.exception.common import CommonException
from app.settings.config import config

'''
对接讯飞语音: https://www.xfyun.cn/doc/asr/ifasr_new/API.html
'''


class IflytekSpeech:
    api_host = 'https://raasr.xfyun.cn/v2/api'
    appid: str = config.IFLYTEK_APPID
    secret_key: str = config.IFLYTEK_SECRET_KEY

    def __init__(self, file_bytes: io.BytesIO, suffix: str, appid: str = None, secret_key: str = None):
        self.suffix = suffix
        self.file_bytes = file_bytes
        self.appid = appid or self.appid
        self.secret_key = secret_key or self.secret_key
        self.ts = str(int(time.time()))
        self.signa = self.get_signa()

    def get_signa(self):
        appid = self.appid
        secret_key = self.secret_key
        m2 = hashlib.md5()
        m2.update((appid + self.ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        return signa

    def upload(self):
        param_dict = {'appId': self.appid,
                      'signa': self.signa,
                      'ts': self.ts,
                      "fileSize": self.file_bytes.getbuffer().nbytes,
                      "fileName": f'{uuid.uuid4()}{self.suffix}',
                      "duration": str(random.randint(1, 100))}

        logger.info(f"upload param：{param_dict}")
        data = self.file_bytes.getvalue()

        url = self.api_host + '/upload' + "?" + urllib.parse.urlencode(param_dict)
        logger.info(f"upload_url: {url}")
        response = requests.post(url=url, headers={"Content-type": "application/json"}, data=data)

        if response.status_code != 200 and response.status_code != 201:
            detail = response.content.decode('utf-8')
            logger.info(f"speech transfer ==> detail: {detail}")
            raise CommonException.system_error(detail=detail)

        result = json.loads(response.text)
        logger.info(f"upload resp: {result}")
        return result

    def get_result(self):
        upload_resp = self.upload()
        param_dict = {'appId': self.appid,
                      'signa': self.signa,
                      'ts': self.ts,
                      'orderId': upload_resp['content']['orderId'],
                      'resultType': "transfer,predict"}

        logger.info(f"get result param：{param_dict}")

        status = 3
        result = None
        url = self.api_host + '/getResult' + "?" + urllib.parse.urlencode(param_dict)
        logger.info(f"get_result_url: {url}")

        while status == 3:
            response = requests.post(url=url, headers={"Content-type": "application/json"})

            if response.status_code != 200 and response.status_code != 201:
                detail = response.content.decode('utf-8')
                logger.info(f"speech transfer ==> detail: {detail}")
                raise CommonException.system_error(detail=detail)

            result = json.loads(response.text)

            status = result['content']['orderInfo']['status']
            logger.info(f"status: {status}")

            if status == 4:
                break

            time.sleep(5)
        return json.loads(result.get('content', {}).get('orderResult', {}))

    @staticmethod
    def parse_text(data: dict):
        lattice = data.get("lattice", None)
        lattice = lattice if lattice else data.get("lattice2", None)
        if not lattice:
            return ""

        text = ""
        for item in lattice:
            json_1best = item.get('json_1best')
            st = json.loads(json_1best).get('st')
            rt = st.get('rt', [])
            for i_rt in rt:
                ws = i_rt.get('ws', [])
                for i_ws in ws:
                    cw = i_ws.get('cw', [])
                    for i_cw in cw:
                        w = i_cw.get('w', '')
                        text = text + w

        return text
