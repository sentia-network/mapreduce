import nest_asyncio
from llama_parse import LlamaParse

from app.settings.config import config


class Llama:
    API_KEY = config.LLAMA_PARSE_KEY

    def __init__(self, api_key: str = None):
        self.api_key = api_key or self.API_KEY

    def parse_pdf(self, file_path: str, result_type="markdown", num_workers=4):
        nest_asyncio.apply()
        llama_parse = LlamaParse(api_key=self.api_key, result_type=result_type, num_workers=num_workers)
        doc = llama_parse.load_data(file_path)
        return doc[0]
