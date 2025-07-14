import asyncio
import hashlib
import io
import itertools
import json
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import addict
import aiofiles
import aiohttp
import aiostream
import anyio
import chardet
import httpx
import pymysql
import requests
import tiktoken
from aiohttp import TCPConnector
from charset_normalizer import from_bytes
from fastapi import HTTPException
from langchain.text_splitter import TokenTextSplitter
from loguru import logger
from pypdf import PdfReader

from app.core.openai_adapter import a_embedding_create_openai
from app.settings.config import config


def sql_database_setup():
    sql_type = getattr(config, 'SQL_TYPE').upper()
    db_url = getattr(config, f'DATABASE_{sql_type}_URL')
    logger.info(f'use {sql_type} for data persistence')
    return db_url


async def a_openai_embed(input_text: Union[str, list[str]],
                         # model: str = 'text-embedding-ada-002',
                         model: str = config.OPENAI_MODEL_EMBEDDING_DEFAULT,
                         vector_only=False,
                         return_vector_and_usage=False,
                         openai_api_key: str = "", openai_api_base: str = None):
    """OpenAI向量化：把字符串做成向量"""
    ret = await a_embedding_create_openai(model, input_text,
                                          openai_api_key=openai_api_key,
                                          openai_api_base=openai_api_base)
    if vector_only:
        vectors = [d['embedding'] for d in ret['data']]
        return vectors
    if return_vector_and_usage:
        vectors = [d['embedding'] for d in ret['data']]
        usage = ret['usage']
        usage_ = {k: v or 0 for k, v in usage.items()}
        return vectors, usage_
    return ret


def sum_openai_usages(usages: Union[list[dict], list['OpenAIObjct']]):
    # return {'prompt_tokens': sum([u['prompt_tokens'] or 0 for u in usages]),
    #         'completion_tokens': sum([u['completion_tokens'] or 0 for u in usages]),
    #         'total_tokens': sum([u['total_tokens'] or 0 for u in usages])}
    return {'prompt_tokens': sum([u.get('prompt_tokens', 0) for u in usages]),
            'completion_tokens': sum([u.get('completion_tokens', 0) for u in usages]),
            'total_tokens': sum([u.get('total_tokens', sum([
                u.get('prompt_tokens', 0), u.get('completion_tokens', 0)])) for u in usages])}


def _get_pages_of_chunk(start_idx, end_idx, pages, chunk):
    while chunk not in ' '.join(pages[start_idx: end_idx]):
        end_idx += 1
    while chunk in ' '.join(pages[start_idx + 1: end_idx]):
        start_idx += 1
    return start_idx, end_idx


def _adjust_split(split_idx, encoded, text, encoding, idx_low_bound):
    def _is_good_split(idx, encoded, string, encoding):
        return encoding.decode(encoded[:idx]) + encoding.decode(encoded[idx:]) == string

    while not _is_good_split(split_idx, encoded, text, encoding):
        split_idx -= 1
        if split_idx < idx_low_bound:
            raise IndexError('Can not find suitable cutting point!')
    return split_idx


def split_text_with_page_ref(pages, chunk_size,
                             chunk_overlap,
                             model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)

    text_full = ' '.join(pages)
    encoded_full = encoding.encode(text_full)
    if len(encoded_full) <= chunk_size:
        return [text_full], [[0]]

    start_idx, cur_idx = 0, 0
    chunks, chunk_pages = [], []
    start_page, cur_page = 0, 0
    while start_idx < len(encoded_full):
        cur_idx = start_idx + chunk_size
        cur_idx = _adjust_split(cur_idx, encoded_full, text_full, encoding,
                                start_idx + chunk_size - chunk_overlap)
        chunk = encoding.decode(encoded_full[start_idx: cur_idx])
        chunks.append(chunk)

        start_page, cur_page = _get_pages_of_chunk(start_page, cur_page, pages, chunk)
        chunk_pages.append([*range(start_page + 1, cur_page + 1)])

        if cur_idx > len(encoded_full):
            break

        start_idx = cur_idx - chunk_overlap
        min_start_idx = cur_idx - chunk_overlap * 2
        start_idx = _adjust_split(start_idx, encoded_full, text_full, encoding,
                                  min_start_idx)

    # consistency check
    ranges = [(text_full.index(c), text_full.index(c) + len(c)) for c in chunks]
    is_all_overlap = all([end >= start for (_, end), (start, _) in zip(ranges[:-1], ranges[1:])])
    if not is_all_overlap:
        raise IndexError("Found missing overlap!, text split failed!")
    return chunks, chunk_pages


def make_chunks(iterable, batch_size=10):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


async def a_make_chunks(streamer: aiostream.core.Streamer, batch_size=10):
    """A helper function to break an iterable into chunks of size batch_size."""
    batch = []
    async for item in streamer:
        batch.append(item)
        if len(batch) == batch_size:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def split_text_safe(pages, chunk_size, chunk_overlap):
    text_splitter = TokenTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    try:
        chunks, chunk_pages = split_text_with_page_ref(pages, chunk_size, chunk_overlap)
    except IndexError as e:
        logger.info(f"Split text with page reference has failed: {e}, "
                    f"falling back to split without page reference")
        full_text = ' '.join(pages)
        chunks = text_splitter.split_text(full_text)
        chunk_pages = None
    return chunks, chunk_pages


async def a_split_text_safe(pages, chunk_size, chunk_overlap):
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor()
    chunks, chunk_pages = await loop.run_in_executor(
        executor,
        split_text_safe,
        pages, chunk_size, chunk_overlap
    )
    return chunks, chunk_pages


def num_tokens_from_string(string, model="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages.
    reference:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        logger.warning(
            "Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        logger.warning("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_token_from_string(string: str, model="gpt-3.5-turbo-0301") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("gpt-3.5-turbo-0301")

    num_tokens = len(encoding.encode(string))
    return num_tokens


def openai_msg_from(msg: dict):
    return {'role': msg['role'], 'content': msg['content'],
            **({} if msg['name'] is None else {'name': msg['name']})}


def get_io_buffer(url):
    response = requests.get(url)
    response.raise_for_status()
    io_buffer = io.BytesIO(response.content)
    return io_buffer


async def a_get_io_buffer(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    response.raise_for_status()
    io_buffer = io.BytesIO(response.content)
    return io_buffer


async def a_get_json_from_url(url):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
    response.raise_for_status()

    return response.json()


def get_pdf_reader(pdf_url=None, io_buffer=None):
    if pdf_url is not None:
        response = requests.get(pdf_url)
        response.raise_for_status()
        io_buffer = io.BytesIO(response.content)
        io_buffer.seek(0)
    elif io_buffer is not None:
        io_buffer.seek(0)
    pdf_reader = PdfReader(io_buffer)
    return pdf_reader


async def a_get_pdf_reader(pdf_url=None, io_buffer=None):
    if pdf_url is not None:
        async with httpx.AsyncClient() as client:
            response = await client.get(pdf_url)
        response.raise_for_status()
        io_buffer = io.BytesIO(response.content)
        io_buffer.seek(0)
    elif io_buffer is not None:
        io_buffer.seek(0)
    pdf_reader = PdfReader(io_buffer)
    return pdf_reader


def to_md5_hash(string=None, io_buffer=None):
    hasher = hashlib.new('md5')
    if string is not None:
        hasher.update(string.encode('utf-8'))
    elif io_buffer is not None:
        io_buffer.seek(0)
        hasher.update(io_buffer.getvalue())  # io.BytesIO
    else:
        raise ValueError('At least one input must not be None!')
    return hasher.hexdigest()


def file_path_to_md5_hash(file_path):
    hasher = hashlib.md5()
    with open(file_path, "rb") as file:
        while True:
            data = file.read(8192)
            if not data:
                break
            hasher.update(data)

    return hasher.hexdigest()


async def a_download_as_temp_file(url: str) -> tempfile.NamedTemporaryFile:
    async with aiohttp.ClientSession(connector=TCPConnector(ssl=False)) as session:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise an exception if the request failed
            content = await response.read()

    # Create a temporary file and write the content to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        async with aiofiles.open(temp_file.name, mode='wb') as f:
            await f.write(content)

    return temp_file


async def a_download_as_temp_file_with_extension(url: str, extension: str) -> tempfile.NamedTemporaryFile:
    async with aiohttp.ClientSession(connector=TCPConnector(ssl=False)) as session:
        async with session.get(url) as response:
            response.raise_for_status()  # Raise an exception if the request failed
            content = await response.read()
    # Create a temporary file and write the content to it
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        async with aiofiles.open(temp_file.name, mode='wb') as f:
            if extension.lower() == "txt":
                decoded_obj = from_bytes(content)
                logger.warning(f'start parse file into extension .111........')
                result = decoded_obj.best()
                logger.warning(f'start parse file into extension ..2222.......')
                await f.write(str(result).encode(encoding='utf-8'))
            else:
                await f.write(content)
    return temp_file


async def a_txt_file_to_utf8_encoding(temp_file: tempfile.NamedTemporaryFile) -> tempfile.NamedTemporaryFile:
    content = None
    with open(temp_file.name, 'rb') as f:
        content = f.read()

    # Create a temporary file and write the content to it
    with tempfile.NamedTemporaryFile(delete=False) as new_temp_file:
        async with aiofiles.open(new_temp_file.name, mode='wb') as f:
            decoded_obj = from_bytes(content)
            result = decoded_obj.best()
            await f.write(str(result).encode())
    return new_temp_file


def get_schema(host='localhost', user='root', password='abc123', db='test'):
    connection = pymysql.connect(**{'host': host,
                                    'user': user,
                                    'password': password,
                                    'db': db})
    cursor = connection.cursor()
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
    sql_scripts = ""
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SHOW CREATE TABLE {table_name}")
        create_table = cursor.fetchone()[1]
        sql_scripts += create_table + ";\n\n"

    cursor.close()
    connection.close()

    return sql_scripts


def reset_db(db_name, host='localhost', user='root', password='abc123'):
    connection = pymysql.connect(**{'host': host,
                                    'user': user,
                                    'password': password, })
    cursor = connection.cursor()
    cursor.execute("SHOW databases;")
    dbs = cursor.fetchall()
    if any([db_name in db_n for db_n in dbs]):
        logger.warning(f'found database {db_name}! drop it')
        cursor.execute(f"DROP DATABASE {db_name}")
    time.sleep(1)
    logger.info(f'now create database {db_name}')
    cursor.execute(f"CREATE DATABASE {db_name}")

    cursor.close()
    connection.close()
    logger.info(f'reset done for database {db_name}')


def drop_all_tables(db_name, host, user, password):
    connection = pymysql.connect(host=host, user=user, password=password, db=db_name)

    try:
        with connection.cursor() as cursor:
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()

            for table in tables:
                print(f'dropping {table}')
                cursor.execute(f"DROP TABLE {table[0]}")
            connection.commit()
    finally:
        connection.close()
    print('finished')


def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']


def make_exception(exception, location=''):
    detail = {'type': type(exception).__name__,
              'location': location}
    if hasattr(exception, "args"):
        detail["args"] = exception.args
    if hasattr(exception, "code"):
        detail["code"] = exception.code
    if hasattr(exception, "message"):
        detail["message"] = exception.message
    if hasattr(exception, "filename"):
        detail["filename"] = exception.filename
    if hasattr(exception, "lineno"):
        detail["lineno"] = exception.lineno
    return HTTPException(status_code=500,
                         detail=detail)


async def azip(xs, *xs_lst: list) -> aiostream.stream.combine.zip:
    from collections.abc import AsyncIterable
    xs = aiostream.stream.iterate(xs) if not isinstance(xs, AsyncIterable) else xs
    xs_lst = [aiostream.stream.iterate(xs) if not isinstance(xs, AsyncIterable) else xs
              for xs in xs_lst]
    return aiostream.stream.zip(xs, *xs_lst)


def syncify(func):
    """
    use as a decorator to turn async function sync:
    >>> @syncify
    >>> async def async_function():
    >>>     await asyncio.sleep(1)  # This is an async operation
    >>>     return "Hello, World!"

    >>> print(async_function())  # Outputs: Hello, World!
    """

    def wrapper(*args, **kwargs):
        with ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, func(*args, **kwargs))
            return future.result()

    return wrapper


def asyncify(func):
    """
    use as a decorator to turn sync function async:
    >>>@asyncify
    >>>def blocking_function():
    >>>    time.sleep(1)  # This is a blocking operation
    >>>    return "Hello, World!"

    >>>async def main():
    >>>    result = await blocking_function()
    >>>    print(result)

    >>>anyio.run(main)
    """

    async def wrapper(*args, **kwargs):
        return await anyio.to_thread.run_sync(func, *args, **kwargs)

    return wrapper


def generate_short_id(previous_id: str = None, size=8):
    assert size <= 64, "size must be less than or equal to 64"
    if not previous_id:
        previous_id = str(uuid.uuid4())
    hash_object = hashlib.sha256(previous_id.encode('utf-8'))
    hex_dig = hash_object.hexdigest()
    return hex_dig[:size]


def load_json(file: str):
    with open(file, 'r') as f:
        return addict.Addict(json.load(f))


def load_jsons(content: str):
    return addict.Addict(json.loads(content))


def dump_json(content: dict, file: str, indent=2):
    with open(file, 'w') as f:
        json.dump(content, f, indent=indent, ensure_ascii=False)


def dump_jsons(content: dict, indent=2):
    return json.dumps(content, indent=indent, ensure_ascii=False)


def ordered_unique(lst: list) -> list:
    return list(dict.fromkeys(lst))
