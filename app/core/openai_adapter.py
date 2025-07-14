from functools import partial
from typing import Union

# from gptcache.adapter.openai import ChatCompletion as CachedChatCompletion
# from gptcache.core import Cache
# from gptcache.manager.factory import get_data_manager
# from gptcache.processor.pre import all_content
# from openai import ChatCompletion as OpenaiChatCompletion
# from openai.error import (APIConnectionError, APIError, RateLimitError,
#                           ServiceUnavailableError)
from openai import AsyncOpenAI, OpenAI

from app.settings.config import config

# from core.adaptations import gptcache_update_cache_callback_azure_adapted

# CachedChatCompletion._update_cache_callback = gptcache_update_cache_callback_azure_adapted

# logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler()])
# logger = logging.getLogger(__name__)

# MODEL_AZURE_ENGINE_MAP = {'gpt-3.5': env.AZURE_ENGINE}
# MODEL_CACHE_MAP = {}
# LLM_API_KEY_USAGE = {}

openai_client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
a_openai_client = AsyncOpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)

oai_completion = partial(openai_client.chat.completions.create)
oai_embed = partial(openai_client.embeddings.create)
a_oai_completion = partial(a_openai_client.chat.completions.create)
a_oai_embed = partial(a_openai_client.embeddings.create)


# def reset_openai(api_base=env.ORIGIN_OPENAI_API_BASE, proxy=env.ORIGIN_OPENAI_API_PROXY):
#     import asyncio
#
#     import openai
#
#     setattr(asyncio.sslproto._SSLProtocolTransport, "_start_tls_compatible", True)
#     openai.api_key = None
#     openai.api_base = api_base or 'https://api.openai.com/v1'
#     openai.api_version = None
#     openai.api_type = 'open_ai'
#     openai.api_key_path = None
#     openai.proxy = proxy
#     logger.info('\n------- Openai API connection config --------\n'
#                 f'API base: {openai.api_base}\n'
#                 f'Proxy: {openai.proxy}\n---------------------------------------------')


# reset_openai()


# def _setup_api_usage():
#     for key in env.ORIGIN_OPENAI_API_KEYS:
#         LLM_API_KEY_USAGE[key] = 0


# def _setup_gptcache():
#     for model in ['gpt-3', 'gpt-4', 'other']:
#         MODEL_CACHE_MAP[model] = Cache()
#         MODEL_CACHE_MAP[model].init(
#             data_manager=get_data_manager(max_size=2000, data_path=f'data_map_for_{model}.txt'),
#             pre_func=all_content)


# _setup_gptcache()
# _setup_api_usage()


# def _get_model_cache(model_name: str):
#     for key, value in MODEL_CACHE_MAP.items():
#         if model_name.startswith(key):
#             return value
#     return MODEL_CACHE_MAP['other']


# def _unpack_custom_keys(key_store: dict):
#     return (
#         key_store.get('custom_azure_api_key', None),
#         key_store.get('custom_azure_api_base', None),
#         key_store.get('custom_azure_api_version', None),
#         key_store.get('custom_azure_api_type', 'azure'),
#         key_store.get('custom_azure_call_engine', None),
#         key_store.get('custom_openai_api_key', None),
#         key_store.get('custom_openai_api_base', None),
#     )


# def _find_matching_value(dct, input_string):
#     for key, value in dct.items():
#         if input_string.startswith(key):
#             return value
#     return None


# def chat_completion_create(model,
#                            messages,
#                            temperature,
#                            presence_penalty,
#                            frequency_penalty,
#                            max_tokens,
#                            stream,
#                            timeout=None,
#                            skip_cache=False,
#                            **kwargs):
#     (custom_azure_api_key,
#      custom_azure_api_base,
#      custom_azure_api_version,
#      custom_azure_api_type,
#      custom_azure_call_engine,
#      custom_openai_api_key,
#      custom_openai_api_base) = _unpack_custom_keys(kwargs)
#
#     call_param = {'model': model,
#                   'messages': messages,
#                   'temperature': temperature,
#                   'presence_penalty': presence_penalty,
#                   'frequency_penalty': frequency_penalty,
#                   'max_tokens': max_tokens,
#                   'stream': stream,
#                   'api_key': None,
#                   'api_base': None,
#                   'api_type': None,
#                   'api_version': None,
#                   'engine': None,
#                   'skip_cache': skip_cache,
#                   'timeout': timeout}
#
#     if all([custom_azure_api_key, custom_azure_api_base, custom_azure_api_version,
#             custom_azure_call_engine]):
#         logger.info('using custom azure APIs')
#         call_param['api_key'] = custom_azure_api_key
#         call_param['api_base'] = custom_azure_api_base
#         call_param['api_type'] = custom_azure_api_type
#         call_param['api_version'] = custom_azure_api_version
#         call_param['engine'] = custom_azure_call_engine
#
#     if custom_openai_api_key:
#         logger.info('using custom openai key')
#         call_param['api_key'] = custom_openai_api_key
#         call_param['api_base'] = custom_openai_api_base
#     if call_param['api_key']:
#         return _chat_completion_create(**call_param)
#
#     logger.info('No custom key config found, fallback to least used key at default')
#     call_param.update(get_least_used_key_config())
#     try:
#         return _chat_completion_create(**call_param)
#     except (RateLimitError, APIConnectionError, ServiceUnavailableError, APIError) as e:
#         logger.warning(f'Caught {e} when calling openai, retry with another key')
#         call_param.update(get_least_used_key_config(call_param['api_key']))
#         ret = _chat_completion_create(**call_param)
#     return ret

def chat_completion_create(model,
                           messages,
                           temperature,
                           presence_penalty,
                           frequency_penalty,
                           max_tokens,
                           stream,
                           timeout=None,
                           openai_api_key: str = "", openai_api_base: str = None,
                           skip_cache=False,
                           **kwargs):
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        ret = client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                             presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                                             max_tokens=max_tokens, stream=stream, timeout=timeout)
    else:
        ret = oai_completion(model=model, messages=messages, temperature=temperature,
                             presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                             max_tokens=max_tokens, stream=stream, timeout=timeout)
    if stream:
        return ret
    return ret.dict()


# async def a_chat_completion_create(model,
#                                    messages,
#                                    temperature,
#                                    presence_penalty,
#                                    frequency_penalty,
#                                    max_tokens,
#                                    stream,
#                                    timeout=None,
#                                    skip_cache=False,
#                                    **kwargs):
#     (custom_azure_api_key,
#      custom_azure_api_base,
#      custom_azure_api_version,
#      custom_azure_api_type,
#      custom_azure_call_engine,
#      custom_openai_api_key,
#      custom_openai_api_base) = _unpack_custom_keys(kwargs)
#
#     call_param = {'model': model,
#                   'messages': messages,
#                   'temperature': temperature,
#                   'presence_penalty': presence_penalty,
#                   'frequency_penalty': frequency_penalty,
#                   'max_tokens': max_tokens,
#                   'stream': stream,
#                   'api_key': None,
#                   'api_base': None,
#                   'api_type': None,
#                   'api_version': None,
#                   'engine': None,
#                   'skip_cache': skip_cache,
#                   'timeout': timeout}
#
#     if all([custom_azure_api_key, custom_azure_api_base, custom_azure_api_version,
#             custom_azure_call_engine]):
#         logger.info('using custom azure APIs')
#         call_param['api_key'] = custom_azure_api_key
#         call_param['api_base'] = custom_azure_api_base
#         call_param['api_type'] = custom_azure_api_type
#         call_param['api_version'] = custom_azure_api_version
#         call_param['engine'] = custom_azure_call_engine
#
#     if custom_openai_api_key:
#         logger.info('using custom openai key')
#         call_param['api_key'] = custom_openai_api_key
#         call_param['api_base'] = custom_openai_api_base
#
#     if call_param['api_key']:
#         return await _a_chat_completion_create(**call_param)
#
#     logger.info('No custom key config found, fallback to least used key at default')
#     call_param.update(get_least_used_key_config())
#     try:
#         ret = await _a_chat_completion_create(**call_param)
#     except (RateLimitError, APIConnectionError, ServiceUnavailableError, APIError) as e:
#         logger.warning(f'Caught {e} when calling openai, retry with another key')
#         call_param.update(get_least_used_key_config(call_param['api_key']))
#         ret = await _a_chat_completion_create(**call_param)
#     return ret

async def a_chat_completion_create(model,
                                   messages,
                                   temperature,
                                   presence_penalty,
                                   frequency_penalty,
                                   max_tokens,
                                   stream,
                                   timeout=None,
                                   openai_api_key: str = "", openai_api_base: str = None,
                                   skip_cache=False,
                                   **kwargs):
    if openai_api_key:
        client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
        ret = await client.chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                   presence_penalty=presence_penalty,
                                                   frequency_penalty=frequency_penalty,
                                                   max_tokens=max_tokens, stream=stream, timeout=timeout)
    else:
        ret = await a_oai_completion(model=model, messages=messages, temperature=temperature,
                                     presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                                     max_tokens=max_tokens, stream=stream, timeout=timeout)
    if stream:
        return ret
    return ret.dict()


# def _check_call_param(kwargs: dict):
#     key_set = {'model', 'messages', 'temperature', 'presence_penalty', 'frequency_penalty',
#                'max_tokens', 'stream', 'api_key', 'api_base', 'api_type', 'api_version',
#                'engine', 'skip_cache', 'timeout'}
#     assert set(kwargs.keys()) == key_set


# def _chat_completion_create(
#         **kwargs):
#     _check_call_param(kwargs)
#     skip_cache = kwargs.pop('skip_cache')
#     if skip_cache:
#         return OpenaiChatCompletion.create(**kwargs)
#     else:
#         cache: Cache = _get_model_cache(kwargs['model'])
#         pre_embedding_data = cache.pre_embedding_func(kwargs)
#         embedding_data = cache.embedding_func(pre_embedding_data)
#         if cache.data_manager.search(embedding_data):
#             logger.info('Cache hit, returning cached result.')
#         return CachedChatCompletion.create(**kwargs, cache_obj=cache)


# async def _a_chat_completion_create(**kwargs):
#     _check_call_param(kwargs)
#     skip_cache = kwargs.pop('skip_cache')
#     if skip_cache:
#         return await OpenaiChatCompletion.acreate(**kwargs)
#     else:
#         cache: Cache = _get_model_cache(kwargs['model'])
#         pre_embedding_data = cache.pre_embedding_func(kwargs)
#         embedding_data = cache.embedding_func(pre_embedding_data)
#         if cache.data_manager.search(embedding_data):
#             logger.info('Cache hit, returning cached result.')
#         return await CachedChatCompletion.acreate(**kwargs, cache_obj=cache)


# def embedding_create_openai(model: str,
#                             input: Union[str, list[str]],
#                             custom_api_key: Optional[str] = None,
#                             custom_api_base: Optional[str] = None):
#     from openai.api_resources.embedding import Embedding
#
#     # reset_openai()
#
#     api_base = None
#     if custom_api_key:
#         logger.info('Using custom openai key for embedding')
#         api_key = custom_api_key
#         if custom_api_base:
#             api_base = custom_api_base
#
#     else:
#         logger.info('Using least used key for embedding at default')
#         api_key = get_least_used_openai_api_key()
#         api_base = get_openai_api_base()
#
#     return Embedding.create(model=model, input=input, api_key=api_key, api_base=api_base)

def embedding_create_openai(model: str, input: Union[str, list[str]],
                            openai_api_key: str = "", openai_api_base: str = None):
    if openai_api_key:
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        ret = client.embeddings.create(model=model, input=input)
    else:
        ret = oai_embed(model=model, input=input)
    return ret.dict()


# async def a_embedding_create_openai(model: str,
#                                     input: Union[str, list[str]],
#                                     custom_api_key: Optional[str] = None,
#                                     custom_api_base: Optional[str] = None):
#     # reset_openai()
#     from openai.api_resources.embedding import Embedding
#
#     def after_log(retry_state):
#         logger.info(f"Retry {retry_state.attempt_number} ended with: {retry_state.outcome}")
#
#     retry_embed_acreate = AsyncRetrying(
#         stop=stop_after_attempt(1), after=after_log).wraps(Embedding.acreate)
#
#     api_base = None
#     if custom_api_key:
#         logger.info('Using custom openai key for embedding')
#         api_key = custom_api_key
#         if custom_api_base:
#             api_base = custom_api_base
#     else:
#         logger.info('Using least used key for embedding at default')
#         api_key = get_least_used_openai_api_key()
#         api_base = get_openai_api_base()
#
#     # return await Embedding.acreate(model=model, input=input, api_key=api_key, api_base=api_base)
#     return await retry_embed_acreate(model=model, input=input, api_key=api_key, api_base=api_base)

async def a_embedding_create_openai(model: str, input: Union[str, list[str]],
                                    openai_api_key: str = "", openai_api_base: str = None):
    if openai_api_key:
        client = AsyncOpenAI(api_key=openai_api_key, base_url=openai_api_base)
        ret = await client.embeddings.create(model=model, input=input)
    else:
        ret = await a_oai_embed(model=model, input=input)
    return ret.dict()

# def get_least_used_openai_api_key(excluded_key=None):
#     sorted_data = sorted(LLM_API_KEY_USAGE.items(), key=lambda x: x[1])
#     sorted_keys = [data[0] for data in sorted_data]
#
#     if excluded_key is not None:
#         sorted_keys.pop(sorted_keys.index(excluded_key))
#     key = sorted_keys[0]
#     LLM_API_KEY_USAGE[key] += 1
#     logger.info(f"key: {key}, usage: {LLM_API_KEY_USAGE[key]}")
#
#     return key
#
#
# def get_azure_api_config():
#     if env.AZURE_API_KEY not in LLM_API_KEY_USAGE.keys():
#         LLM_API_KEY_USAGE[env.AZURE_API_KEY] = 0
#     return {'api_key': env.AZURE_API_KEY,
#             'api_base': env.AZURE_API_BASE,
#             'api_type': env.AZURE_API_TYPE,
#             'api_version': env.AZURE_API_VERSION,
#             'engine': env.AZURE_ENGINE}
#
#
# def get_least_used_key_config(excluded_key=None):
#     sorted_data = sorted(LLM_API_KEY_USAGE.items(), key=lambda x: x[1])
#     sorted_keys = [data[0] for data in sorted_data]
#     if excluded_key is not None:
#         sorted_keys.pop(sorted_keys.index(excluded_key))
#     key = sorted_keys[0]
#     LLM_API_KEY_USAGE[key] += 1
#
#     logger.info(f"key: {key}, usage: {LLM_API_KEY_USAGE[key]}")
#     if key == env.AZURE_API_KEY:
#         return get_azure_api_config()
#
#     return {'api_key': key,
#             'api_base': get_openai_api_base(),
#             'api_type': None,
#             'api_version': None,
#             'engine': None}
#
#
# def get_openai_api_base():
#     return env.ORIGIN_OPENAI_API_BASE
