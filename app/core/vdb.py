import asyncio
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing.pool import ApplyResult
from typing import Any, Dict, List, Optional, Tuple, Type

from elasticsearch7 import Elasticsearch, helpers
from elasticsearch7.exceptions import NotFoundError as ESNotFoundError
from loguru import logger
from pinecone import FetchResponse
from pinecone import IndexList as PineconeIndexList
from pinecone import ServerlessSpec
from pinecone import Vector as PineconeVector
#from pinecone.data.index import Index as PineconeIndex
from pinecone.db_data.index import Index as PineconeIndex
from pinecone.db_data import IndexAsyncio
from pinecone.grpc import PineconeGRPC as PineconeSync

from app.schemas.public import EmbModelConf
from app.schemas.vector import QueryResponse as VDBQueryResultItem
from app.schemas.vector import (QueryResult, VectorBatch, VectorBatchOptional,
                                VectorOptional, VectorQuery)

try:
    from pinecone import QueryResult as PineconeQueryResult  # v2
except ImportError:
    from pinecone import QueryResponse as PineconeQueryResult  # v3

from pydantic import BaseModel
from tcvectordb import VectorDBClient
from tcvectordb.model.document import Document as TCDocument
from tcvectordb.model.document import Filter as TCFilter
from tcvectordb.model.enum import FieldType as TCFieldType
from tcvectordb.model.enum import IndexType as TCIndexType
from tcvectordb.model.enum import MetricType as TCMetricType
from tcvectordb.model.enum import ReadConsistency
from tcvectordb.model.index import FilterIndex as TCFilterIndex
from tcvectordb.model.index import HNSWParams as TCHNSWParams
from tcvectordb.model.index import Index as TCIndex
from tcvectordb.model.index import VectorIndex as TCVectorIndex

from app.settings.config import config


def get_index_name(emb_conf: EmbModelConf) -> str:
    emb_conf = emb_conf.model_dump(mode='json')
    return f"{emb_conf['model']}_{emb_conf['dim']}{config.VDB_INDEX_SUFFIX}"


def get_model_name(index_name: str) -> str:
    return index_name.split('_')[0]


class VectorDataBaseAbstract(ABC):
    index_name: str

    @abstractmethod
    async def a_upsert(self, vectors: VectorBatch):
        pass

    @abstractmethod
    async def a_query(self,
                      query: VectorQuery,
                      include_metadata: bool = False,
                      include_values: bool = False) -> QueryResult:
        pass

    @abstractmethod
    async def a_fetch(self, ids: List[str], fields: Optional[List[str]] = None) -> VectorBatch:
        pass

    @abstractmethod
    async def a_update(self, vectors: VectorBatchOptional):
        pass

    @abstractmethod
    async def a_delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict] = None):
        pass

    @abstractmethod
    def reset_vdb(self):
        pass

    @classmethod
    @abstractmethod
    async def global_vdb_delete(cls, ids: List[str], model_names: List[str], query_filter: Optional[Dict] = None):
        ...


class GaussES(VectorDataBaseAbstract):
    def __init__(self, index_name: str, dim: int, create=True):
        self.host = config.GAUSS_ES_VDB_HOST
        self.index_name = index_name
        self.vector_dim = dim

        self.client = Elasticsearch(self.host, verify_certs=False, ssl_show_warn=False)
        self.executor = ThreadPoolExecutor()
        if not self.client.indices.exists(index=self.index_name) and create:
            logger.info(f'Index `{self.index_name}` not found, creating...')
            self.create_index()
        elif not self.client.indices.exists(index=self.index_name) and not create:
            raise IOError(f'Index `{self.index_name}` not found!')
        else:
            logger.info(f'Index `{self.index_name}` ready!')

    @staticmethod
    def list_index_names() -> List[str]:
        client = Elasticsearch(config.GAUSS_ES_VDB_HOST, verify_certs=False, ssl_show_warn=False)
        index_names = client.indices.get_alias().keys()
        return [n for n in index_names if not n.startswith('.') and n.endswith(config.VDB_INDEX_SUFFIX)]

    @staticmethod
    def get_config_from_env():
        return {
            'host': config.GAUSS_ES_VDB_HOST,
        }

    def create_index(self):
        index_mapping = {
            "settings": {"index": {"vector": True},
                         "number_of_shards": 3,
                         "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "vector",
                        "dimension": self.vector_dim,
                        "indexing": True,
                        "algorithm": "GRAPH",
                        "metric": "cosine"}}}}
        self.client.indices.create(index=self.index_name, body=index_mapping)
        logger.info(f'Index `{self.index_name}` created!')

    async def a_upsert(self, vector_batch: VectorBatch):
        data = [{'_index': self.index_name, '_id': vector.id,
                 'vector': vector.vector, **vector.metadata}
                for vector in vector_batch.vectors]
        loop = asyncio.get_event_loop()
        upsert = partial(helpers.bulk, self.client, data, request_timeout=3600)
        n, _ = await loop.run_in_executor(self.executor, upsert)
        resp_msg = f'ES bulk upsert done! {n} items upserted!'
        logger.info(resp_msg)

    async def a_update(self, vector_batch: VectorBatchOptional):
        data = [{"_op_type": "update", '_index': self.index_name, '_id': vector.id,
                 'doc': {**({'vector': vector.vector} if vector.vector else {}),
                         **(vector.metadata if vector.metadata else {})}}
                for vector in vector_batch.vectors]
        loop = asyncio.get_event_loop()
        upsert = partial(helpers.bulk, self.client, data, request_timeout=3600)
        n, _ = await loop.run_in_executor(self.executor, upsert)
        resp_msg = f'ES bulk update done! {n} items updated!'
        logger.info(resp_msg)

    async def a_query(self,
                      query: VectorQuery,
                      include_metadata: bool = False,
                      include_values: bool = False) -> QueryResult:
        es_filter = self._process_filter(query.filter)
        top_k, vector = query.top_k, query.vector
        if not es_filter:
            query_body = {
                "size": top_k,
                '_source': self._fill_source(include_metadata, include_values),
                "query": {"vector": {"vector": {"vector": vector, "topk": top_k}}}}
            log_query_body = {
                "size": top_k,
                '_source': self._fill_source(include_metadata, include_values),
                "query": {"vector": {"vector": {"vector": ..., "topk": top_k}}}}
        else:
            query_body = {
                "size": top_k,
                '_source': self._fill_source(include_metadata, include_values),
                "query": {
                    "script_score": {
                        **({"query": {"bool": {"filter": es_filter}} if es_filter else {}}),
                        "script": {"source": "vector_score",
                                   "lang": "vector",
                                   "params": {"field": "vector",
                                              "vector": vector,
                                              "metric": "cosine"}}}}}
            log_query_body = {
                "size": top_k,
                '_source': self._fill_source(include_metadata, include_values),
                "query": {
                    "script_score": {
                        **({"query": {"bool": {"filter": es_filter}} if es_filter else {}}),
                        "script": {"source": "vector_score",
                                   "lang": "vector",
                                   "params": {"field": "vector",
                                              "vector": '...',
                                              "metric": "cosine"}}}}}
        logger.info(f'filter: {filter}, ES query body: {log_query_body}')
        loop = asyncio.get_event_loop()
        query = partial(self.client.search, body=query_body, index=self.index_name)
        res = await loop.run_in_executor(self.executor, query)
        return QueryResult.parse_ga_res(res, include_vector=include_values)

    @staticmethod
    def _process_filter(filter: dict) -> dict:

        def parse_filter(filter: dict, prefix=''):
            if len(filter) > 1:
                return {'bool': {'must': [parse_filter({k: v}, prefix) for k, v in filter.items()]}}

            op_key = list(filter.keys())[0]
            op_val = filter.get(op_key)
            if op_key == '$and':
                return {'bool': {'must': [parse_filter(f, prefix) for f in op_val]}}
            if op_key == '$or':
                return {'bool': {'should': [parse_filter(f, prefix) for f in op_val]}}

            if isinstance(op_val, dict):
                sub_key, sub_val = list(op_val.items())[0]
                if sub_key == '$in':
                    return {'terms': {prefix + op_key: sub_val}}
                if sub_key == '$nin':
                    return {'bool': {'must_not': [{'terms': {prefix + op_key: sub_val}}]}}
                if sub_key == '$ne':
                    return {'bool': {'must_not': [parse_filter({op_key: sub_val}, prefix)]}}
                if sub_key == '$eq':
                    return {'term': {prefix + op_key: sub_val}}

            return {'term': {prefix + op_key: op_val}}

        out_filter = parse_filter(filter) if filter else {}

        if out_filter:
            out_filter = {'bool': {'must': [
                *([out_filter] if out_filter else []),
            ]}}
        return out_filter

    @staticmethod
    def _fill_source(include_metadata: bool, include_values: bool):
        if include_metadata and include_values:
            return True
        if include_metadata:
            return {"excludes": ["vector"]}
        if include_values:
            return "vector"
        return False

    async def a_fetch(self, ids: List[str], fields: Optional[List[str]] = None) -> VectorBatch | VectorBatchOptional:
        loop = asyncio.get_event_loop()
        if fields:
            body = {'docs': [{'_id': id, '_source': fields} for id in ids]}
            mget = partial(self.client.mget, index=self.index_name, body=body)
            resp = await loop.run_in_executor(self.executor, mget)
            return VectorBatchOptional.parse_ga_resp(resp)
        else:
            body = {'ids': ids}
            mget = partial(self.client.mget, index=self.index_name, body=body)
            resp = await loop.run_in_executor(self.executor, mget)
            return VectorBatch.parse_ga_resp(resp)

    async def a_delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict] = None):
        loop = asyncio.get_event_loop()

        if ids:
            def delete():
                for id in ids:
                    try:
                        self.client.delete(index=self.index_name, id=id)
                    except ESNotFoundError:
                        logger.warning(f'ES delete failed! id: `{id}` not found')
                return f'ES delete done! {len(ids)} items deleted'
        else:
            es_filter = self._process_filter(filter)
            if not es_filter:
                raise ValueError('Delete filter cannot be empty!')
            body = {"query": {"bool": {"filter": es_filter}}}
            delete = partial(self.client.delete_by_query, index=self.index_name, body=body)

        await loop.run_in_executor(self.executor, delete)

    @classmethod
    async def global_vdb_delete(cls, ids: List[str], model_names: List[str], query_filter: Optional[Dict] = None):
        index_names = cls.list_index_names()
        vdb_set = {n: cls(n, -1, False) for n in index_names if get_model_name(n) in model_names}
        for vdb in vdb_set.values():
            await vdb.a_delete(ids, query_filter)

    def reset_vdb(self):
        logger.warning(f'Deleting ES vdb index `{self.index_name}`!')
        self.client.indices.delete(index=self.index_name)
        self.create_index()


class PineconeSL(VectorDataBaseAbstract):
    pinecone_api_key = config.PINECONE_API_KEY

    def __init__(self, index_name: str, dim: int, create=True):
        self.pc = PineconeSync(api_key=config.PINECONE_API_KEY)
        self.api_key = config.PINECONE_API_KEY
        self.index_name = index_name
        self.vector_dim = dim
        self.host = None

        legal_index_name = index_name.replace('_', '-')
        has_index = self.pc.has_index(legal_index_name)
        if not has_index and create:
            logger.info(f'Creating index `{index_name}`...')
            self.pc.create_index(name=legal_index_name, dimension=dim, metric="cosine",
                                 spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            logger.info(f'Created pinecone serverless vdb index `{self.index_name}`!')
            resp = self.pc.describe_index(legal_index_name)
            self.host = resp['host']
        elif not has_index and not create:
            raise Exception(f'Index `{self.index_name}` not found!')
        else:
            resp = self.pc.describe_index(legal_index_name)
            self.host = resp['host']

        self.index = PineconeIndex(config.PINECONE_API_KEY, self.host)
        logger.info(f"Index `{self.index_name}` is {resp['status']['state']}!")

    def legal_index_name(self):
        return self.index_name.replace('_', '-')

    @classmethod
    def legal_name(cls, name: str):
        return name.replace('_', '-')

    async def a_upsert(self, vectors: VectorBatch):
        async with IndexAsyncio(self.api_key, self.host) as index:
            resp = await index.upsert(vectors.to_pinecone_vectors(), show_progress=False)
        return f"Pinecone upsert count: {resp.upserted_count}"

    async def a_fetch(self, ids: List[str], fields: Optional[List[str]] = None) -> VectorBatch:
        async with IndexAsyncio(self.api_key, self.host) as index:
            resp = await index.fetch(ids)
        return VectorBatch.from_pinecone_vectors(resp.vectors.values())

    async def a_query(self,
                      query: VectorQuery,
                      include_metadata: bool = False,
                      include_values: bool = False) -> QueryResult:
        async with IndexAsyncio(self.api_key, self.host) as index:
            resp = await index.query(top_k=query.top_k,
                                     vector=query.vector,
                                     filter=query.filter,
                                     include_metadata=include_metadata,
                                     include_values=include_values)
        return QueryResult.parse_pinecone_v7_res(resp)

    async def a_update(self, vector_batch: VectorBatchOptional):
        async with IndexAsyncio(self.api_key, self.host) as index:
            tasks = [index.update(v.id, v.vector, v.metadata) for v in vector_batch.vectors]
            results = await asyncio.gather(*tasks)
        return results

    async def a_delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict] = None):
        async with IndexAsyncio(self.api_key, self.host) as index:
            resp = await index.delete(ids=ids, filter=filter)
        return resp

    def reset_vdb(self):
        self.index.delete(delete_all=True)
        logger.info(f'reset index `{self.index_name}` done!')

    @classmethod
    def list_index_info(cls) -> PineconeIndexList:
        pc = PineconeSync(api_key=config.PINECONE_API_KEY)
        return pc.list_indexes()

    @classmethod
    async def global_vdb_delete(cls, ids: List[str], model_names: List[str], query_filter: Optional[Dict] = None):
        index_info_lst = cls.list_index_info()
        legit_model_names = [cls.legal_name(n) for n in model_names]
        for index_info in index_info_lst:
            if any([n in index_info.name for n in legit_model_names]):
                async with IndexAsyncio(config.PINECONE_API_KEY, index_info.host) as index:
                    await index.delete(ids=ids, filter=query_filter)


def vdb_factory() -> Type[VectorDataBaseAbstract | GaussES | PineconeSL]:
    return {
        "gauss_es": GaussES,
        "pinecone": PineconeSL,
    }[config.VDB_TYPE]


# ###############################
# code below are to be deprecated
# ###############################
class VDBQueryResult(BaseModel):
    results: List[VDBQueryResultItem]

    @classmethod
    def parse_pinecone_v3_res(cls, res: PineconeQueryResult) -> 'VDBQueryResult':
        return cls(results=[VDBQueryResultItem.parse_pinecone_v3_res(m) for m in res.matches])

    @classmethod
    def parse_pinecone_v3_fetch(cls, fetch_resp: FetchResponse) -> 'VDBQueryResult':
        items = fetch_resp.vectors.values()
        return cls(results=[VDBQueryResultItem.parse_pinecone_v3_res(r) for r in items])

    @classmethod
    def parse_pinecone_v2_res(cls, res: dict) -> 'VDBQueryResult':
        return cls(results=[VDBQueryResultItem.parse_pinecone_v2_res(r) for r in res['matches']])

    @classmethod
    def parse_pinecone_v2_fetch(cls, fetch_resp: FetchResponse) -> 'VDBQueryResult':
        res_dict = fetch_resp.to_dict()
        return cls(results=[VDBQueryResultItem.parse_pinecone_v2_res(v | {'score': -1})
                            for v in res_dict['vectors'].values()])

    @classmethod
    def parse_tencent_res(cls, res: List[List[dict]]) -> 'VDBQueryResult':
        res = [i for r in res for i in r]
        return cls(results=[VDBQueryResultItem.parse_tencent_res(r) for r in res])

    @classmethod
    def parse_volcano_res(cls, res: Any) -> 'VDBQueryResult':
        pass

    @classmethod
    def parse_es_res(cls, res: dict, include_vector: bool) -> 'VDBQueryResult':
        res = res['hits']['hits']
        return cls(results=[VDBQueryResultItem.parse_es_res(r, include_vector) for r in res])

    @classmethod
    def parse_ga_res(cls, res: dict, include_vector: bool) -> 'VDBQueryResult':
        hits = res['hits']['hits']
        return cls(results=[VDBQueryResultItem.parse_ga_res(h, include_vector) for h in hits])

    @classmethod
    def parse_ga_fetch(cls, fetch_resp: dict) -> 'VDBQueryResult':
        docs = [d for d in fetch_resp['docs'] if d['found']]
        return cls(results=[VDBQueryResultItem.parse_ga_res(d | {'_score': -1}, include_vector=True)
                            for d in docs])

    def get_ids(self) -> List[str]:
        return [r.id for r in self.results]

    def get_upsert_data(self, exclude: Optional[List[str]] = None) -> List[Tuple[str, list, dict]]:
        exclude = exclude or []
        return [(r.id, r.vector, r.metadata) for r in self.results if r.id not in exclude]

    def __eq__(self, other):
        if not isinstance(other, VDBQueryResult):
            return False
        if {*[r.id for r in self.results]} != {*[r.id for r in other.results]}:
            return False
        other_res_map = {r.id: r for r in other.results}
        try:
            eq = all([res == other_res_map[res.id] for res in self.results])
        except KeyError:
            return False
        return eq


class VDBBase(ABC):

    @abstractmethod
    async def a_upsert(self, vectors: List[Tuple[str, list, dict]],
                       namespace: Optional[str] = config.PINECONE_DEFAULT_NAMESPACE) -> str:
        pass

    @abstractmethod
    async def a_query(self, vector: List[float],
                      top_k: int, namespace: Optional[str] = None,
                      filter: Optional[dict] = None,
                      include_metadata: bool = False,
                      include_values: bool = False) -> VDBQueryResult:
        pass

    @abstractmethod
    async def a_fetch(self, ids: List[str], namespace: Optional[str] = None) -> VDBQueryResult:
        pass

    @abstractmethod
    async def a_update(self, id: str, vector: List[float] = None, metadata: Optional[dict] = None,
                       namespace: Optional[str] = None) -> Any:
        pass

    @abstractmethod
    async def a_delete(self, ids: List[str], namespace: Optional[str] = None) -> Any:
        pass

    @abstractmethod
    def reset_vdb(self) -> Any:
        pass


class GaussESVDB(VDBBase):
    def __init__(self, create=True, custom_index_name=None, **gauss_vdb_config):
        vdb_config_from_env = self.get_config_from_env()
        vdb_config = {**vdb_config_from_env, **gauss_vdb_config}
        self.host = vdb_config['host']
        self.index_name = custom_index_name or vdb_config['index_name']
        self.vector_dim = vdb_config.get('vector_dim', 1536)

        self.client = Elasticsearch(self.host, verify_certs=False, ssl_show_warn=False)
        self.executor = ThreadPoolExecutor()
        if not self.client.indices.exists(index=self.index_name) and create:
            logger.info(f'Index `{self.index_name}` not found, creating...')
            self.create_index()
        elif not create:
            raise IOError(f'Index `{self.index_name}` not found!')
        else:
            logger.info(f'Index `{self.index_name}` ready!')

    @staticmethod
    def get_config_from_env():
        return {
            'host': config.GAUSS_ES_VDB_HOST,
            'index_name': config.GAUSS_ES_VDB_INDEX
        }

    def create_index(self):
        index_mapping = {
            "settings": {"index": {"vector": True},
                         "number_of_shards": 3,
                         "number_of_replicas": 1},
            "mappings": {
                "properties": {
                    "vector": {
                        "type": "vector",
                        "dimension": self.vector_dim,
                        "indexing": True,
                        "algorithm": "GRAPH",
                        "metric": "cosine"}}}}
        self.client.indices.create(index=self.index_name, body=index_mapping)
        logger.info(f'Index `{self.index_name}` created!')

    async def a_upsert(self, vectors: List[Tuple[str, list, dict]],
                       namespace: Optional[str] = config.PINECONE_DEFAULT_NAMESPACE) -> Any:
        data = [{'_index': self.index_name, '_id': id,
                 'vector': vector, **metadata, 'namespace': namespace}
                for id, vector, metadata in vectors]
        loop = asyncio.get_event_loop()
        upsert = partial(helpers.bulk, self.client, data, request_timeout=3600)
        n, _ = await loop.run_in_executor(self.executor, upsert)
        resp_msg = f'ES bulk upsert done! {n} items upserted: {_}'
        return resp_msg

    async def a_update(self, id: str, vector: List[float] = None, metadata: Optional[dict] = None,
                       namespace: Optional[str] = None) -> Any:
        loop = asyncio.get_event_loop()
        update = partial(self.client.update, index=self.index_name, id=id,
                         body={'doc': {**({'vector': vector} if vector else {}),
                                       **(metadata if metadata else {})}})
        resp = await loop.run_in_executor(self.executor, update)
        return resp

    async def a_query(self, vector: List[float],
                      top_k: int, namespace: Optional[str] = None,
                      filter: Optional[dict] = None,
                      include_metadata: bool = False,
                      include_values: bool = False) -> VDBQueryResult:
        es_filter = self._process_filter(filter, namespace=namespace)
        if not es_filter:
            query_body = {
                "size": top_k,
                '_source': self._fill_source(include_metadata, include_values),
                "query": {"vector": {"vector": {"vector": vector, "topk": top_k}}}}
            log_query_body = {
                "size": top_k,
                '_source': self._fill_source(include_metadata, include_values),
                "query": {"vector": {"vector": {"vector": ..., "topk": top_k}}}}
        else:
            query_body = {
                "size": top_k,
                '_source': self._fill_source(include_metadata, include_values),
                "query": {
                    "script_score": {
                        **({"query": {"bool": {"filter": es_filter}} if es_filter else {}}),
                        "script": {"source": "vector_score",
                                   "lang": "vector",
                                   "params": {"field": "vector",
                                              "vector": vector,
                                              "metric": "cosine"}}}}}
            log_query_body = {
                "size": top_k,
                '_source': self._fill_source(include_metadata, include_values),
                "query": {
                    "script_score": {
                        **({"query": {"bool": {"filter": es_filter}} if es_filter else {}}),
                        "script": {"source": "vector_score",
                                   "lang": "vector",
                                   "params": {"field": "vector",
                                              "vector": '...',
                                              "metric": "cosine"}}}}}
        logger.info(f'filter: {filter}, ES query body: {log_query_body}')
        loop = asyncio.get_event_loop()
        query = partial(self.client.search, body=query_body, index=self.index_name)
        res = await loop.run_in_executor(self.executor, query)
        return VDBQueryResult.parse_ga_res(res, include_vector=include_values)

    async def a_fetch(self, ids: List[str], namespace: Optional[str] = None) -> VDBQueryResult:
        loop = asyncio.get_event_loop()
        mget = partial(self.client.mget, index=self.index_name, body={'ids': ids})
        res = await loop.run_in_executor(self.executor, mget)
        return VDBQueryResult.parse_ga_fetch(res)

    @staticmethod
    def _process_filter(filter: dict, namespace: Optional[str] = None) -> dict:

        def parse_filter(filter: dict, prefix=''):
            if len(filter) > 1:
                return {'bool': {'must': [parse_filter({k: v}, prefix) for k, v in filter.items()]}}

            op_key = list(filter.keys())[0]
            op_val = filter.get(op_key)
            if op_key == '$and':
                return {'bool': {'must': [parse_filter(f, prefix) for f in op_val]}}
            if op_key == '$or':
                return {'bool': {'should': [parse_filter(f, prefix) for f in op_val]}}

            if isinstance(op_val, dict):
                sub_key, sub_val = list(op_val.items())[0]
                if sub_key == '$in':
                    return {'terms': {prefix + op_key: sub_val}}
                if sub_key == '$nin':
                    return {'bool': {'must_not': [{'terms': {prefix + op_key: sub_val}}]}}
                if sub_key == '$ne':
                    return {'bool': {'must_not': [parse_filter({op_key: sub_val}, prefix)]}}
                if sub_key == '$eq':
                    return {'term': {prefix + op_key: sub_val}}

            return {'term': {prefix + op_key: op_val}}

        out_filter = parse_filter(filter) if filter else {}

        if out_filter or namespace:
            out_filter = {'bool': {'must': [
                *([out_filter] if out_filter else []),
                *([{"term": {"namespace": namespace}}] if namespace else [])
            ]}}
        return out_filter

    @staticmethod
    def _fill_source(include_metadata: bool, include_values: bool):
        if include_metadata and include_values:
            return True
        if include_metadata:
            return {"excludes": ["vector"]}
        if include_values:
            return "vector"
        return False

    async def a_delete(self, ids: List[str],
                       namespace: Optional[str] = config.PINECONE_DEFAULT_NAMESPACE) -> Any:
        loop = asyncio.get_event_loop()

        def delete():
            for id in ids:
                try:
                    self.client.delete(self.index_name, id)
                except ESNotFoundError:
                    logger.warning(f'ES delete failed! id: `{id}` not found')
            return f'ES delete done! {len(ids)} items deleted'

        return await loop.run_in_executor(self.executor, delete)

    def reset_vdb(self) -> Any:
        logger.warning(f'Deleting ES vdb index `{self.index_name}`!')
        self.client.indices.delete(index=self.index_name)
        self.create_index()

    async def to_json(self, json_file):
        from pinecone import Vector
        from tenacity import Retrying, stop_after_attempt, wait_exponential
        from tortoise import Tortoise

        from app.models import Segment
        from app.settings.config import TORTOISE_ORM

        def after_log(retry_state):
            logger.warning(f"Retry {retry_state.attempt_number} ended with: {retry_state.outcome}")

        logger.info('initiating tortoise ...')
        await Tortoise.init(config=TORTOISE_ORM)

        logger.info('initiating pinecone ...')
        pinecone_vdb = PineconeV3()
        pinecone_index = pinecone_vdb.get_index()

        pinecone_fetch = Retrying(
            stop=stop_after_attempt(5),
            wait=wait_exponential(multiplier=1, max=20, min=5),
            after=after_log,
        ).wraps(pinecone_index.fetch)

        batch = await Segment.all().values('id')
        all_ids = [str(b['id']) for b in batch]
        logger.info(f'total record: {len(all_ids)}')

        batch_size = 10
        data = []
        for i in range(len(all_ids) // batch_size + 1):
            logger.info(f'loading batch {i + 1}/{len(all_ids) // batch_size + 1} ...')
            ids = all_ids[i * batch_size: (i + 1) * batch_size]
            vector_map: dict[str, Vector] = pinecone_fetch(
                ids, namespace=config.PINECONE_DEFAULT_NAMESPACE)
            data.extend([(id, vector.values, vector.metadata)
                         for id, vector in vector_map.items()])

        json.dump(data, open(json_file, 'w'))
        logger.info(f'json dumped to {json_file} with {len(data)} '
                    f'vectors of totally {len(all_ids)}.')


class VolcanoESVDB(VDBBase):
    def __init__(self, create=True, custom_index_name=None):
        self.client = Elasticsearch(config.VOLC_ES_VDB_HOST, verify_certs=False, ssl_show_warn=False)
        self.index_name = custom_index_name or config.VOLC_ES_VDB_INDEX
        self.executor = ThreadPoolExecutor()
        if not self.client.indices.exists(index=self.index_name) and create:
            logger.info(f'Index `{self.index_name}` not found, creating...')
            self.create_index()
        elif not create:
            raise IOError(f'Index `{self.index_name}` not found!')
        else:
            logger.info(f'Index `{self.index_name}` ready!')

    def create_index(self):
        self.client.indices.create(
            index=self.index_name,
            body={"settings": {"index.knn": True,
                               "index.knn.space_type": "cosinesimil",
                               "index.number_of_shards": 3,
                               "index.refresh_interval": "10s"},
                  "mappings": {"dynamic": "true",
                               "properties": {
                                   "vector": {
                                       "type": "knn_vector",
                                       "dimension": 1536},
                                   "namespace": {
                                       "type": "text"},
                                   "id": {"type": "keyword"},
                                   "metadata": {"type": "object"}}}
                  }
        )
        logger.info(f'Index `{self.index_name}` created!')

    async def a_upsert(self, vectors: List[Tuple[str, list, dict]],
                       namespace: Optional[str] = config.PINECONE_DEFAULT_NAMESPACE) -> Any:
        data_source = [{'id': id, 'vector': vector, 'metadata': metadata, 'namespace': namespace}
                       for id, vector, metadata in vectors]
        data_extra = [{"index": {"_index": self.index_name, "_id": d['id']}}
                      for d in data_source]
        data = [d for pair in zip(data_extra, data_source) for d in pair]
        loop = asyncio.get_event_loop()
        upsert = partial(self.client.bulk, data, index=self.index_name)
        resp = await loop.run_in_executor(self.executor, upsert)
        if resp['errors']:
            logger.error(f'ES bulk upsert failed! details: '
                         f'{[i for i in resp["items"] if not str(i["index"]["status"]).startswith("2")]}\n'
                         f'Undoing upsert...')
            await self.a_delete([d['id'] for d in data_source], namespace=namespace)
            logger.error('Undoing upsert done!')
            raise IOError('ES bulk upsert failed!')
        resp_msg = f'ES bulk upsert done! {len(data_source)} items upserted'
        return resp_msg

    async def a_update(self, id: str, vector: List[float] = None, metadata: Optional[dict] = None,
                       namespace: Optional[str] = None) -> Any:
        loop = asyncio.get_event_loop()
        update = partial(self.client.update, index=self.index_name, id=id,
                         body={"doc": {**({"vector": vector} if vector else {}),
                                       **({"metadata": metadata} if metadata else {})}})
        resp = await loop.run_in_executor(self.executor, update)
        return resp

    async def a_fetch(self, ids: List[str], namespace: Optional[str] = None) -> VDBQueryResult:
        raise NotImplementedError

    async def a_query(self, vector: List[float],
                      top_k: int,
                      namespace: Optional[str] = None,
                      filter: Optional[dict] = None,
                      include_metadata: bool = False,
                      include_values: bool = False) -> VDBQueryResult:
        es_filter = self._process_filter(filter, prefix='metadata.')
        if es_filter or namespace:
            es_filter = {'bool': {'must': [
                *([es_filter] if es_filter else []),
                *([{"term": {"namespace": namespace}}] if namespace else [])
            ]}}
            query_body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        **({"query": es_filter} if es_filter else {}),
                        "script": {"source": "knn_score",
                                   "lang": "knn",
                                   "params": {"field": "vector",
                                              "query_value": vector,
                                              "space_type": "cosinesimil"}}}}}
            log_query_body = {
                "size": top_k,
                "query": {
                    "script_score": {
                        **({"query": es_filter} if es_filter else {}),
                        "script": {"source": "knn_score",
                                   "lang": "knn",
                                   "params": {"field": "vector",
                                              "query_value": '...',
                                              "space_type": "cosinesimil"}}}}}
        else:
            query_body = {
                "size": top_k,
                "query": {"knn": {"vector": {"vector": vector, "k": top_k}}}}
            log_query_body = {
                "size": top_k,
                "query": {"knn": {"vector": {"vector": '...', "k": top_k}}}}
        # query_body = {
        #     "size": top_k,
        #     "query": {
        #         "bool": {
        #             **({"filter": es_filter} if es_filter else {}),
        #             "must": [{"knn": {"vector": {"vector": vector, "k": top_k}}}],
        #         }}}
        # log_query_body = {
        #     "size": top_k,
        #     "query": {
        #         "bool": {
        #             **({"filter": es_filter} if es_filter else {}),
        #             "must": [{"knn": {"vector": {"vector": '...', "k": top_k}}}],
        #         }}}
        logger.info(f'parse filter dict: {filter} to ES query: {log_query_body}')

        loop = asyncio.get_event_loop()
        query = partial(self.client.search, body=query_body, index=self.index_name)
        res = await loop.run_in_executor(self.executor, query)
        return VDBQueryResult.parse_es_res(res, include_vector=include_values)

    @staticmethod
    def _process_filter(filter: dict, prefix='') -> dict:

        def parse_filter(filter: dict, prefix=''):
            if len(filter) > 1:
                return {'bool': {'must': [parse_filter({k: v}, prefix) for k, v in filter.items()]}}

            op_key = list(filter.keys())[0]
            op_val = filter.get(op_key)
            if op_key == '$and':
                return {'bool': {'must': [parse_filter(f, prefix) for f in op_val]}}
            if op_key == '$or':
                return {'bool': {'should': [parse_filter(f, prefix) for f in op_val]}}

            if isinstance(op_val, dict):
                sub_key, sub_val = list(op_val.items())[0]
                if sub_key == '$in':
                    return {'terms': {prefix + op_key: sub_val}}
                if sub_key == '$nin':
                    return {'bool': {'must_not': [{'terms': {prefix + op_key: sub_val}}]}}
                if sub_key == '$ne':
                    return {'bool': {'must_not': [parse_filter({op_key: sub_val}, prefix)]}}
                if sub_key == '$eq':
                    return {'term': {prefix + op_key: sub_val}}

            return {'term': {prefix + op_key: op_val}}

        return parse_filter(filter, prefix) if filter else {}

    async def a_delete(self, ids: List[str],
                       namespace: Optional[str] = config.PINECONE_DEFAULT_NAMESPACE) -> Any:
        loop = asyncio.get_event_loop()

        def delete():
            for id in ids:
                try:
                    self.client.delete(self.index_name, id)
                except ESNotFoundError:
                    logger.warning(f'ES delete failed! id: `{id}` not found')
            return f'ES delete done! {len(ids)} items deleted'

        return await loop.run_in_executor(self.executor, delete)

    def reset_vdb(self) -> Any:
        logger.warning(f'Deleting ES vdb index `{self.index_name}`!')
        self.client.indices.delete(index=self.index_name)
        self.create_index()


class TencentVDB(VDBBase):
    # reference: https://cloud.tencent.com/document/product/1709/95826
    api_url = config.TENCENT_VDB_API_URL
    api_key = config.TENCENT_VDB_API_KEY
    api_username = config.TENCENT_VDB_API_USERNAME
    db_name = config.TENCENT_VDB_DB_NAME
    collection_name = config.TENCENT_VDB_COLLECTION_NAME
    db = None
    collection = None

    def __init__(self, create=True, custom_collection_name: Optional[str] = None):
        # tcvectordb.debug.DebugEnable = False
        self.collection_name = custom_collection_name or self.collection_name
        self.executor = ThreadPoolExecutor()
        self.client = VectorDBClient(
            url=self.api_url,
            username=self.api_username,
            key=self.api_key,
            read_consistency=ReadConsistency.EVENTUAL_CONSISTENCY,
            timeout=30)
        self.db = self.get_db(create=create)
        self.collection = self.get_collection(create=create)
        logger.info(f'======== initiating tencent vdb ========\n'
                    f'api_url: {self.api_url}\n'
                    f'db_name: {self.db_name}\n'
                    f'collection_name: {self.collection_name}\n'
                    f'========================================')

    def get_collection(self, create: bool = True):
        coll_names = [coll.collection_name for coll in self.db.list_collections()]
        if self.collection_name not in coll_names and create:
            logger.info(f'Collection `{self.collection_name}` not found, creating...')
            index = TCIndex().add(
                TCVectorIndex(
                    'vector',
                    1536,
                    TCIndexType.HNSW,
                    TCMetricType.COSINE,
                    TCHNSWParams(m=16, efconstruction=200))
            ).add(
                TCFilterIndex(
                    'id',
                    TCFieldType.String,
                    TCIndexType.PRIMARY_KEY),
            ).add(
                TCFilterIndex(
                    config.CHUNK_METADATA_PINECONE_HASH,
                    TCFieldType.String,
                    TCIndexType.FILTER),
            ).add(
                TCFilterIndex(
                    config.CHUNK_METADATA_PINECONE_ENABLE,
                    TCFieldType.String,
                    TCIndexType.FILTER),
            ).add(
                TCFilterIndex(
                    'tenant_id',
                    TCFieldType.String,
                    TCIndexType.FILTER),
            )

            self.db.create_collection(name=self.collection_name,
                                      shard=3,
                                      replicas=2,
                                      description='',
                                      index=index)
        return self.db.collection(self.collection_name)

    def get_db(self, create: bool = True):
        db_names = [db.database_name for db in self.client.list_databases()]
        if self.db_name not in db_names and create:
            logger.info(f'Database `{self.db_name}` not found, creating...')
            self.client.create_database(self.db_name)
        return self.client.database(self.db_name)

    async def a_fetch(self, ids: List[str], namespace: Optional[str] = None) -> VDBQueryResult:
        raise NotImplementedError

    async def a_upsert(self, vectors: List[Tuple[str, list, dict]], namespace: Optional[str] = None):
        doc_list = [TCDocument(id=id, vector=vector, **self.process_metadata(metadata))
                    for id, vector, metadata in vectors]
        loop = asyncio.get_event_loop()
        upsert = partial(self.collection.upsert, doc_list)
        await loop.run_in_executor(self.executor, upsert)
        return f'Tencent vdb upsert done! {len(doc_list)} items upserted'

    @staticmethod
    def process_metadata(metadata: dict) -> dict:
        metadata = {k: json.dumps(v) if not type(v) in [int, float, str] else v
                    for k, v in metadata.items()}
        return metadata

    async def a_query(self, vector: List[float], top_k: int, namespace: Optional[str] = None,
                      filter: Optional[dict] = None, include_metadata: bool = False,
                      include_values: bool = False) -> VDBQueryResult:
        loop = asyncio.get_event_loop()
        query = partial(self.collection.search, vectors=[vector],
                        filter=self._process_filter(filter),
                        retrieve_vector=include_values, limit=top_k)
        res = await loop.run_in_executor(self.executor, query)
        return VDBQueryResult.parse_tencent_res(res)

    @staticmethod
    def _process_filter(filter: dict) -> Optional[TCFilter]:
        # reference: https://cloud.tencent.com/document/product/1709/95099
        def parse_filter(filter: dict) -> Optional[TCFilter]:
            op_key = list(filter.keys())[0]
            op_val = filter.get(op_key)
            if len(filter) > 1:
                head, *tail = list(filter.items())
                return parse_filter(dict([head])).And(parse_filter(dict(tail)).cond)
            if op_key in ['$and', '$or']:
                filter_op = {'$and': 'And', '$or': 'Or'}[op_key]
                head, *tail = filter[op_key]
                if not tail:
                    return parse_filter(head)
                head_filter = parse_filter(head)
                tail_filter = parse_filter({op_key: tail})
                if '$and' in head or '$or' in head:
                    head_filter._cond = f'({head_filter.cond})'
                    tail_filter._cond = f'({tail_filter.cond})'
                return getattr(head_filter, filter_op)(tail_filter.cond)

            if isinstance(op_val, dict):
                sub_key, sub_val = list(op_val.items())[0]
                if type(sub_val) is bool:
                    sub_val = {True: 'true', False: 'false'}[sub_val]
                if type(sub_val) is str:
                    sub_val = f'"{sub_val}"'
                if '$in' in op_val:
                    return TCFilter(TCFilter.In(op_key, sub_val))
                if '$nin' in op_val:
                    cond = TCFilter.In(op_key, sub_val)
                    return TCFilter(f'not {cond}')
                if '$eq' in op_val:
                    return TCFilter(f'{op_key}={sub_val}')
                if '$ne' in op_val:
                    return TCFilter(f'{op_key}!={sub_val}')
            if any([k.startswith('$') for k in filter.keys()]):
                raise NotImplementedError(f'Found unsupported filter key(s): {list(filter.keys())}')
            if isinstance(op_val, str):
                return TCFilter(f'{op_key}="{op_val}"')
            if isinstance(op_val, bool):
                op_val_ = {True: 'true', False: 'false'}[op_val]
                return TCFilter(f'{op_key}={op_val_}')
            if isinstance(op_val, (int, float)):
                return TCFilter(f'{op_key}={op_val}')

            raise NotImplementedError(f'No parse branch for filter: {filter}')

        if filter:
            tc_filter = parse_filter(filter)
            logger.info(f'parsed filter: {tc_filter.cond} from {filter}')
            return tc_filter
        return None

    async def a_update(self, id: str, vector: Optional[List[float]] = None,
                       metadata: Optional[dict] = None, namespace: Optional[str] = None):
        raise NotImplementedError

    async def a_delete(self, ids: List[str], namespace: Optional[str] = None):
        loop = asyncio.get_event_loop()
        delete = partial(self.collection.delete, ids)
        return await loop.run_in_executor(self.executor, delete)

    def reset_vdb(self):
        logger.warning(f'Dropping tencent vdb collection `{self.collection_name}`!')
        self.db.drop_collection(self.collection_name)
        self.collection = self.get_collection(create=True)
        logger.info(f'Reset tencent vdb collection `{self.collection_name}` done!')


class PineconeV2(VDBBase):
    def __init__(self, **pinecone_config):
        import pinecone
        from pinecone.core.client.configuration import \
            Configuration as OpenApiConfiguration

        pinecone_config_from_env = self.get_config_from_env(self)
        pinecone_config = {**pinecone_config_from_env, **pinecone_config}
        self.pinecone_env = pinecone_config.get('pinecone_env', None)
        self.pinecone_api_key = pinecone_config.get('pinecone_api_key', None)
        self.pinecone_index_name = pinecone_config.get('pinecone_index_name', None)
        self.pinecone_proxy = pinecone_config.get('pinecone_proxy', None)
        self.pinecone_host = pinecone_config.get('pinecone_host', None)
        self.config_suffix = pinecone_config.get('config_suffix', None)

        self.executor = ThreadPoolExecutor()
        self.index = None

        logger.info('\n------ pinecone API connection config -------\n'
                    f'Host: {self.pinecone_host}\n'
                    f'Proxy: {self.pinecone_proxy}\n'
                    f'---------------------------------------------')
        openapi_config = None
        if self.pinecone_proxy:
            openapi_config = OpenApiConfiguration.get_default_copy()
            openapi_config.proxy = self.pinecone_proxy

        try:
            logger.info(f'initiating pinecone under {self.config_suffix} setting ...')
            pinecone.init(api_key=self.pinecone_api_key,
                          environment=self.pinecone_env,
                          host=self.pinecone_host,
                          openapi_config=openapi_config)
            logger.info(f'pinecone index list: {pinecone.list_indexes()}')
        except Exception as e:
            logger.error(f'Pinecone initiation failed! details: {e}')
        logger.info(f'using `{self.pinecone_index_name}` as pinecone default index name')

    @staticmethod
    def get_config_from_env(self):
        config_suffix = config.PINECONE_CONFIG_TYPE.upper()
        pinecone_env = getattr(config, f"PINECONE_ENV_{config_suffix}", None)
        pinecone_api_key = getattr(config, f"PINECONE_API_KEY_{config_suffix}", None)
        pinecone_index_name = getattr(config, f'PINECONE_INDEX_NAME_{config_suffix}', None)
        pinecone_proxy = getattr(config, 'PINECONE_PROXY', None)
        pinecone_host = getattr(config, 'PINECONE_HOST', None)
        pinecone_host = pinecone_host and pinecone_host.format(pinecone_env=self.pinecone_env)

        return {'pinecone_env': pinecone_env,
                'pinecone_api_key': pinecone_api_key,
                'pinecone_index_name': pinecone_index_name,
                'pinecone_proxy': pinecone_proxy,
                'pinecone_host': pinecone_host,
                'config_suffix': config_suffix}

    def get_index(self):
        import pinecone
        if self.index:
            return self.index

        try:
            index = pinecone.Index(self.pinecone_index_name, pool_threads=1)
            logger.info(f'Pinecone index stats: {index.describe_index_stats()}')
            self.index = index
            return self.index
        except Exception as e:
            logger.error(f'Failed to get pinecone stats! details: {e}')

    async def a_fetch(self, ids: List[str], namespace: Optional[str] = None) -> VDBQueryResult:
        index = self.get_index()
        loop = asyncio.get_event_loop()
        get = partial(index.fetch, ids=ids, namespace=namespace)
        fetch_resp = await loop.run_in_executor(self.executor, get)
        return VDBQueryResult.parse_pinecone_v2_fetch(fetch_resp)

    async def a_upsert(self, vectors, namespace=None) -> Any:
        index = self.get_index()
        loop = asyncio.get_event_loop()
        upsert = partial(index.upsert, vectors, namespace=namespace)
        resp = await loop.run_in_executor(self.executor, upsert)
        return f"Pinecone upsert count: {resp.upserted_count}"

    async def a_query(self, vector, top_k, namespace=None, filter=None, include_metadata=False, include_values=False):
        index = self.get_index()
        loop = asyncio.get_event_loop()
        query = partial(index.query, vector=vector, top_k=top_k,
                        **({'namespace': namespace} if namespace else {}),
                        **({'include_metadata': include_metadata} if include_metadata else {}),
                        **({'include_values': include_values} if include_values else {}),
                        **({'filter': filter} if filter else {}))

        res = await loop.run_in_executor(self.executor, query)
        return VDBQueryResult.parse_pinecone_v2_res(res)

    async def a_update(self, id, vector=None, metadata=None, namespace=None):
        index = self.get_index()
        loop = asyncio.get_event_loop()
        update = partial(index.update, id=id, values=vector, set_metadata=metadata,
                         namespace=namespace or config.PINECONE_DEFAULT_NAMESPACE)
        return await loop.run_in_executor(self.executor, update)

    async def a_delete(self, ids, namespace=None):
        index = self.get_index()
        loop = asyncio.get_event_loop()
        delete = partial(index.delete, ids=ids, namespace=namespace)
        return await loop.run_in_executor(self.executor, delete)

    def remove_pinecone_namespace(self, namespace: str = ''):
        index = self.get_index()
        logger.info(f'removing `{namespace}` from pinecone for index<{self.pinecone_index_name}>')
        index.delete(delete_all=True, namespace=namespace)

    def reset_vdb(self):
        index = self.get_index()
        namespace_info = index.describe_index_stats().to_dict().get('namespaces', {})
        logger.info(f"""current namespaces: {
        '; '.join([f"`{n}` ({c.get('vector_count', 0)} vectors)"
                   for n, c in namespace_info.items()])
        }""")
        for n in namespace_info.keys():
            self.remove_pinecone_namespace(n)
        logger.info('reset pinecone done!')


class PineconeV3(VDBBase):
    config_suffix = config.PINECONE_CONFIG_TYPE.upper()
    pinecone_env = getattr(config, f"PINECONE_ENV_{config_suffix}", None)
    pinecone_api_key = getattr(config, f"PINECONE_API_KEY_{config_suffix}", None)
    pinecone_index_name = getattr(config, f'PINECONE_INDEX_NAME_{config_suffix}', None)

    def __init__(self):
        from pinecone import Pinecone
        self.client = Pinecone(api_key=self.pinecone_api_key)
        self.index = None
        self.executor = ThreadPoolExecutor()

    def get_index(self):
        if not self.index:
            self.index = self.client.Index(self.pinecone_index_name)
        return self.index

    async def a_upsert(self, vectors: List[Tuple[str, list, dict]],
                       namespace: Optional[str] = config.PINECONE_DEFAULT_NAMESPACE):
        index = self.get_index()
        resp = await index.upsert(vectors, async_req=True, namespace=namespace)
        return f"Pinecone upsert count: {resp.upserted_count}"

    async def a_fetch(self, ids: List[str], namespace: Optional[str] = config.PINECONE_DEFAULT_NAMESPACE
                      ) -> VDBQueryResult:
        index = self.get_index()
        loop = asyncio.get_running_loop()
        to_get: ApplyResult = index.fetch(ids, async_req=True, namespace=namespace)
        with ThreadPoolExecutor() as executor:
            resp = await loop.run_in_executor(executor, to_get.get)
        return VDBQueryResult.parse_pinecone_v3_fetch(resp)

    async def a_query(self, vector: List[float],
                      top_k: int, namespace: Optional[str] = None, filter: Optional[dict] = None,
                      include_metadata: bool = False, include_values: bool = False) -> VDBQueryResult:
        index = self.get_index()
        loop = asyncio.get_event_loop()
        query = partial(index.query, vector=vector, top_k=top_k,
                        **({'namespace': namespace} if namespace else {}),
                        **({'include_metadata': include_metadata} if include_metadata else {}),
                        **({'include_values': include_values} if include_values else {}),
                        **({'filter': filter} if filter else {}))

        res = await loop.run_in_executor(self.executor, query)
        return VDBQueryResult.parse_pinecone_v3_res(res)

    async def a_update(self, id, vector: List[float] = None,
                       metadata: dict = None, namespace=None):
        index = self.get_index()
        loop = asyncio.get_event_loop()
        update = partial(index.update, id=id, values=vector, set_metadata=metadata,
                         namespace=namespace or config.PINECONE_DEFAULT_NAMESPACE)
        return await loop.run_in_executor(self.executor, update)

    async def a_delete(self, ids: List[str], namespace: Optional[str] = None):
        index = self.get_index()
        loop = asyncio.get_event_loop()
        delete = partial(index.delete, ids=ids, namespace=namespace)
        return await loop.run_in_executor(self.executor, delete)

    def remove_pinecone_namespace(self, namespace: str = ''):
        index = self.get_index()
        logger.info(f'removing `{namespace}` from pinecone for index<{self.pinecone_index_name}>')
        index.delete_all(namespace=namespace)

    def reset_vdb(self):
        index = self.get_index()
        namespace_info = index.describe_index_stats().to_dict().get('namespaces', {})
        logger.info(f"""current namespaces: {
        '; '.join([f"`{n}` ({c.get('vector_count', 0)} vectors)"
                   for n, c in namespace_info.items()])
        }""")
        for n in namespace_info.keys():
            self.remove_pinecone_namespace(n)
        logger.info('reset pinecone done!')


def setup_pinecone():
    import subprocess
    pip_process = subprocess.Popen(["pip", "list"], stdout=subprocess.PIPE)
    grep_process = subprocess.Popen(["grep", "pinecone-client"], stdin=pip_process.stdout, stdout=subprocess.PIPE)
    pip_process.stdout.close()
    output = grep_process.communicate()[0].decode('utf-8')
    version_str = output.strip().split(' ')[-1]
    logger.info(f'pinecone-client version: {version_str}')
    version = 2 if version_str.startswith('2') else 3

    if version == 2:
        logger.info('using pinecone-client v2')
        vdb = PineconeV2()
    else:
        logger.info('using pinecone-client v3')
        vdb = PineconeV3()

    return vdb


def setup_vdb() -> VDBBase:
    vdb_cls = {'pinecone': setup_pinecone,
               'tencent': TencentVDB,
               'vol_es': VolcanoESVDB,
               'gauss_es': GaussESVDB
               }.get(config.VDB_TYPE)
    logger.info(f'Initiating {vdb_cls.__name__} ...')
    vdb = vdb_cls()
    return vdb

# vdb = setup_vdb()

# if env.RESET_PINECONE:
#     vdb.reset_vdb()
