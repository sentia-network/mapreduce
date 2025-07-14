import json
from typing import Any, Dict, List, Optional, Tuple

from pinecone import Vector as PineconeVector
from pinecone.core.openapi.db_data.model.scored_vector import \
    ScoredVector as PineconeScoredVector
from pydantic import BaseModel, Field

try:
    from pinecone import QueryResult as PineconeQueryResponse  # v2
except ImportError:
    from pinecone import QueryResponse as PineconeQueryResponse  # v3

from pinecone import FetchResponse as PineconeFetchResponse


class VectorBase(BaseModel):
    vector: List[float] = Field(..., title='向量')

    def __repr__(self):
        return f'VectorBase<Anon>'


class Vector(VectorBase):
    id: str = Field(..., title='向量数据ID')
    metadata: Dict = Field(default_factory=dict, title='元数据')

    def __repr__(self):
        return f'Vector<{self.id}>'


class VectorOptional(Vector):
    vector: List[float] | None = Field(None, title='向量')
    metadata: Dict | None = Field(None, title='元数据')


class VectorBatch(BaseModel):
    vectors: List[Vector] = Field(..., title='向量列表')

    @classmethod
    def parse_ga_resp(cls, resp: Dict) -> 'VectorBatch':
        vectors = []
        for doc in resp.get('docs', []):
            if not doc.get('found'):
                continue
            vector = doc['_source'].get('vector')
            metadata = {k: v for k, v in doc.get('_source', {}).items()
                        if k not in ['_index', '_type', '_id', '_score', 'vector']}
            vectors.append(Vector(id=doc['_id'], vector=vector, metadata=metadata))
        return cls(vectors=vectors)

    def to_pinecone_vectors(self) -> List[PineconeVector]:
        return [PineconeVector(v.id, v.vector, v.metadata) for v in self.vectors]

    @classmethod
    def from_pinecone_vectors(cls, pinecone_vectors: List[PineconeVector]) -> 'VectorBatch':
        return cls(vectors=[Vector(vector=v.values, id=v.id, metadata=v.metadata) for v in pinecone_vectors])


class VectorBatchOptional(BaseModel):
    vectors: List[VectorOptional] = Field(..., title='向量列表')

    @classmethod
    def parse_ga_resp(cls, resp: Dict) -> 'VectorBatchOptional':
        vectors = []
        for doc in resp.get('docs', []):
            if not doc.get('found'):
                continue
            vector = doc['_source'].get('vector')
            metadata = {k: v for k, v in doc.get('_source', {}).items()
                        if k not in ['_index', '_type', '_id', '_score', 'vector']}
            vectors.append(VectorOptional(id=doc['_id'], vector=vector, metadata=metadata))
        return cls(vectors=vectors)

    def to_pinecone_vectors(self) -> List[PineconeVector]:
        return [PineconeVector(v.id, v.vector or [], v.metadata) for v in self.vectors]


class VectorQuery(VectorBase):
    top_k: int = Field(10, title='top k')
    filter: Dict = Field(default_factory=dict,
                         title='过滤参数，语法参考：https://docs.pinecone.io/docs/metadata-filtering')


class QueryResponse(Vector):
    vector: List[float] | None = Field(None, title='向量')
    score: float = Field(0, title='相似度或关联度分数')

    @classmethod
    def parse_pinecone_v7_matches(cls, resp: PineconeScoredVector) -> 'QueryResponse':
        return cls(vector=resp.values or None, id=resp.id, metadata=resp.metadata or {}, score=resp.score)

    @classmethod
    def parse_pinecone_v3_res(cls, res: PineconeQueryResponse) -> 'QueryResponse':
        return cls(id=res.id, vector=res.values, score=res.score or 0, metadata=res.metadata)

    @classmethod
    def parse_pinecone_v2_res(cls, res: dict) -> 'QueryResponse':
        return cls(id=res['id'], vector=res.get('values'), score=res['score'],
                   metadata=res.get('metadata'))

    @classmethod
    def parse_tencent_res(cls, res: dict) -> 'QueryResponse':
        prime_keys = ['id', 'vector', 'score']
        return cls(id=res['id'], vector=res.get('vector'), score=res['score'],
                   metadata={k: v if type(v) in [int, float, str] else json.loads(v)
                             for k, v in res.items() if k not in prime_keys})

    @classmethod
    def parse_volcano_res(cls, res: Any) -> 'QueryResponse':
        pass

    @classmethod
    def parse_es_res(cls, res: dict, include_vector: bool) -> 'QueryResponse':
        vector = res['_source'].get('vector') if include_vector else None
        _s = res['_score'] or 1e-10
        cos_sim = (_s * 2 - 1) / _s
        return cls(id=res['_id'], vector=vector, score=cos_sim,
                   metadata=res['_source']['metadata'])

    @classmethod
    def parse_ga_res(cls, res: dict, include_vector=False) -> 'QueryResponse':
        # include_vector = include_vector or res.get('vector')
        vector = res['_source'].get('vector') if include_vector else None
        metadata = {k: v for k, v in res.get('_source', {}).items()
                    if k not in ['_index', '_type', '_id', '_score', 'vector']}
        return cls(id=res['_id'], vector=vector, score=res['_score'],
                   metadata=metadata)

    def __eq__(self, other):
        if not isinstance(other, QueryResponse):
            return False
        tol = 5e-5
        return all([self.id == other.id,
                    self.vector == other.vector,
                    {**self.metadata, 'namespace': ''} == {**other.metadata, 'namespace': ''},
                    other.score - tol < self.score < other.score + tol])


class QueryResult(BaseModel):
    items: List[QueryResponse] = Field(..., title='召回结果列表')

    @classmethod
    def parse_pinecone_v7_res(cls, res: PineconeQueryResponse) -> 'QueryResult':
        return cls(items=[QueryResponse.parse_pinecone_v7_matches(m) for m in res.matches])

    @classmethod
    def parse_pinecone_v3_res(cls, res: PineconeQueryResponse) -> 'QueryResult':
        return cls(items=[QueryResponse.parse_pinecone_v3_res(m) for m in res.matches])

    @classmethod
    def parse_pinecone_v3_fetch(cls, fetch_resp: PineconeFetchResponse) -> 'QueryResult':
        items = fetch_resp.vectors.values()
        return cls(items=[QueryResponse.parse_pinecone_v3_res(r) for r in items])

    @classmethod
    def parse_pinecone_v2_res(cls, res: dict) -> 'QueryResult':
        return cls(items=[QueryResponse.parse_pinecone_v2_res(r) for r in res['matches']])

    @classmethod
    def parse_pinecone_v2_fetch(cls, fetch_resp: PineconeFetchResponse) -> 'QueryResult':
        res_dict = fetch_resp.to_dict()
        return cls(items=[QueryResponse.parse_pinecone_v2_res(v | {'score': -1})
                          for v in res_dict['vectors'].values()])

    @classmethod
    def parse_tencent_res(cls, res: List[List[dict]]) -> 'QueryResult':
        res = [i for r in res for i in r]
        return cls(items=[QueryResponse.parse_tencent_res(r) for r in res])

    @classmethod
    def parse_volcano_res(cls, res: Any) -> 'QueryResult':
        pass

    @classmethod
    def parse_es_res(cls, res: dict, include_vector: bool) -> 'QueryResult':
        res = res['hits']['hits']
        return cls(items=[QueryResponse.parse_es_res(r, include_vector) for r in res])

    @classmethod
    def parse_ga_res(cls, res: dict, include_vector: bool) -> 'QueryResult':
        hits = res['hits']['hits']
        return cls(items=[QueryResponse.parse_ga_res(h, include_vector) for h in hits])

    @classmethod
    def parse_ga_fetch(cls, fetch_resp: dict) -> 'QueryResult':
        docs = [d for d in fetch_resp['docs'] if d['found']]
        return cls(items=[QueryResponse.parse_ga_res(d | {'_score': -1}, include_vector=True)
                          for d in docs])

    def get_ids(self) -> List[str]:
        return [r.id for r in self.items]

    def get_upsert_data(self, exclude: Optional[List[str]] = None) -> List[Tuple[str, list, dict]]:
        exclude = exclude or []
        return [(r.id, r.vector, r.metadata) for r in self.items if r.id not in exclude]

    def __eq__(self, other):
        if not isinstance(other, QueryResult):
            return False
        if {*[r.id for r in self.items]} != {*[r.id for r in other.items]}:
            return False
        other_res_map = {r.id: r for r in other.items}
        try:
            eq = all([res == other_res_map[res.id] for res in self.items])
        except KeyError:
            return False
        return eq
