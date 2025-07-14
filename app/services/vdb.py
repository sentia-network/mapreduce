from typing import Dict, List, Optional

from singleton_decorator import singleton

from app.core.vdb import (GaussES, PineconeSL, VectorDataBaseAbstract,
                          get_index_name, vdb_factory)
from app.schemas.public import EmbModelConf, OpenAIEmbModels


@singleton
class VDBService:

    def __init__(self):
        self.vdb_set: Dict[str, GaussES | PineconeSL] = {}

    def get_vdb(self, config: EmbModelConf) -> VectorDataBaseAbstract:
        index_name = get_index_name(config)
        if index_name in self.vdb_set:
            vdb = self.vdb_set[index_name]
        else:
            vdb = vdb_factory()(index_name, config.dim)
            self.vdb_set[index_name] = vdb
        return vdb

    @staticmethod
    async def global_vdb_delete(ids: List[str], query_filter: Optional[Dict] = None):
        models = [m.value for m in list(OpenAIEmbModels)]
        vdb_cls = vdb_factory()
        await vdb_cls.global_vdb_delete(ids, models, query_filter)


if __name__ == '__main__':
    from app.schemas.vector import (Vector, VectorBatch, VectorBatchOptional,
                                    VectorOptional, VectorQuery)
    from app.services.vdb import *

    vdb = VDBService().get_vdb(EmbModelConf.default())
    filter_dict1 = {'meta_test_field': 'test1', "common": 1}
    filter_dict2 = {'meta_test_field': 'test2', "common": 1}
    filter_dict = {"common": 1}
    test_vectors = VectorBatch(vectors=[
        Vector(id='123e4567-e89b-12d3-a456-426614174000', vector=[0.1] * 1536, metadata=filter_dict1),
        Vector(id='123e4567-e89b-12d3-a456-426614174001', vector=[0.1] * 1536, metadata=filter_dict2),
    ])
    test_update_m = VectorBatchOptional(vectors=[
        VectorOptional(id='123e4567-e89b-12d3-a456-426614174000', vector=None, metadata={"common": 5})])
    test_update_v = VectorBatchOptional(vectors=[
        VectorOptional(id='123e4567-e89b-12d3-a456-426614174000', vector=[0.2] * 1536, metadata=None)])
    # uncomment to test
    # await vdb.a_upsert(test_vectors)
    # # by ids
    # await vdb.a_delete(['123e4567-e89b-12d3-a456-426614174000', '123e4567-e89b-12d3-a456-426614174001'])
    # # by query
    # await vdb.a_delete(filter=filter_dict)
    #
    # ret = await vdb.a_fetch(['123e4567-e89b-12d3-a456-426614174000', '123e4567-e89b-12d3-a456-426614174001'])
    # await vdb.a_update(test_update_m)
    # await vdb.a_update(test_update_v)
    # ret = await vdb.a_query(VectorQuery(vector=[0.1] * 1536, filter={}), True, True)
    # await vdb_factory().global_vdb_delete(['123e4567-e89b-12d3-a456-426614174000'],
    #                                       ['text-embedding-ada-002-1536lsdkjf'])
