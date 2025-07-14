from fastapi import APIRouter

from .admin import chunk, job, llm_ops, retrieve, sync_ops

admin_router = APIRouter(prefix="/admin")
admin_router.include_router(job.router, tags=["Job"])
admin_router.include_router(sync_ops.router, tags=["Vector Sync Ops"])
admin_router.include_router(retrieve.router, tags=["Vector Query"])
admin_router.include_router(chunk.router, tags=["Chunk Text"])
admin_router.include_router(llm_ops.router, tags=["LLM Ops"])
