from typing import Dict, List, Tuple

from app.core.chunker import SimpleChunker
from app.schemas.chunk import ChunkOutlineCreate, ChunkSplitCreate


class ChunkService:

    def __init__(self):
        ...

    @staticmethod
    def split_text(body: ChunkSplitCreate) -> List[str]:
        if body.chunk_config.splitter is None:
            chunk_set = SimpleChunker.chunk_markdown(content=body.content,
                                                     chunk_size=body.chunk_config.chunk_size,
                                                     chunk_overlap=body.chunk_config.chunk_overlap)
        else:
            chunk_set = SimpleChunker.chunk_with_splitter(splitter=body.chunk_config.splitter,
                                                          content=body.content,
                                                          chunk_size=body.chunk_config.chunk_size,
                                                          chunk_overlap=body.chunk_config.chunk_overlap,
                                                          strip_splitter=body.chunk_config.strip_splitter)

        return [c.raw_content for c in chunk_set.chunks]

    @staticmethod
    def extract_outline(body: ChunkOutlineCreate) -> Tuple[Dict[str, List[str]], Dict]:
        chunk_id_to_heading_ids, outline = SimpleChunker.parse_chunks(data=body)
        return chunk_id_to_heading_ids, outline
