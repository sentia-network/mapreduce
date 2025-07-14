import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pandas import DataFrame
from pydantic import BaseModel, Field

from app.schemas.chunk import ChunkOutlineCreate, ChunkSet, RawContentIndexed
from app.utils import generate_short_id, ordered_unique


class LineGroup(BaseModel):
    heading: str
    level: int
    lines: List[str]


class MarkdownSkeleton(BaseModel):
    id: str = Field(default_factory=generate_short_id)
    heading: str
    sub_sections: Optional[List['MarkdownSkeleton']] = None

    def collect(self, key: str | None = None, acc_mode=False, acc=None) -> List[Any] | List[List[Any]]:
        obj = getattr(self, key) if key else self
        ret = [obj] if not acc_mode else [(acc or []) + [obj]]
        next_acc = (acc or []) + [obj] if acc_mode else None
        if self.sub_sections:
            for s in self.sub_sections:
                ret.extend(s.collect(key, acc_mode, next_acc))
        return ret

    @property
    def full_headings(self):
        ids_lst = self.collect('id')
        headings_lst = self.collect('heading', True)
        header_with_id_strs = [f'{"  " * len(hs[:-1])}{hs[-1]} <id={id}>' for hs, id in zip(headings_lst, ids_lst)]
        full_header_str = '\n'.join(header_with_id_strs).replace("# ", '').replace('#', '')
        return full_header_str

    @property
    def light_headings(self):
        headings_lst = self.collect('heading', True)
        header_strs = [f'{"  " * len(hs[:-1])}{hs[-1]}' for hs in headings_lst]
        light_header_str = '\n'.join(header_strs).replace("# ", '').replace('#', '')
        return light_header_str


class MarkdownSection(MarkdownSkeleton):
    # id: str = Field(default_factory=generate_short_id)
    # heading: str
    sub_sections: Optional[List['MarkdownSection']] = None
    content: str

    @classmethod
    def parse_markdown(cls, markdown_text: str, root_name: str) -> 'MarkdownSection':
        lines = markdown_text.split('\n')
        start_with_level_0 = not lines[0].startswith('#')
        line_groups = [*([LineGroup(heading=root_name, lines=[], level=0)] if start_with_level_0 else [])]
        for l in lines:
            if l.startswith('#'):
                level = l.split(' ')[0].count('#')
                line_groups.append(LineGroup(heading=l, lines=[], level=level))
            else:
                line_groups[-1].lines.append(l)

        def parse_section(line_groups: List[LineGroup]) -> List[MarkdownSection]:
            first_level = line_groups[0].level
            level_groups = []
            for lg in line_groups:
                if lg.level == first_level:
                    level_groups.append([lg])
                elif lg.level > first_level:
                    level_groups[-1].append(lg)
                else:
                    raise ValueError(f"Invalid heading level! Please Check: {line_groups[0].heading} -> {lg.heading}")

            sections = []
            for i, level_group in enumerate(level_groups):
                line_group, *rest_line_group = level_group
                sub_sections = parse_section(rest_line_group) if rest_line_group else None
                sections.append(cls(heading=line_group.heading,
                                    content='\n'.join(line_group.lines),
                                    sub_sections=sub_sections))
            return sections

        secs = parse_section(line_groups)

        if start_with_level_0:
            return secs[0]
        return cls(heading=root_name, content='', sub_sections=secs)

    @classmethod
    def parse_chunks(cls, chunks: List[str], root_name: str) -> Tuple[List[List[str]], 'MarkdownSection']:
        md_sec = cls.parse_markdown('\n'.join(chunks), root_name)
        ids_lst = md_sec.collect('id', True)
        first_ids = ids_lst[0]
        headings = md_sec.collect('heading')
        heading_to_ids = {h: id_lst for h, id_lst in zip(headings, ids_lst)}
        chunk_heading_ids = []
        previous_heading_ids = []
        for c in chunks:
            first_line, *rest_lines = c.split('\n')
            title_heading_ids = heading_to_ids[first_line] if first_line.startswith('#') else previous_heading_ids
            heading_ids = [i for h, id_lst in heading_to_ids.items() for i in id_lst if h in rest_lines]
            chunk_heading_ids.append(ordered_unique(first_ids + title_heading_ids + heading_ids))
            previous_heading_ids = heading_ids or previous_heading_ids
        return chunk_heading_ids, md_sec

    def to_skeleton(self) -> MarkdownSkeleton:
        return MarkdownSkeleton.model_validate(self.model_dump())


class SimpleChunker:

    @classmethod
    def parse_chunks(cls, data: ChunkOutlineCreate) -> Tuple[Dict[str, List[str]], Dict]:
        chunk_ids = [i.id for i in data.items]
        chunks = [i.raw_content for i in data.items]
        chunk_heading_ids, md_sec = MarkdownSection.parse_chunks(chunks, data.file_name)
        assert len(chunk_ids) == len(chunk_heading_ids), 'Lengths of chunks and ids should be the same'
        chunk_id_to_heading_ids = {i: id_lst for i, id_lst in zip(chunk_ids, chunk_heading_ids)}
        return chunk_id_to_heading_ids, md_sec.to_skeleton().model_dump(mode='json')

    @classmethod
    def chunk_markdown_by_heading(cls, content: str, root_name: str) -> Tuple[ChunkSet, List[List[str]], Dict]:
        md_sec = MarkdownSection.parse_markdown(content, root_name)
        chunks = md_sec.collect('content')
        headings = md_sec.collect('heading')
        ids_lst = md_sec.collect('id', True)
        assert len(chunks) == len(headings) == len(ids_lst), 'Lengths of chunks, headings and ids should be the same'
        non_empty_indices = [i for i, c in enumerate(chunks) if c]
        contents = [f'{headings[i]}\n{chunks[i]}' for i in non_empty_indices]
        ids_lst = [ids_lst[i] for i in non_empty_indices]
        df = pd.DataFrame(contents, columns=['raw_content'])
        df['query_content'] = df['raw_content']
        return cls.to_chunk_set(df), ids_lst, md_sec.to_skeleton().model_dump(mode='json')

    @classmethod
    def chunk_markdown(cls, content: str, chunk_size=500, chunk_overlap=0) -> ChunkSet:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        df = pd.DataFrame(splitter.split_text(content), columns=['raw_content'])
        df['query_content'] = df['raw_content']
        return cls.to_chunk_set(df)

    @classmethod
    def chunk_qa_table(cls, df: DataFrame, q_col: str, a_col: str) -> ChunkSet:
        df = df.rename({q_col: 'query_content', a_col: 'raw_content'}, axis=1)
        return cls.to_chunk_set(df)

    @classmethod
    def chunk_table(cls, df: DataFrame) -> ChunkSet:
        chunks = []
        for _, row in df.iterrows():
            content = []
            for col in df.columns:
                if row[col] != '':
                    content.append(f'{col}: {row[col]}')

            if len(content) != 0:
                chunks.append('\n'.join(content) + '\n')

        return cls.to_chunk_set(pd.DataFrame({'raw_content': chunks, 'query_content': chunks}))

    @classmethod
    def chunk_with_splitter(cls, splitter, content: str, chunk_size=500, chunk_overlap=0,
                            strip_splitter=True) -> ChunkSet:
        if strip_splitter:
            content_ls = content.split(splitter)
        else:
            split_ls = re.split(f'({splitter})', content)
            content_ls = []
            idx = 0
            while True:
                if idx > len(split_ls) - 1:
                    break
                if split_ls[idx] != splitter:
                    content_ls.append(split_ls[idx])
                else:
                    if idx + 1 <= len(split_ls):
                        content_ls.append(splitter + split_ls[idx + 1])
                        idx += 1
                    else:
                        content_ls.append(splitter)
                idx += 1

        content_ls = [x for x in content_ls if x != '']
        df = pd.DataFrame(content_ls, columns=['raw_split'])
        sp = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        df['raw_content'] = df['raw_split'].apply(sp.split_text)
        df = df.explode('raw_content').drop('raw_split', axis=1).reset_index(drop=True)
        df['raw_content'] = df['raw_content']
        df['query_content'] = df['raw_content']
        return cls.to_chunk_set(df)

    @classmethod
    def chunk_with_pattern(cls, pattern: str, content: str, strip_splitter: bool = True) -> ChunkSet:
        segments = re.split(pattern, content)
        segments = [segment.strip() for segment in segments if segment.strip()]

        if strip_splitter:
            segments_with_header = [segments[i + 1] for i in range(0, len(segments), 2)]
        else:
            segments_with_header = ['\n'.join(segments[i: i + 2]) for i in range(0, len(segments), 2)]

        df = pd.DataFrame(segments_with_header, columns=['raw_content'])
        df['query_content'] = df['raw_content']
        return cls.to_chunk_set(df)

    @staticmethod
    def to_chunk_set(df: DataFrame) -> ChunkSet:
        df = df[['raw_content', 'query_content']].copy()
        df['sort_id'] = range(len(df))
        df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
        df['next_id'] = df['id'].shift(-1).fillna('')
        df = df.dropna()
        df[['raw_content', 'query_content']] = df[['raw_content', 'query_content']].astype(str)
        df = df[df['raw_content'].apply(len) > 0]
        df = df[df['query_content'].apply(len) > 0]
        return ChunkSet(chunks=df.to_dict(orient='records'))


if __name__ == '__main__':
    # parse tests
    # # invalid headings
    md_sec1 = MarkdownSection.parse_markdown("""abc
### C
content1
# A
content1
## A1
content2
# B
content3""", 'root')
    # # fragmented chunking
    ids, md_sec2 = MarkdownSection.parse_chunks([
        "cde",
        """abc
content1
# A
content1""",
        "content0",
        "content1",
        """## A1
content2
# B
content3""",
        "# C"
    ], 'root')
