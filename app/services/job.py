import io
import json
import re
import zipfile

import httpx
import pandas as pd
from loguru import logger

from app.core.chunker import SimpleChunker
from app.core.kafka import MQManager
from app.core.parser import (DataBase2MarkdownParser, Image2MarkdownParser,
                             MarkdownParserFactory, Pdf2MarkdownParser,
                             Ppt2MarkdownParser, Table2MarkdownParser,
                             detect_encoding)
from app.middlewares.exception.common import CommonException
from app.schemas.chunk import QueryContentIndexed
from app.schemas.embed import EmbedCreateWithFileUrl
from app.schemas.facade import (ChunkCreate, DatasetJobCategory,
                                DatasetJobStatus)
from app.schemas.file import (AllowedDocumentSuffix, AllowedTableSuffix,
                              DatasetCreate)
from app.schemas.kafka import (KafkaJobDatasetMessage,
                               KafkaJobEmbedCreateMessage)
from app.schemas.public import AssetType, ChunkUpdateMode, OcrModel, OrcUsage
from app.schemas.request.job import JobDatasetBody, JobEmbedChunksBody
from app.services.embed import EmbedService
from app.services.external.facade import ServiceFacade
from app.settings.config import config


class JobService:
    facade_service = ServiceFacade()

    def __init__(self):
        ...

    @classmethod
    async def add_dataset_job(cls, body: JobDatasetBody):

        # send message to kafka
        data = DatasetCreate.model_validate(body.model_dump())
        kafka_msg = KafkaJobDatasetMessage(data=data, topic=config.KAFKA_TOPIC_DATASET_JOB)
        try:
            await MQManager().produce_message(message=kafka_msg)
        except Exception as e:
            raise CommonException.service_error(detail=str(e))

    @staticmethod
    async def process_dataset_job(message: KafkaJobDatasetMessage):
        data = message.data
        facade_service = ServiceFacade()
        embed_service = EmbedService()

        async def update_job_status(status, detail, usage=None):
            await facade_service.a_update_job_status(
                status, DatasetJobCategory.DATASET_CREATE, detail, extra=data.extra, usage=usage)

        await update_job_status(DatasetJobStatus.START, detail='开始执行数据集任务')

        # download file
        try:
            await update_job_status(DatasetJobStatus.PROCESSING, detail='下载源文件')
            with httpx.Client(timeout=15) as client:
                response = client.get(str(data.file_url))

            if response.status_code != 200:
                raise CommonException.system_error(detail='Failed to download file from url')
        except Exception as e:
            await update_job_status(DatasetJobStatus.EXCEPT, detail='下载文件失败')
            raise e

        file_bytes = io.BytesIO(response.read())
        # check QA table with columns '问题' and '答案'
        if isinstance(data.file_suffix, AllowedTableSuffix) and data.is_qa is True:
            try:
                await update_job_status(DatasetJobStatus.PROCESSING, detail='检验QA表格')
                logger.info('check QA table')
                match data.file_suffix:
                    case AllowedTableSuffix.XLSX:
                        df = pd.read_excel(file_bytes, engine='openpyxl')
                    case AllowedTableSuffix.CSV:
                        df = pd.read_csv(file_bytes, encoding=detect_encoding(file_bytes))
                    case _:
                        raise CommonException.system_error(f'Unsupported file suffix: {data.file_suffix}')
                file_bytes.seek(0)
                cols = df.columns.to_list()
                logger.info(f'columns: {cols}')
                if '问题' not in cols or '答案' not in cols:
                    raise CommonException.system_error('QA table must have columns "问题" and "答案"')
                else:
                    logger.info('QA table check passed')
            except Exception as e:
                await update_job_status(DatasetJobStatus.EXCEPT, detail='检验QA表格不通过')
                raise e

        # check markdown zip file has only one .md file
        if data.file_suffix == AllowedDocumentSuffix.ZIP:
            with zipfile.ZipFile(file_bytes, 'r') as zip_ref:
                all_md_file = [x for x in zip_ref.namelist()
                               if x.endswith('.md')
                               and '__MACOSX' not in x
                               and '/' not in x]
                if len(all_md_file) == 0:
                    logger.error('no .md file in zip')
                    await update_job_status(DatasetJobStatus.EXCEPT, detail='压缩包中没有.md文件')
                    return
                elif len(all_md_file) > 1:
                    logger.error('more than one .md file in zip')
                    await update_job_status(DatasetJobStatus.EXCEPT, detail='压缩包中有多个.md文件')
                    return
            file_bytes.seek(0)

        # parse file to Markdown format
        try:
            await update_job_status(DatasetJobStatus.PROCESSING, detail='解析源文件')

            def fn(b, name, suffix):
                try:
                    _data = facade_service.put_assets(file_bytes=b,
                                                      atype=AssetType.IMAGE,
                                                      name=name,
                                                      suffix=suffix,
                                                      extra=data.extra)
                    return True, _data.get('full_url'), _data.get('id')
                except Exception as err:
                    logger.warning(f'upload image failed: {str(err)}')
                    return False, '', ''

            parser = MarkdownParserFactory.get_parser(file_bytes=file_bytes, suffix=data.file_suffix, image_save_fn=fn)
            md = parser.load()
            logger.info('Markdown content:\n' + md[:1000] + '...' if len(md) > 1000 else 'Markdown content:\n' + md)

            # usage count
            if isinstance(parser, Image2MarkdownParser):
                await facade_service.send_usage(usage=OrcUsage(model=OcrModel.BAIDU, used_count=1))

            if not isinstance(parser, Table2MarkdownParser):
                if len(md.replace('\n', '').replace('\r', '')) >= data.max_len:
                    await update_job_status(DatasetJobStatus.EXCEPT, detail='源文件长度超过限制')
                    return

        except Exception as e:
            await update_job_status(DatasetJobStatus.EXCEPT, detail='解析源文件失败')
            raise e

        # chunk content
        header_ids_lst, outline = None, None
        try:
            await update_job_status(DatasetJobStatus.PROCESSING, detail='文档切分')
            chunk_size = data.chunk_config.chunk_size
            if isinstance(parser, Table2MarkdownParser):
                if data.is_qa is True:
                    chunk_set = SimpleChunker.chunk_qa_table(df=parser.table, q_col='问题', a_col='答案')
                else:
                    chunk_set = SimpleChunker.chunk_table(df=parser.table)
                    md = '\n\n'.join([x.raw_content for x in chunk_set.chunks])
            elif isinstance(parser, Ppt2MarkdownParser):
                chunk_set = SimpleChunker.chunk_with_pattern(pattern=r"(#\s*<第\d+页>)", content=md)
            elif isinstance(parser, Pdf2MarkdownParser) and parser.is_ppt is True:
                chunk_set = SimpleChunker.chunk_with_pattern(pattern=r"(#\s*<第\d+页>)", content=md)
            elif isinstance(parser, DataBase2MarkdownParser):
                chunk_set = SimpleChunker.chunk_with_splitter(splitter=r";", content=md)
            elif (splitter := data.chunk_config.splitter) is not None:
                chunk_set = SimpleChunker.chunk_with_splitter(content=md, chunk_size=chunk_size, splitter=splitter,
                                                              strip_splitter=data.chunk_config.strip_splitter)
            elif data.create_outline:
                chunk_set, header_ids_lst, outline = SimpleChunker.chunk_markdown_by_heading(md, data.file_name)
            else:
                chunk_set = SimpleChunker.chunk_markdown(content=md, chunk_size=chunk_size)

            logger.info('chunk head 5:\n' + '\n---<split>---\n'.join([c.raw_content for c in chunk_set.chunks[:5]]))

            if len(chunk_set.chunks) == 0:
                await update_job_status(DatasetJobStatus.FINISHED, '没有切分数据')
                return

            # fill asset ids to chunk
            pattern = r'!\[.*?\]\((.*?)(?:\s+".*?")?\)'
            for chunk in chunk_set.chunks:
                paths = re.findall(pattern, chunk.raw_content)
                assets = [asset.aid for p in paths if (asset := parser.get_asset_by_path(p)) is not None]
                chunk.asset_ids.extend(assets)

            # create chunk to facade
            facade_chunks = await facade_service.a_put_chunks(
                chunk_create=ChunkCreate.parse_from(chunk_set=chunk_set, extra=data.extra),
                md=md
            )

            # # update chunk id, next_id, sort_id, metadata
            # for chunk, facade_chunk in zip(chunk_set.chunks, facade_chunks):
            #     chunk.id = facade_chunk.id
            #     chunk.metadata = facade_chunk.chunk_metadata
            chunk_set.align_with_facade_chunks(facade_chunks)

        except Exception as e:
            await update_job_status(DatasetJobStatus.EXCEPT, detail='文档切分失败')
            raise e

        # add header ids to chunk metadata
        if header_ids_lst:
            try:
                facade_chunks = await facade_service.a_update_chunks_metadata(
                    extra=data.extra,
                    ids=[c.id for c in chunk_set.chunks],
                    metadata_lst=[{facade_service.OUTLINE_IDS: header_ids} for header_ids in header_ids_lst]
                )
                # # update chunk id, next_id, sort_id, metadata
                # for chunk, facade_chunk in zip(chunk_set.chunks, facade_chunks):
                #     chunk.id = facade_chunk.id
                #     chunk.metadata = facade_chunk.chunk_metadata
                chunk_set.align_with_facade_chunks(facade_chunks)

            except Exception as e:
                await update_job_status(DatasetJobStatus.EXCEPT, detail='更新切片元数据失败')
                raise e

        # update outline on dataset
        if outline:
            try:
                await facade_service.a_update_dataset(extra=data.extra, outline=outline)
            except Exception as e:
                await update_job_status(DatasetJobStatus.EXCEPT, detail='更新数据集大纲失败')
                raise e

        # embedding
        llm_usage = None
        try:
            content_list = chunk_set.to_content_list()
            llm_usage, e = await embed_service.a_embed_by_chunk(content_list=content_list,
                                                                embed_config=data.embed_config,
                                                                update_mode=ChunkUpdateMode.REPLACE)
            if e:
                raise e
            else:
                await update_job_status(DatasetJobStatus.PROCESSING, '切片向量化', usage=llm_usage)
                await facade_service.a_chunks_flag_embedded([c.id for c in content_list])
        except Exception as e:
            await update_job_status(DatasetJobStatus.EXCEPT, f'向量化失败，{e}', usage=llm_usage)
            raise e

        # sync finish state
        await update_job_status(DatasetJobStatus.FINISHED, '完成')

    @classmethod
    async def add_embed_create_job(cls, body: JobEmbedChunksBody):

        # send message to kafka
        data = EmbedCreateWithFileUrl.model_validate(body.model_dump())

        kafka_msg = KafkaJobEmbedCreateMessage(data=data, topic=config.KAFKA_TOPIC_EMBED_CREATE_JOB)
        try:
            await MQManager().produce_message(message=kafka_msg)
        except Exception as e:
            raise CommonException.service_error(detail=str(e))

    @staticmethod
    async def process_embed_create_job(message: KafkaJobEmbedCreateMessage):
        async def update_job_status(status, detail, usage=None):
            await facade_service.a_update_job_status(
                status, DatasetJobCategory.EMBED_CREATE, detail, extra=data.extra, usage=usage)

        data = message.data

        if data.items:
            items = data.items
        elif data.file_url:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(str(data.file_url))
                if resp.status_code != 200:
                    await update_job_status(DatasetJobStatus.EXCEPT, f'向量化失败，下载文件失败')
                    raise CommonException.cos_error(detail='download file failed')
            items = [QueryContentIndexed.model_validate(x) for x in json.loads(resp.content)]
        else:
            await update_job_status(DatasetJobStatus.EXCEPT, f'向量化失败，没有文件下载连接')
            raise CommonException.system_error('No items or file url provided')

        facade_service = ServiceFacade()
        embed_service = EmbedService()

        await update_job_status(DatasetJobStatus.START, '开始执行向量化')

        llm_usage, e = await embed_service.a_embed_by_chunk(content_list=items,
                                                            embed_config=data.embed_config,
                                                            update_mode=ChunkUpdateMode.REPLACE)

        if e:
            await update_job_status(DatasetJobStatus.EXCEPT, f'向量化失败，{e}', usage=llm_usage)
            raise e
        else:
            await facade_service.a_chunks_flag_embedded([c.id for c in items])
            await update_job_status(DatasetJobStatus.FINISHED, '向量化完成', usage=llm_usage)
