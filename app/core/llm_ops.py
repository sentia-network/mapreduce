from addict import Addict
from loguru import logger

from app.services.completion import CompletionService
from app.utils import load_jsons


async def a_propose_outline_sections(q: str, toc: str) -> Addict[str, int]:
    prompt = (
        f"Look at the following question:\n"
        f"{q}\n"
        f"and the following table of contents:\n"
        f"{toc}\n"
        f"Observe how the table of contents can be related to the questions "
        f"and output ids of the most relevant section titles in json format:\n"
        f"{{\n"
        f'   "observations": "The question is related to the table of content in such way.",\n'
        f'   "section_ids": ["id1", "id2"]\n'
        f'}}\n'
        f'Return empty list if all information need can be found in table of contents '
        f'or when no relevant section titles are found.'
    )
    resp = await CompletionService().a_create_completion(
        messages=[{'role': 'user', 'content': prompt}], response_format={"type": "json_object"})
    section_ids = load_jsons(resp.choices[0].message.content)
    logger.info(section_ids.observations)
    logger.info(section_ids.section_ids)
    return section_ids
