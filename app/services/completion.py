from openai.types.chat import ChatCompletion
from singleton_decorator import singleton

from app.schemas.public import CplModelConf, LLMUsage
from app.services.external.facade import ServiceFacade
from app.services.external.llm import LLMService


@singleton
class CompletionService:
    llm_service = LLMService()
    facade = ServiceFacade()

    async def a_create_completion(
            self, config: CplModelConf = None, **kw
    ) -> ChatCompletion:
        api_key_tag = self.facade.get_api_key()
        credential = dict(
            api_key=api_key_tag.openai_api_key, base_url=api_key_tag.openai_api_base
        ) if api_key_tag and api_key_tag.has_openai_api_key() else {}

        config = config or CplModelConf()
        resp = await self.llm_service.a_oai_completion(**(kw | config.to_kw() | credential))

        usage = LLMUsage.from_llm_resp(model=config.model, supplier=config.supplier, usage=resp.usage)
        if api_key_tag and (not api_key_tag.has_openai_api_key() or api_key_tag.is_platform_openai_api_key()):
            await self.facade.send_usage(usage)
        return resp
