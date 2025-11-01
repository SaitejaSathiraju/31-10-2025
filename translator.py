import html
import json
import logging
import os
import re
import unicodedata
from copy import copy
from string import Template
from typing import cast
import deepl
import ollama
import openai
import requests
import xinference_client
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential
from tencentcloud.common import credential
from tencentcloud.tmt.v20180321.models import (
    TextTranslateRequest,
    TextTranslateResponse,
)
from tencentcloud.tmt.v20180321.tmt_client import TmtClient

from pdf2zh.cache import TranslationCache
from pdf2zh.config import ConfigManager

logger = logging.getLogger(__name__)

# Apply compatibility patches before importing IndicTrans
try:
    from pdf2zh import indic_trans_compat  # Apply compatibility fixes
except ImportError:
    pass

# IndicTrans imports
INDIC_TRANS_AVAILABLE = False
IndicProcessor = None

try:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    # Try multiple import paths
    try:
        from IndicTransToolkit import IndicProcessor
        INDIC_TRANS_AVAILABLE = True
        logger.info("IndicTransToolkit imported successfully (direct import)")
    except ImportError:
        try:
            from IndicTransToolkit.processor import IndicProcessor
            INDIC_TRANS_AVAILABLE = True
            logger.info("IndicTransToolkit imported successfully (processor import)")
        except ImportError:
            try:
                # Try alternative import method
                import IndicTransToolkit as itk
                IndicProcessor = getattr(itk, 'IndicProcessor', None)
                if IndicProcessor is not None:
                    INDIC_TRANS_AVAILABLE = True
                    logger.info("IndicTransToolkit imported successfully (attribute access)")
            except Exception:
                pass
except ImportError as e:
    logger.warning(f"IndicTransToolkit dependencies not available: {e}")
except Exception as e:
    logger.warning(f"Error checking IndicTransToolkit availability: {e}")


from tenacity import retry, retry_if_exception_type
from tenacity import stop_after_attempt
from tenacity import wait_exponential


logger = logging.getLogger(__name__)


def remove_control_characters(s):
    return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")


class BaseTranslator:
    name = "base"
    envs = {}
    lang_map: dict[str, str] = {}
    CustomPrompt = False

    def __init__(self, lang_in: str, lang_out: str, model: str, ignore_cache: bool):
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        self.model = model
        self.ignore_cache = ignore_cache

        self.cache = TranslationCache(
            self.name,
            {
                "lang_in": lang_in,
                "lang_out": lang_out,
                "model": model,
            },
        )

    def set_envs(self, envs):
        # Detach from self.__class__.envs
        # Cannot use self.envs = copy(self.__class__.envs)
        # because if set_envs called twice, the second call will override the first call
        self.envs = copy(self.envs)
        if ConfigManager.get_translator_by_name(self.name):
            self.envs = ConfigManager.get_translator_by_name(self.name)
        needUpdate = False
        for key in self.envs:
            if key in os.environ:
                self.envs[key] = os.environ[key]
                needUpdate = True
        if needUpdate:
            ConfigManager.set_translator_by_name(self.name, self.envs)
        if envs is not None:
            for key in envs:
                self.envs[key] = envs[key]
            ConfigManager.set_translator_by_name(self.name, self.envs)

    def add_cache_impact_parameters(self, k: str, v):
        """
        Add parameters that affect the translation quality to distinguish the translation effects under different parameters.
        :param k: key
        :param v: value
        """
        self.cache.add_params(k, v)

    def translate(self, text: str, ignore_cache: bool = False) -> str:
        """
        Translate the text, and the other part should call this method.
        :param text: text to translate
        :return: translated text
        """
        if not (self.ignore_cache or ignore_cache):
            cache = self.cache.get(text)
            if cache is not None:
                return cache

        translation = self.do_translate(text)
        self.cache.set(text, translation)
        return translation

    def do_translate(self, text: str) -> str:
        """
        Actual translate text, override this method
        :param text: text to translate
        :return: translated text
        """
        raise NotImplementedError

    def prompt(
        self, text: str, prompt_template: Template | None = None
    ) -> list[dict[str, str]]:
        try:
            return [
                {
                    "role": "user",
                    "content": cast(Template, prompt_template).safe_substitute(
                        {
                            "lang_in": self.lang_in,
                            "lang_out": self.lang_out,
                            "text": text,
                        }
                    ),
                }
            ]
        except AttributeError:  # `prompt_template` is None
            pass
        except Exception:
            logging.exception("Error parsing prompt, use the default prompt.")

        return [
            {
                "role": "user",
                "content": (
                    "You are a professional, authentic machine translation engine. "
                    "Only Output the translated text, do not include any other text."
                    "\n\n"
                    f"Translate the following markdown source text to {self.lang_out}. "
                    "Keep the formula notation {v*} unchanged. "
                    "Output translation directly without any additional text."
                    "\n\n"
                    f"Source Text: {text}"
                    "\n\n"
                    "Translated Text:"
                ),
            },
        ]

    def __str__(self):
        return f"{self.name} {self.lang_in} {self.lang_out} {self.model}"

    def get_rich_text_left_placeholder(self, id: int):
        return f"<b{id}>"

    def get_rich_text_right_placeholder(self, id: int):
        return f"</b{id}>"

    def get_formular_placeholder(self, id: int):
        return self.get_rich_text_left_placeholder(
            id
        ) + self.get_rich_text_right_placeholder(id)


class GoogleTranslator(BaseTranslator):
    name = "google"
    lang_map = {"zh": "zh-CN"}

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.session = requests.Session()
        self.endpoint = "https://translate.google.com/m"
        self.headers = {
            "User-Agent": "Mozilla/4.0 (compatible;MSIE 6.0;Windows NT 5.1;SV1;.NET CLR 1.1.4322;.NET CLR 2.0.50727;.NET CLR 3.0.04506.30)"  # noqa: E501
        }

    def do_translate(self, text):
        text = text[:5000]  # google translate max length
        response = self.session.get(
            self.endpoint,
            params={"tl": self.lang_out, "sl": self.lang_in, "q": text},
            headers=self.headers,
        )
        re_result = re.findall(
            r'(?s)class="(?:t0|result-container)">(.*?)<', response.text
        )
        if response.status_code == 400:
            result = "IRREPARABLE TRANSLATION ERROR"
        else:
            response.raise_for_status()
            result = html.unescape(re_result[0])
        return remove_control_characters(result)


class BingTranslator(BaseTranslator):
    # https://github.com/immersive-translate/old-immersive-translate/blob/6df13da22664bea2f51efe5db64c63aca59c4e79/src/background/translationService.js
    name = "bing"
    lang_map = {"zh": "zh-Hans"}

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.session = requests.Session()
        self.endpoint = "https://www.bing.com/translator"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",  # noqa: E501
        }

    def find_sid(self):
        response = self.session.get(self.endpoint)
        response.raise_for_status()
        url = response.url[:-10]
        ig = re.findall(r"\"ig\":\"(.*?)\"", response.text)[0]
        iid = re.findall(r"data-iid=\"(.*?)\"", response.text)[-1]
        key, token = re.findall(
            r"params_AbusePreventionHelper\s=\s\[(.*?),\"(.*?)\",", response.text
        )[0]
        return url, ig, iid, key, token

    def do_translate(self, text):
        text = text[:1000]  # bing translate max length
        url, ig, iid, key, token = self.find_sid()
        response = self.session.post(
            f"{url}ttranslatev3?IG={ig}&IID={iid}",
            data={
                "fromLang": self.lang_in,
                "to": self.lang_out,
                "text": text,
                "token": token,
                "key": key,
            },
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()[0]["translations"][0]["text"]


class DeepLTranslator(BaseTranslator):
    # https://github.com/DeepLcom/deepl-python
    name = "deepl"
    envs = {
        "DEEPL_AUTH_KEY": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        auth_key = self.envs["DEEPL_AUTH_KEY"]
        self.client = deepl.Translator(auth_key)

    def do_translate(self, text):
        response = self.client.translate_text(
            text, target_lang=self.lang_out, source_lang=self.lang_in
        )
        return response.text


class DeepLXTranslator(BaseTranslator):
    # https://deeplx.owo.network/endpoints/free.html
    name = "deeplx"
    envs = {
        "DEEPLX_ENDPOINT": "https://api.deepl.com/translate",
        "DEEPLX_ACCESS_TOKEN": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.endpoint = self.envs["DEEPLX_ENDPOINT"]
        self.session = requests.Session()
        auth_key = self.envs["DEEPLX_ACCESS_TOKEN"]
        if auth_key:
            self.endpoint = f"{self.endpoint}?token={auth_key}"

    def do_translate(self, text):
        response = self.session.post(
            self.endpoint,
            json={
                "source_lang": self.lang_in,
                "target_lang": self.lang_out,
                "text": text,
            },
            verify=False,  # noqa: S506
        )
        response.raise_for_status()
        return response.json()["data"]


class OllamaTranslator(BaseTranslator):
    # https://github.com/ollama/ollama-python
    name = "ollama"
    envs = {
        "OLLAMA_HOST": "http://127.0.0.1:11434",
        "OLLAMA_MODEL": "gemma2",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in: str,
        lang_out: str,
        model: str,
        envs=None,
        prompt: Template | None = None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["OLLAMA_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {
            "temperature": 0,  # 随机采样可能会打断公式标记
            "num_predict": 2000,
        }
        self.client = ollama.Client(host=self.envs["OLLAMA_HOST"])
        self.prompt_template = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])

    def do_translate(self, text: str) -> str:
        if (max_token := len(text) * 5) > self.options["num_predict"]:
            self.options["num_predict"] = max_token

        response = self.client.chat(
            model=self.model,
            messages=self.prompt(text, self.prompt_template),
            options=self.options,
        )
        content = self._remove_cot_content(response.message.content or "")
        return content.strip()

    @staticmethod
    def _remove_cot_content(content: str) -> str:
        """Remove text content with the thought chain from the chat response

        :param content: Non-streaming text content
        :return: Text without a thought chain
        """
        return re.sub(r"^<think>.+?</think>", "", content, count=1, flags=re.DOTALL)


class XinferenceTranslator(BaseTranslator):
    # https://github.com/xorbitsai/inference
    name = "xinference"
    envs = {
        "XINFERENCE_HOST": "http://127.0.0.1:9997",
        "XINFERENCE_MODEL": "gemma-2-it",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["XINFERENCE_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = xinference_client.RESTfulClient(self.envs["XINFERENCE_HOST"])
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])

    def do_translate(self, text):
        maxlen = max(2000, len(text) * 5)
        for model in self.model.split(";"):
            try:
                xf_model = self.client.get_model(model)
                xf_prompt = self.prompt(text, self.prompttext)
                xf_prompt = [
                    {
                        "role": "user",
                        "content": xf_prompt[0]["content"]
                        + "\n"
                        + xf_prompt[1]["content"],
                    }
                ]
                response = xf_model.chat(
                    generate_config=self.options,
                    messages=xf_prompt,
                )

                response = response["choices"][0]["message"]["content"].replace(
                    "<end_of_turn>", ""
                )
                if len(response) > maxlen:
                    raise Exception("Response too long")
                return response.strip()
            except Exception as e:
                print(e)
        raise Exception("All models failed")


class OpenAITranslator(BaseTranslator):
    # https://github.com/openai/openai-python
    name = "openai"
    envs = {
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_KEY": None,
        "OPENAI_MODEL": "gpt-4o-mini",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        if not model:
            model = self.envs["OPENAI_MODEL"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = openai.OpenAI(
            base_url=base_url or self.envs["OPENAI_BASE_URL"],
            api_key=api_key or self.envs["OPENAI_API_KEY"],
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))
        think_filter_regex = r"^<think>.+?\n*(</think>|\n)*(</think>)\n*"
        self.add_cache_impact_parameters("think_filter_regex", think_filter_regex)
        self.think_filter_regex = re.compile(think_filter_regex, flags=re.DOTALL)

    @retry(
        retry=retry_if_exception_type(openai.RateLimitError),
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        before_sleep=lambda retry_state: logger.warning(
            f"RateLimitError, retrying in {retry_state.next_action.sleep} seconds... "
            f"(Attempt {retry_state.attempt_number}/100)"
        ),
    )
    def do_translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
        )
        if not response.choices:
            if hasattr(response, "error"):
                raise ValueError("Error response from Service", response.error)
        content = response.choices[0].message.content.strip()
        content = self.think_filter_regex.sub("", content).strip()
        return content

    def get_formular_placeholder(self, id: int):
        return "{{v" + str(id) + "}}"

    def get_rich_text_left_placeholder(self, id: int):
        return self.get_formular_placeholder(id)

    def get_rich_text_right_placeholder(self, id: int):
        return self.get_formular_placeholder(id + 1)


class AzureOpenAITranslator(BaseTranslator):
    name = "azure-openai"
    envs = {
        "AZURE_OPENAI_BASE_URL": None,  # e.g. "https://xxx.openai.azure.com"
        "AZURE_OPENAI_API_KEY": None,
        "AZURE_OPENAI_MODEL": "gpt-4o-mini",
        "AZURE_OPENAI_API_VERSION": "2024-06-01",  # default api version
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = self.envs["AZURE_OPENAI_BASE_URL"]
        if not model:
            model = self.envs["AZURE_OPENAI_MODEL"]
        api_version = self.envs.get("AZURE_OPENAI_API_VERSION", "2024-06-01")
        if api_key is None:
            api_key = self.envs["AZURE_OPENAI_API_KEY"]
        super().__init__(lang_in, lang_out, model, ignore_cache)
        self.options = {"temperature": 0}
        self.client = openai.AzureOpenAI(
            azure_endpoint=base_url,
            azure_deployment=model,
            api_version=api_version,
            api_key=api_key,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("temperature", self.options["temperature"])
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))

    def do_translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=self.prompt(text, self.prompttext),
        )
        return response.choices[0].message.content.strip()


class ModelScopeTranslator(OpenAITranslator):
    name = "modelscope"
    envs = {
        "MODELSCOPE_BASE_URL": "https://api-inference.modelscope.cn/v1",
        "MODELSCOPE_API_KEY": None,
        "MODELSCOPE_MODEL": "Qwen/Qwen2.5-32B-Instruct",
    }
    CustomPrompt = True

    def __init__(
        self,
        lang_in,
        lang_out,
        model,
        base_url=None,
        api_key=None,
        envs=None,
        prompt=None,
        ignore_cache=False,
    ):
        self.set_envs(envs)
        base_url = "https://api-inference.modelscope.cn/v1"
        api_key = self.envs["MODELSCOPE_API_KEY"]
        if not model:
            model = self.envs["MODELSCOPE_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class ZhipuTranslator(OpenAITranslator):
    # https://bigmodel.cn/dev/api/thirdparty-frame/openai-sdk
    name = "zhipu"
    envs = {
        "ZHIPU_API_KEY": None,
        "ZHIPU_MODEL": "glm-4-flash",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://open.bigmodel.cn/api/paas/v4"
        api_key = self.envs["ZHIPU_API_KEY"]
        if not model:
            model = self.envs["ZHIPU_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))

    def do_translate(self, text) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                **self.options,
                messages=self.prompt(text, self.prompttext),
            )
        except openai.BadRequestError as e:
            if (
                json.loads(response.choices[0].message.content.strip())["error"]["code"]
                == "1301"
            ):
                return "IRREPARABLE TRANSLATION ERROR"
            raise e
        return response.choices[0].message.content.strip()


class SiliconTranslator(OpenAITranslator):
    # https://docs.siliconflow.cn/quickstart
    name = "silicon"
    envs = {
        "SILICON_API_KEY": None,
        "SILICON_MODEL": "Qwen/Qwen2.5-7B-Instruct",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.siliconflow.cn/v1"
        api_key = self.envs["SILICON_API_KEY"]
        if not model:
            model = self.envs["SILICON_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class X302AITranslator(OpenAITranslator):
    # https://doc.302.ai/
    name = "302ai"
    envs = {
        "X302AI_API_KEY": None,
        "X302AI_MODEL": "Gemma-7B",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.302.ai/v1"
        api_key = self.envs["X302AI_API_KEY"]
        if not model:
            model = self.envs["X302AI_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class GeminiTranslator(OpenAITranslator):
    # https://ai.google.dev/gemini-api/docs/openai
    name = "gemini"
    envs = {
        "GEMINI_API_KEY": None,
        "GEMINI_MODEL": "gemini-1.5-flash",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        api_key = self.envs["GEMINI_API_KEY"]
        if not model:
            model = self.envs["GEMINI_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt
        self.add_cache_impact_parameters("prompt", self.prompt("", self.prompttext))


class AzureTranslator(BaseTranslator):
    # https://github.com/Azure/azure-sdk-for-python
    name = "azure"
    envs = {
        "AZURE_ENDPOINT": "https://api.translator.azure.cn",
        "AZURE_API_KEY": None,
    }
    lang_map = {"zh": "zh-Hans"}

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model, ignore_cache)
        endpoint = self.envs["AZURE_ENDPOINT"]
        api_key = self.envs["AZURE_API_KEY"]
        credential = AzureKeyCredential(api_key)
        self.client = TextTranslationClient(
            endpoint=endpoint, credential=credential, region="chinaeast2"
        )
        # https://github.com/Azure/azure-sdk-for-python/issues/9422
        logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
        logger.setLevel(logging.WARNING)

    def do_translate(self, text) -> str:
        response = self.client.translate(
            body=[text],
            from_language=self.lang_in,
            to_language=[self.lang_out],
        )
        translated_text = response[0].translations[0].text
        return translated_text


class TencentTranslator(BaseTranslator):
    # https://github.com/TencentCloud/tencentcloud-sdk-python
    name = "tencent"
    envs = {
        "TENCENTCLOUD_SECRET_ID": None,
        "TENCENTCLOUD_SECRET_KEY": None,
    }

    def __init__(
        self, lang_in, lang_out, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_in, lang_out, model)
        try:
            cred = credential.DefaultCredentialProvider().get_credential()
        except EnvironmentError:
            cred = credential.Credential(
                self.envs["TENCENTCLOUD_SECRET_ID"],
                self.envs["TENCENTCLOUD_SECRET_KEY"],
            )
        self.client = TmtClient(cred, "ap-beijing")
        self.req = TextTranslateRequest()
        self.req.Source = self.lang_in
        self.req.Target = self.lang_out
        self.req.ProjectId = 0

    def do_translate(self, text):
        self.req.SourceText = text
        resp: TextTranslateResponse = self.client.TextTranslate(self.req)
        return resp.TargetText


class AnythingLLMTranslator(BaseTranslator):
    name = "anythingllm"
    envs = {
        "AnythingLLM_URL": None,
        "AnythingLLM_APIKEY": None,
    }
    CustomPrompt = True

    def __init__(
        self, lang_out, lang_in, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        super().__init__(lang_out, lang_in, model, ignore_cache)
        self.api_url = self.envs["AnythingLLM_URL"]
        self.api_key = self.envs["AnythingLLM_APIKEY"]
        self.headers = {
            "accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.prompttext = prompt

    def do_translate(self, text):
        messages = self.prompt(text, self.prompttext)
        payload = {
            "message": messages,
            "mode": "chat",
            "sessionId": "translation_expert",
        }

        response = requests.post(
            self.api_url, headers=self.headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        data = response.json()

        if "textResponse" in data:
            return data["textResponse"].strip()


class DifyTranslator(BaseTranslator):
    name = "dify"
    envs = {
        "DIFY_API_URL": None,  # 填写实际 Dify API 地址
        "DIFY_API_KEY": None,  # 替换为实际 API 密钥
    }

    def __init__(
        self, lang_out, lang_in, model, envs=None, ignore_cache=False, **kwargs
    ):
        self.set_envs(envs)
        super().__init__(lang_out, lang_in, model, ignore_cache)
        self.api_url = self.envs["DIFY_API_URL"]
        self.api_key = self.envs["DIFY_API_KEY"]

    def do_translate(self, text):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "inputs": {
                "lang_out": self.lang_out,
                "lang_in": self.lang_in,
                "text": text,
            },
            "response_mode": "blocking",
            "user": "translator-service",
        }

        # 向 Dify 服务器发送请求
        response = requests.post(
            self.api_url, headers=headers, data=json.dumps(payload)
        )
        response.raise_for_status()
        response_data = response.json()

        # 解析响应
        return response_data.get("answer", "")


class ArgosTranslator(BaseTranslator):
    name = "argos"

    def __init__(self, lang_in, lang_out, model, ignore_cache=False, **kwargs):
        try:
            import argostranslate.package
            import argostranslate.translate
        except ImportError:
            logger.warning(
                "argos-translate is not installed, if you want to use argostranslate, please install it. If you don't use argostranslate translator, you can safely ignore this warning."
            )
            raise
        super().__init__(lang_in, lang_out, model, ignore_cache)
        lang_in = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out = self.lang_map.get(lang_out.lower(), lang_out)
        self.lang_in = lang_in
        self.lang_out = lang_out
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        try:
            available_package = list(
                filter(
                    lambda x: x.from_code == self.lang_in
                    and x.to_code == self.lang_out,
                    available_packages,
                )
            )[0]
        except Exception:
            raise ValueError(
                "lang_in and lang_out pair not supported by Argos Translate."
            )
        download_path = available_package.download()
        argostranslate.package.install_from_path(download_path)

    def translate(self, text: str, ignore_cache: bool = False):
        # Translate
        import argotranslate.translate  # noqa: F401

        installed_languages = (
            argostranslate.translate.get_installed_languages()  # noqa: F821
        )
        from_lang = list(filter(lambda x: x.code == self.lang_in, installed_languages))[
            0
        ]
        to_lang = list(filter(lambda x: x.code == self.lang_out, installed_languages))[
            0
        ]
        translation = from_lang.get_translation(to_lang)
        translatedText = translation.translate(text)
        return translatedText


class GrokTranslator(OpenAITranslator):
    # https://docs.x.ai/docs/overview#getting-started
    name = "grok"
    envs = {
        "GROK_API_KEY": None,
        "GROK_MODEL": "grok-2-1212",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.x.ai/v1"
        api_key = self.envs["GROK_API_KEY"]
        if not model:
            model = self.envs["GROK_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


class GroqTranslator(OpenAITranslator):
    name = "groq"
    envs = {
        "GROQ_API_KEY": None,
        "GROQ_MODEL": "llama-3-3-70b-versatile",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.groq.com/openai/v1"
        api_key = self.envs["GROQ_API_KEY"]
        if not model:
            model = self.envs["GROQ_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


class DeepseekTranslator(OpenAITranslator):
    name = "deepseek"
    envs = {
        "DEEPSEEK_API_KEY": None,
        "DEEPSEEK_MODEL": "deepseek-chat",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://api.deepseek.com/v1"
        api_key = self.envs["DEEPSEEK_API_KEY"]
        if not model:
            model = self.envs["DEEPSEEK_MODEL"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


class OpenAIlikedTranslator(OpenAITranslator):
    name = "openailiked"
    envs = {
        "OPENAILIKED_BASE_URL": None,
        "OPENAILIKED_API_KEY": None,
        "OPENAILIKED_MODEL": None,
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        if self.envs["OPENAILIKED_BASE_URL"]:
            base_url = self.envs["OPENAILIKED_BASE_URL"]
        else:
            raise ValueError("The OPENAILIKED_BASE_URL is missing.")
        if not model:
            if self.envs["OPENAILIKED_MODEL"]:
                model = self.envs["OPENAILIKED_MODEL"]
            else:
                raise ValueError("The OPENAILIKED_MODEL is missing.")
        if self.envs["OPENAILIKED_API_KEY"] is None:
            api_key = "openailiked"
        else:
            api_key = self.envs["OPENAILIKED_API_KEY"]
        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt


class QwenMtTranslator(OpenAITranslator):
    """
    Use Qwen-MT model from Aliyun. it's designed for translating.
    Since Traditional Chinese is not yet supported by Aliyun. it will be also translated to Simplified Chinese, when it's selected.
    There's special parameters in the message to the server.
    """

    name = "qwen-mt"
    envs = {
        "ALI_MODEL": "qwen-mt-turbo",
        "ALI_API_KEY": None,
        "ALI_DOMAINS": "This sentence is extracted from a scientific paper. When translating, please pay close attention to the use of specialized troubleshooting terminologies and adhere to scientific sentence structures to maintain the technical rigor and precision of the original text.",
    }
    CustomPrompt = True

    def __init__(
        self, lang_in, lang_out, model, envs=None, prompt=None, ignore_cache=False
    ):
        self.set_envs(envs)
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        api_key = self.envs["ALI_API_KEY"]

        if not model:
            model = self.envs["ALI_MODEL"]

        super().__init__(
            lang_in,
            lang_out,
            model,
            base_url=base_url,
            api_key=api_key,
            ignore_cache=ignore_cache,
        )
        self.prompttext = prompt

    @staticmethod
    def lang_mapping(input_lang: str) -> str:
        """
        Mapping the language code to the language code that Aliyun Qwen-Mt model supports.
        Since all existings languagues codes used in gui.py are able to be mapped, the original
        languague code will not be checked.
        """
        langdict = {
            "zh": "Chinese",
            "zh-TW": "Chinese",
            "en": "English",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "ru": "Russian",
            "es": "Spanish",
            "it": "Italian",
        }

        return langdict[input_lang]

    def do_translate(self, text) -> str:
        """
        Qwen-MT Model reqeust to send translation_options to the server.
        domains are options, but suggested. it must be in English.
        """
        translation_options = {
            "source_lang": self.lang_mapping(self.lang_in),
            "target_lang": self.lang_mapping(self.lang_out),
            "domains": self.envs["ALI_DOMAINS"],
        }
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=[{"role": "user", "content": text}],
            extra_body={"translation_options": translation_options},
        )
        return response.choices[0].message.content.strip()


class IndicTransTranslator(BaseTranslator):
    """
    Translator using IndicTransToolkit for Indic language translation.
    Supports translation between English and Indic languages (Telugu, Kannada, Hindi, Tamil, etc.)
    """
    name = "indic-trans"
    envs = {
        "INDICTRANS_MODEL_EN_INDIC": "ai4bharat/indictrans2-en-indic-1B",  # English -> Indic (1B)
        "INDICTRANS_MODEL_INDIC_EN": "ai4bharat/indictrans2-indic-en-1B",  # Indic -> English (1B)
        "INDICTRANS_MODEL": "ai4bharat/indictrans2-indic-en-1B",  # Default
    }
    lang_map = {
        "en": "eng_Latn",
        "eng": "eng_Latn",
        "te": "tel_Telu",
        "tel": "tel_Telu",
        "telugu": "tel_Telu",
        "kn": "kan_Knda",
        "kan": "kan_Knda",
        "kannada": "kan_Knda",
        "hi": "hin_Deva",
        "hin": "hin_Deva",
        "hindi": "hin_Deva",
        "ta": "tam_Taml",
        "tam": "tam_Taml",
        "tamil": "tam_Taml",
        "mr": "mar_Deva",
        "mar": "mar_Deva",
        "marathi": "mar_Deva",
        "gu": "guj_Gujr",
        "guj": "guj_Gujr",
        "gujarati": "guj_Gujr",
        "bn": "ben_Beng",
        "ben": "ben_Beng",
        "bengali": "ben_Beng",
        "or": "ori_Orya",
        "ori": "ori_Orya",
        "odia": "ori_Orya",
        "pa": "pan_Guru",
        "pan": "pan_Guru",
        "punjabi": "pan_Guru",
        "ml": "mal_Mlym",
        "mal": "mal_Mlym",
        "malayalam": "mal_Mlym",
        "as": "asm_Beng",
        "asm": "asm_Beng",
        "assamese": "asm_Beng",
    }

    def __init__(self, lang_in, lang_out, model=None, ignore_cache=False, **kwargs):
        # Initialize envs from class variable first (before set_envs)
        self.envs = copy(self.__class__.envs)
        self.set_envs(kwargs.get("envs", None))
        
        if not INDIC_TRANS_AVAILABLE:
            raise ImportError(
                "IndicTransToolkit not installed. Please install it:\n"
                "git clone https://github.com/VarunGumma/IndicTransToolkit.git\n"
                "cd IndicTransToolkit\n"
                "pip install --editable ./\n"
                "Also ensure PyTorch is installed."
            )
        
        # Convert language codes to IndicTrans format (need to do this before super().__init__)
        lang_in_normalized = self.lang_map.get(lang_in.lower(), lang_in)
        lang_out_normalized = self.lang_map.get(lang_out.lower(), lang_out)
        
        # Choose correct 1B model based on direction BEFORE calling super().__init__()
        is_english_to_indic = lang_in_normalized == "eng_Latn" and lang_out_normalized != "eng_Latn"
        is_indic_to_english = lang_in_normalized != "eng_Latn" and lang_out_normalized == "eng_Latn"
        
        # Select the appropriate 1B model
        # Handle None, empty string, or default model
        if not model or model == "" or model == self.envs.get("INDICTRANS_MODEL", ""):
            if is_english_to_indic:
                model = self.envs.get("INDICTRANS_MODEL_EN_INDIC", "ai4bharat/indictrans2-en-indic-1B")
                logging.info("Using English->Indic 1B model: indictrans2-en-indic-1B")
            elif is_indic_to_english:
                model = self.envs.get("INDICTRANS_MODEL_INDIC_EN", "ai4bharat/indictrans2-indic-en-1B")
                logging.info("Using Indic->English 1B model: indictrans2-indic-en-1B")
            else:
                # Both Indic languages - use English->Indic model
                model = self.envs.get("INDICTRANS_MODEL_EN_INDIC", "ai4bharat/indictrans2-en-indic-1B")
                logging.info("Using English->Indic 1B model for Indic->Indic")
        
        # Ensure model is set (safety check)
        if not model or model == "":
            # Fallback to hardcoded default
            model = "ai4bharat/indictrans2-en-indic-1B"
            logging.warning(f"Model was empty, using hardcoded default: {model}")
        
        # Now call super().__init__() with the correct model
        super().__init__(lang_in, lang_out, model, ignore_cache)
        
        # Store normalized language codes for later use
        self.src_lang = lang_in_normalized
        self.tgt_lang = lang_out_normalized
        self.is_english_to_indic = is_english_to_indic
        self.is_indic_to_english = is_indic_to_english
        
        # Ensure self.model is set (it should be from super().__init__())
        if not self.model or self.model == "":
            self.model = model  # Use the model we calculated
        
        # Safety check - if model is still empty, use default
        if not self.model or self.model == "":
            self.model = self.envs.get("INDICTRANS_MODEL_EN_INDIC", "ai4bharat/indictrans2-en-indic-1B")
            if not self.model or self.model == "":
                self.model = "ai4bharat/indictrans2-en-indic-1B"  # Hardcoded fallback
            logging.error(f"Model was still empty after super().__init__(), forcing to: {self.model}")
        
        # Initialize model and processor
        logging.info(f"Loading IndicTrans 1B model: {self.model}")
        logging.info(f"Translation direction: {self.src_lang} -> {self.tgt_lang}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            # Use torch_dtype="auto" for better performance (as suggested)
            self.model_instance = AutoModelForSeq2SeqLM.from_pretrained(
                self.model, 
                trust_remote_code=True,
                torch_dtype="auto"
            )
            self.model_instance = self.model_instance.to(self.device)
            self.model_instance.eval()
            self.ip = IndicProcessor(inference=True)
            logging.info(f"✓ IndicTrans 1B model loaded successfully")
            logging.info(f"  Model: {self.model}")
            logging.info(f"  Direction: {self.src_lang} -> {self.tgt_lang}")
            logging.info(f"  Device: {self.device}")
        except Exception as e:
            logging.error(f"Failed to load IndicTrans model: {e}")
            raise

    def do_translate(self, text: str) -> str:
        """
        Translate text using IndicTrans, preserving formula placeholders like {v1}, {v2}, etc.
        """
        if not text or not text.strip():
            return text
        
        # Extract and protect placeholders (formula markers)
        placeholder_pattern = r'\{[vV]\d+\}|<b\d+>|</b\d+>'
        placeholders = re.findall(placeholder_pattern, text)
        
        # Replace placeholders with temporary markers
        temp_text = text
        placeholder_map = {}
        for i, ph in enumerate(placeholders):
            temp_marker = f"__PLACEHOLDER_{i}__"
            placeholder_map[temp_marker] = ph
            temp_text = temp_text.replace(ph, temp_marker, 1)
        
        try:
            # Preprocess with correct language codes
            # Now we have the right model for each direction (1B models)
            text_len = len(temp_text)
            # Only log if text is significant (avoid spam for tiny chunks)
            if text_len > 10:
                logging.info(f"Translating text chunk (length: {text_len} chars)...")
            batch = self.ip.preprocess_batch(
                [temp_text],
                src_lang=self.src_lang,
                tgt_lang=self.tgt_lang,
            )
            
            logging.debug(f"Preprocessed batch: {batch}")
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.device)
            
            # Generate translation
            # Optimize for speed: use greedy decoding (num_beams=1) for faster inference
            # On CPU, beam search with num_beams=5 is very slow
            input_len = inputs['input_ids'].shape[1]
            # Use shorter max_length for faster generation on CPU
            max_len = min(128, input_len + 20)  # Cap at 128 tokens for speed
            if text_len > 10:  # Only log for significant chunks
                logging.debug(f"Generating translation (input: {input_len} tokens, max: {max_len})...")
            with torch.no_grad():
                generated_tokens = self.model_instance.generate(
                    **inputs,
                    use_cache=True,
                    min_length=0,
                    max_length=max_len,  # Reduced from 256 for speed
                    num_beams=1,  # Use greedy decoding for speed (was 5, too slow on CPU)
                    num_return_sequences=1,
                    do_sample=False,  # Deterministic output
                )
            if text_len > 10:
                logging.debug(f"Generated {generated_tokens.shape[1]} tokens")
            
            # Decode
            with self.tokenizer.as_target_tokenizer():
                generated_tokens = self.tokenizer.batch_decode(
                    generated_tokens.detach().cpu().tolist(),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            
            logging.debug(f"Decoded tokens: {generated_tokens}")
            
            # Postprocess with target language
            translations = self.ip.postprocess_batch(generated_tokens, lang=self.tgt_lang)
            translated_text = translations[0] if translations else temp_text
            
            if text_len > 10:  # Only log for significant chunks to reduce spam
                logging.info(f"Translation completed: {len(translated_text)} chars")
                preview = translated_text[:50] + "..." if len(translated_text) > 50 else translated_text
                logging.debug(f"Preview: '{preview}'")
            
            # Verify translation worked (output should be different from input)
            if translated_text.lower().strip() == temp_text.lower().strip():
                logging.warning(f"Translation returned same as input. This may indicate an issue.")
            
            # Restore placeholders
            for temp_marker, original_ph in placeholder_map.items():
                translated_text = translated_text.replace(temp_marker, original_ph)
            
            return remove_control_characters(translated_text)
        
        except Exception as e:
            logging.error(f"Translation failed: {e}", exc_info=True)
            # Fallback: restore placeholders and return original
            restored_text = temp_text
            for temp_marker, original_ph in placeholder_map.items():
                restored_text = restored_text.replace(temp_marker, original_ph)
            return restored_text
