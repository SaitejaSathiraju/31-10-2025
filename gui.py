import asyncio
import cgi
import os
import shutil
import uuid
from asyncio import CancelledError
from pathlib import Path
import typing as T

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template
import logging

from pdf2zh import __version__
from pdf2zh.high_level import translate
from pdf2zh.doclayout import ModelInstance
from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    AnythingLLMTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DifyTranslator,
    ArgosTranslator,
    GeminiTranslator,
    GoogleTranslator,
    IndicTransTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
    X302AITranslator,
)
from babeldoc.docvision.doclayout import OnnxModel
from babeldoc import __version__ as babeldoc_version

logger = logging.getLogger(__name__)

BABELDOC_MODEL = OnnxModel.load_available()
# The following variables associate strings with translators
service_map: dict[str, BaseTranslator] = {
    "IndicTrans": IndicTransTranslator,
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
    "302.AI": X302AITranslator,
}

# The following variables associate strings with specific languages
# Indic languages first for visibility!
lang_map = {
    # INDIC LANGUAGES (First!)
    "Telugu": "te",
    "Kannada": "kn",
    "Hindi": "hi",
    "Tamil": "ta",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Odia": "or",
    "Punjabi": "pa",
    "Malayalam": "ml",
    "Assamese": "as",
    # Other languages
    "English": "en",
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

# The following variable associate strings with page ranges
page_map = {
    "All": None,
    "First": [0],
    "First 5 pages": list(range(0, 5)),
    "Others": None,
}

# Check if this is a public demo, which has resource limits
flag_demo = False

# Limit resources
if ConfigManager.get("PDF2ZH_DEMO"):
    flag_demo = True
    service_map = {
        "IndicTrans": IndicTransTranslator,
    }
    page_map = {
        "First": [0],
        "First 20 pages": list(range(0, 20)),
    }
    client_key = ConfigManager.get("PDF2ZH_CLIENT_KEY")
    server_key = ConfigManager.get("PDF2ZH_SERVER_KEY")


# Limit Enabled Services
enabled_services: T.Optional[T.List[str]] = ConfigManager.get("ENABLED_SERVICES")
if isinstance(enabled_services, list):
    default_services = ["IndicTrans", "Google", "Bing"]  # Make IndicTrans first/default
    enabled_services_names = [str(_).lower().strip() for _ in enabled_services]
    enabled_services = [
        k
        for k in service_map.keys()
        if str(k).lower().strip() in enabled_services_names
    ]
    if len(enabled_services) == 0:
        raise RuntimeError("No services available.")
    # Ensure IndicTrans is included
    if "IndicTrans" not in enabled_services and "IndicTrans" in service_map:
        enabled_services = ["IndicTrans"] + enabled_services
    enabled_services = default_services + [s for s in enabled_services if s not in default_services]
else:
    enabled_services = list(service_map.keys())
    # Ensure IndicTrans is first
    if "IndicTrans" in enabled_services:
        enabled_services.remove("IndicTrans")
        enabled_services.insert(0, "IndicTrans")


# Configure about Gradio show keys
hidden_gradio_details: bool = bool(ConfigManager.get("HIDDEN_GRADIO_DETAILS"))


# Public demo control
def verify_recaptcha(response):
    """
    This function verifies the reCAPTCHA response.
    """
    recaptcha_url = "https://www.google.com/recaptcha/api/siteverify"
    data = {"secret": server_key, "response": response}
    result = requests.post(recaptcha_url, data=data).json()
    return result.get("success")


def download_with_limit(url: str, save_path: str, size_limit: int) -> str:
    """
    This function downloads a file from a URL and saves it to a specified path.

    Inputs:
        - url: The URL to download the file from
        - save_path: The path to save the file to
        - size_limit: The maximum size of the file to download

    Returns:
        - The path of the downloaded file
    """
    chunk_size = 1024
    total_size = 0
    with requests.get(url, stream=True, timeout=10) as response:
        response.raise_for_status()
        content = response.headers.get("Content-Disposition")
        try:  # filename from header
            _, params = cgi.parse_header(content)
            filename = params["filename"]
        except Exception:  # filename from url
            filename = os.path.basename(url)
        filename = os.path.splitext(os.path.basename(filename))[0] + ".pdf"
        with open(save_path / filename, "wb") as file:
            for chunk in response.iter_content(chunk_size=chunk_size):
                total_size += len(chunk)
                if size_limit and total_size > size_limit:
                    raise gr.Error("Exceeds file size limit")
                file.write(chunk)
    return save_path / filename


def stop_translate_file(state: dict) -> None:
    """
    This function stops the translation process.

    Inputs:
        - state: The state of the translation process

    Returns:- None
    """
    session_id = state["session_id"]
    if session_id is None:
        return
    if session_id in cancellation_event_map:
        logger.info(f"Stopping translation for session {session_id}")
        cancellation_event_map[session_id].set()


def translate_file(
    file_type,
    file_input,
    link_input,
    service,
    lang_from,
    lang_to,
    page_range,
    page_input,
    prompt,
    threads,
    skip_subset_fonts,
    ignore_cache,
    vfont,
    use_babeldoc,
    recaptcha_response,
    state,
    progress=gr.Progress(),
    *envs,
):
    """
    This function translates a PDF file from one language to another.

    Inputs:
        - file_type: The type of file to translate
        - file_input: The file to translate
        - link_input: The link to the file to translate
        - service: The translation service to use
        - lang_from: The language to translate from
        - lang_to: The language to translate to
        - page_range: The range of pages to translate
        - page_input: The input for the page range
        - prompt: The custom prompt for the llm
        - threads: The number of threads to use
        - recaptcha_response: The reCAPTCHA response
        - state: The state of the translation process
        - progress: The progress bar
        - envs: The environment variables

    Returns:
        - The translated file
        - The translated file
        - The translated file
        - The progress bar
        - The progress bar
        - The progress bar
    """
    session_id = uuid.uuid4()
    state["session_id"] = session_id
    cancellation_event_map[session_id] = asyncio.Event()
    # Translate PDF content using selected service.
    if flag_demo and not verify_recaptcha(recaptcha_response):
        raise gr.Error("reCAPTCHA fail")

    progress(0, desc="Starting translation...")

    output = Path("pdf2zh_files")
    output.mkdir(parents=True, exist_ok=True)

    if file_type == "File":
        if not file_input:
            raise gr.Error("No input")
        file_path = shutil.copy(file_input, output)
    else:
        if not link_input:
            raise gr.Error("No input")
        file_path = download_with_limit(
            link_input,
            output,
            5 * 1024 * 1024 if flag_demo else None,
        )

    filename = os.path.splitext(os.path.basename(file_path))[0]
    file_raw = output / f"{filename}.pdf"
    file_mono = output / f"{filename}-mono.pdf"
    file_dual = output / f"{filename}-dual.pdf"

    translator = service_map[service]
    logger.info(f"Selected translation service: {service} -> {translator.name}")
    logger.info(f"Translating from: {lang_from} to: {lang_to}")
    
    if page_range != "Others":
        selected_page = page_map[page_range]
    else:
        selected_page = []
        for p in page_input.split(","):
            if "-" in p:
                start, end = p.split("-")
                selected_page.extend(range(int(start) - 1, int(end)))
            else:
                selected_page.append(int(p) - 1)
    lang_from = lang_map[lang_from]
    lang_to = lang_map[lang_to]
    
    logger.info(f"Using translator: {translator.name} with language codes: {lang_from} -> {lang_to}")

    _envs = {}
    for i, env in enumerate(translator.envs.items()):
        _envs[env[0]] = envs[i]
    for k, v in _envs.items():
        if str(k).upper().endswith("API_KEY") and str(v) == "***":
            # Load Real API_KEYs from local configure file
            real_keys: str = ConfigManager.get_env_by_translatername(
                translator, k, None
            )
            _envs[k] = real_keys

    print(f"Files before translation: {os.listdir(output)}")

    def progress_bar(t: tqdm.tqdm):
        desc = getattr(t, "desc", "Translating...")
        if desc == "":
            desc = "Translating..."
        progress(t.n / t.total, desc=desc)

    try:
        threads = int(threads)
    except (ValueError, TypeError):
        threads = 4

    param = {
        "files": [str(file_raw)],
        "pages": selected_page,
        "lang_in": lang_from,
        "lang_out": lang_to,
        "service": f"{translator.name}",  # This uses translator.name ("indic-trans") not service map key
        "output": output,
        "thread": int(threads),
        "callback": progress_bar,
        "cancellation_event": cancellation_event_map[session_id],
        "envs": _envs,
        "prompt": Template(prompt) if prompt else None,
        "skip_subset_fonts": True,  # Skip font subsetting - it's very slow on CPU and causes hanging
        "ignore_cache": ignore_cache,
        "vfont": vfont,  # 添加自定义公式字体正则表达式
        "model": ModelInstance.value,
    }
    
    logger.info(f"Translation parameters: service={param['service']}, lang_in={param['lang_in']}, lang_out={param['lang_out']}")

    try:
        progress(0.1, desc="Starting translation...")
        if use_babeldoc:
            return babeldoc_translate_file(**param)
        progress(0.3, desc="Loading PDF and analyzing content...")
        progress(0.5, desc="Translating text content (this may take a while on CPU)...")
        
        # Call translate - it will report progress via callback
        logger.info("Starting translate() function call...")
        translate(**param)
        logger.info("translate() function returned successfully")
        
        progress(0.85, desc="Translation complete! Rendering PDF...")
        
    except CancelledError:
        del cancellation_event_map[session_id]
        raise gr.Error("Translation cancelled")
    except Exception as e:
        logger.error(f"Translation error: {e}", exc_info=True)
        progress(1.0, desc=f"Error: {str(e)[:100]}")
        raise gr.Error(f"Translation failed: {str(e)}")
    
    # Check if files exist
    import time
    logger.info("Checking for output files...")
    logger.info(f"Looking for: {file_mono.name} and {file_dual.name}")
    logger.info(f"In directory: {output}")
    
    max_wait = 30  # Wait up to 30 seconds for files
    waited = 0
    while waited < max_wait:
        mono_exists = file_mono.exists()
        dual_exists = file_dual.exists()
        if mono_exists and dual_exists:
            logger.info("✓ Both PDF files found!")
            break
        if waited % 5 == 0:  # Log every 5 seconds
            logger.info(f"Waiting for files... ({waited:.0f}s) mono={mono_exists}, dual={dual_exists}")
        time.sleep(0.5)
        waited += 0.5
        progress(0.85 + (waited/max_wait) * 0.1, desc=f"Waiting for PDF files... ({waited:.0f}s)")
    
    files_in_dir = os.listdir(output)
    logger.info(f"Files after translation: {files_in_dir}")
    print(f"Files after translation: {files_in_dir}")

    if not file_mono.exists() or not file_dual.exists():
        progress(1.0, desc="Translation failed - no output generated")
        raise gr.Error("No output - files not found after translation")
    
    progress(0.95, desc="Finalizing output files...")

    progress(1.0, desc="✓ Translation complete! Files ready.")

    return (
        str(file_mono),
        str(file_mono),
        str(file_dual),
        gr.update(visible=True),
        gr.update(visible=True),
        gr.update(visible=True),
    )


def babeldoc_translate_file(**kwargs):
    from babeldoc.high_level import init as babeldoc_init

    babeldoc_init()
    from babeldoc.high_level import async_translate as babeldoc_translate
    from babeldoc.translation_config import TranslationConfig as YadtConfig

    for translator in [
        IndicTransTranslator,
        GoogleTranslator,
        BingTranslator,
        DeepLTranslator,
        DeepLXTranslator,
        OllamaTranslator,
        XinferenceTranslator,
        AzureOpenAITranslator,
        OpenAITranslator,
        ZhipuTranslator,
        ModelScopeTranslator,
        SiliconTranslator,
        GeminiTranslator,
        AzureTranslator,
        TencentTranslator,
        DifyTranslator,
        AnythingLLMTranslator,
        ArgosTranslator,
        GrokTranslator,
        GroqTranslator,
        DeepseekTranslator,
        OpenAIlikedTranslator,
        QwenMtTranslator,
        X302AITranslator,
    ]:
        if kwargs["service"] == translator.name:
            translator = translator(
                kwargs["lang_in"],
                kwargs["lang_out"],
                "",
                envs=kwargs["envs"],
                prompt=kwargs["prompt"],
                ignore_cache=kwargs["ignore_cache"],
            )
            break
    else:
        raise ValueError("Unsupported translation service")
    import asyncio
    from babeldoc.main import create_progress_handler

    for file in kwargs["files"]:
        file = file.strip("\"'")
        yadt_config = YadtConfig(
            input_file=file,
            font=None,
            pages=",".join((str(x) for x in getattr(kwargs, "raw_pages", []))),
            output_dir=kwargs["output"],
            doc_layout_model=BABELDOC_MODEL,
            translator=translator,
            debug=False,
            lang_in=kwargs["lang_in"],
            lang_out=kwargs["lang_out"],
            no_dual=False,
            no_mono=False,
            qps=kwargs["thread"],
            use_rich_pbar=False,
            disable_rich_text_translate=not isinstance(translator, OpenAITranslator),
            skip_clean=kwargs["skip_subset_fonts"],
            report_interval=0.5,
        )

        async def yadt_translate_coro(yadt_config):
            progress_context, progress_handler = create_progress_handler(yadt_config)

            # 开始翻译
            with progress_context:
                async for event in babeldoc_translate(yadt_config):
                    progress_handler(event)
                    if yadt_config.debug:
                        logger.debug(event)
                    kwargs["callback"](progress_context)
                    if kwargs["cancellation_event"].is_set():
                        yadt_config.cancel_translation()
                        raise CancelledError
                    if event["type"] == "finish":
                        result = event["translate_result"]
                        logger.info("Translation Result:")
                        logger.info(f"  Original PDF: {result.original_pdf_path}")
                        logger.info(f"  Time Cost: {result.total_seconds:.2f}s")
                        logger.info(f"  Mono PDF: {result.mono_pdf_path or 'None'}")
                        logger.info(f"  Dual PDF: {result.dual_pdf_path or 'None'}")
                        file_mono = result.mono_pdf_path
                        file_dual = result.dual_pdf_path
                        break
            import gc

            gc.collect()
            return (
                str(file_mono),
                str(file_mono),
                str(file_dual),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )

        return asyncio.run(yadt_translate_coro(yadt_config))


# Global setup
custom_blue = gr.themes.Color(
    c50="#E8F3FF",
    c100="#BEDAFF",
    c200="#94BFFF",
    c300="#6AA1FF",
    c400="#4080FF",
    c500="#165DFF",  # Primary color
    c600="#0E42D2",
    c700="#0A2BA6",
    c800="#061D79",
    c900="#03114D",
    c950="#020B33",
)

custom_css = """
    .secondary-text {color: #999 !important;}
    footer {visibility: hidden}
    .env-warning {color: #dd5500 !important;}
    .env-success {color: #559900 !important;}
    .service-info {
        padding: 10px;
        background-color: #f0f8ff;
        border-radius: 6px;
        border-left: 4px solid #165DFF;
        margin: 10px 0;
    }
    
    /* Language button styling */
    button[variant="primary"] {
        font-size: 14px !important;
        padding: 8px 12px !important;
        margin: 4px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
    }
    
    button[variant="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(22, 93, 255, 0.3);
    }

    /* Add dashed border to input-file class */
    .input-file {
        border: 1.2px dashed #165DFF !important;
        border-radius: 6px !important;
    }

    .progress-bar-wrap {
        border-radius: 8px !important;
    }

    .progress-bar {
        border-radius: 8px !important;
    }

    .pdf-canvas canvas {
        width: 100%;
    }
    """

demo_recaptcha = """
    <script src="https://www.google.com/recaptcha/api.js?render=explicit" async defer></script>
    <script type="text/javascript">
        var onVerify = function(token) {
            el=document.getElementById('verify').getElementsByTagName('textarea')[0];
            el.value=token;
            el.dispatchEvent(new Event('input'));
        };
    </script>
    """

tech_details_string = f"""
                    <summary>Technical details</summary>
                    - GitHub: <a href="https://github.com/Byaidu/PDFMathTranslate">Byaidu/PDFMathTranslate</a><br>
                    - BabelDOC: <a href="https://github.com/funstory-ai/BabelDOC">funstory-ai/BabelDOC</a><br>
                    - GUI by: <a href="https://github.com/reycn">Rongxin</a><br>
                    - pdf2zh Version: {__version__} <br>
                    - BabelDOC Version: {babeldoc_version}
                """
cancellation_event_map = {}


# The following code creates the GUI
with gr.Blocks(
    title="PDFMathTranslate - PDF Translation with preserved formats",
    theme=gr.themes.Default(
        primary_hue=custom_blue, spacing_size="md", radius_size="lg"
    ),
    css=custom_css,
    head=demo_recaptcha if flag_demo else "",
) as demo:
    gr.Markdown(
        """
        # 🌐 PDFMathTranslate - Indic Language Translation
        
        Translate PDFs with exact layout preservation using **IndicTrans** for Indic languages.
        
        ### 🇮🇳 Supported Indic Languages:
        **Telugu (తెలుగు) | Kannada (ಕನ್ನಡ) | Hindi (हिंदी) | Tamil (தமிழ்) | Marathi (मराठी) | Gujarati (ગુજરાતી) | Bengali (বাংলা) | Odia (ଓଡ଼ିଆ) | Punjabi (ਪੰਜਾਬੀ) | Malayalam (മലയാളം) | Assamese (অসমীয়া)**
        
        ### 📝 Translation Services:
        - **✅ IndicTrans** (Default): Runs locally, NO API key needed, best quality for Indic languages
        - **🌐 Google/Bing**: Free web translation (requires internet, no API key)
        - **🔑 OpenAI/Gemini**: Require API keys
        
        **💡 Tip:** All Indic languages are available in the language dropdown - scroll down to see them!
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## File | < 5 MB" if flag_demo else "## File")
            file_type = gr.Radio(
                choices=["File", "Link"],
                label="Type",
                value="File",
            )
            file_input = gr.File(
                label="File",
                file_count="single",
                file_types=[".pdf"],
                type="filepath",
                elem_classes=["input-file"],
            )
            link_input = gr.Textbox(
                label="Link",
                visible=False,
                interactive=True,
            )
            gr.Markdown("## Option")
            # FORCE IndicTrans to be first and default
            service_choices = enabled_services.copy()
            if "IndicTrans" in service_choices:
                service_choices.remove("IndicTrans")
                service_choices.insert(0, "IndicTrans")
            
            # Debug: Log what we're creating
            logger.info(f"=== SERVICE DROPDOWN DEBUG ===")
            logger.info(f"service_choices: {service_choices[:5]}")
            logger.info(f"IndicTrans in choices: {'IndicTrans' in service_choices}")
            logger.info(f"Default value: IndicTrans")
            
            # Force clear any cached state - ensure IndicTrans is visible
            service = gr.Dropdown(
                label="Translation Service",
                choices=service_choices,  # Already has IndicTrans first
                value="IndicTrans",  # FORCE IndicTrans as default, no conditional
                info="Select IndicTrans for Indic language translation",
                interactive=True,
                allow_custom_value=False,
            )
            envs = []
            for i in range(3):
                envs.append(
                    gr.Textbox(
                        visible=False,
                        interactive=True,
                    )
                )
            gr.Markdown("### 🌍 Language Selection")
            # FORCE Indic languages first in dropdown
            lang_choices = list(lang_map.keys())
            # Ensure Indic languages are at the top
            indic_langs = [k for k in lang_choices if k in ["Telugu", "Kannada", "Hindi", "Tamil", "Marathi", "Gujarati", "Bengali", "Odia", "Punjabi", "Malayalam", "Assamese"]]
            other_langs = [k for k in lang_choices if k not in indic_langs]
            lang_choices_ordered = indic_langs + other_langs
            
            # Debug: Log what we're creating
            logger.info(f"=== LANGUAGE DROPDOWN DEBUG ===")
            logger.info(f"First 15 lang choices: {lang_choices_ordered[:15]}")
            logger.info(f"Telugu in choices: {'Telugu' in lang_choices_ordered}")
            logger.info(f"Default lang_to value: Telugu")
            
            with gr.Row():
                lang_from = gr.Dropdown(
                    label="🔤 Translate from",
                    choices=lang_choices_ordered,
                    value="English" if "English" in lang_choices_ordered else lang_choices_ordered[0],
                    info="Select source language",
                    allow_custom_value=False,
                    scale=1,
                )
                lang_to = gr.Dropdown(
                    label="🎯 Translate to (Indic languages shown first)",
                    choices=lang_choices_ordered,  # Already has Indic languages first
                    value="Telugu",  # FORCE Telugu as default, no conditional
                    info="Default: Telugu. Indic languages appear first in dropdown!",
                    allow_custom_value=False,
                    scale=1,
                    interactive=True,
                )
            
            # Quick-select buttons for Indic languages
            gr.Markdown("### 🇮🇳 Quick Select Indic Languages:")
            
            # First row of buttons
            with gr.Row():
                btn_telugu = gr.Button("తెలుగు Telugu", variant="primary")
                btn_kannada = gr.Button("ಕನ್ನಡ Kannada", variant="primary")
                btn_hindi = gr.Button("हिंदी Hindi", variant="primary")
                btn_tamil = gr.Button("தமிழ் Tamil", variant="primary")
            
            # Second row of buttons
            with gr.Row():
                btn_marathi = gr.Button("मराठी Marathi", variant="primary")
                btn_gujarati = gr.Button("ગુજરાતી Gujarati", variant="primary")
                btn_bengali = gr.Button("বাংলা Bengali", variant="primary")
                btn_odia = gr.Button("ଓଡ଼ିଆ Odia", variant="primary")
            
            # Third row of buttons
            with gr.Row():
                btn_punjabi = gr.Button("ਪੰਜਾਬੀ Punjabi", variant="primary")
                btn_malayalam = gr.Button("മലയാളം Malayalam", variant="primary")
                btn_assamese = gr.Button("অসমীয়া Assamese", variant="primary")
                btn_english = gr.Button("English", variant="secondary")
            
            # Button click handlers - update lang_to dropdown
            def set_language(lang):
                return gr.update(value=lang)
            
            btn_telugu.click(fn=lambda: set_language("Telugu"), outputs=lang_to)
            btn_kannada.click(fn=lambda: set_language("Kannada"), outputs=lang_to)
            btn_hindi.click(fn=lambda: set_language("Hindi"), outputs=lang_to)
            btn_tamil.click(fn=lambda: set_language("Tamil"), outputs=lang_to)
            btn_marathi.click(fn=lambda: set_language("Marathi"), outputs=lang_to)
            btn_gujarati.click(fn=lambda: set_language("Gujarati"), outputs=lang_to)
            btn_bengali.click(fn=lambda: set_language("Bengali"), outputs=lang_to)
            btn_odia.click(fn=lambda: set_language("Odia"), outputs=lang_to)
            btn_punjabi.click(fn=lambda: set_language("Punjabi"), outputs=lang_to)
            btn_malayalam.click(fn=lambda: set_language("Malayalam"), outputs=lang_to)
            btn_assamese.click(fn=lambda: set_language("Assamese"), outputs=lang_to)
            btn_english.click(fn=lambda: set_language("English"), outputs=lang_to)
            page_range = gr.Radio(
                choices=page_map.keys(),
                label="Pages",
                value=list(page_map.keys())[0],
            )

            page_input = gr.Textbox(
                label="Page range",
                visible=False,
                interactive=True,
            )

            with gr.Accordion("Advanced Options", open=False):
                threads = gr.Slider(
                    label="Number of threads", 
                    minimum=1, 
                    maximum=8, 
                    value=4, 
                    step=1,
                    interactive=True
                )
                skip_subset_fonts = gr.Checkbox(
                    label="Skip font subsetting", interactive=True, value=False
                )
                ignore_cache = gr.Checkbox(
                    label="Ignore translation cache", interactive=True, value=False
                )
                vfont = gr.Textbox(
                    label="Custom formula font regex",
                    interactive=True,
                    value=ConfigManager.get("PDF2ZH_VFONT", ""),
                    visible=False
                )
                prompt = gr.Textbox(
                    label="Custom Prompt for LLM", interactive=True, visible=False
                )
                use_babeldoc = gr.Checkbox(
                    label="Use BabelDOC backend", interactive=True, value=False
                )
                envs.append(prompt)

            def on_select_service(service, evt: gr.EventData):
                translator = service_map[service]
                logger.info(f"Service changed to: {service} ({translator.name})")
                
                # Show info about the selected service
                service_info = []
                if service == "IndicTrans":
                    service_info.append("✓ **IndicTrans**: Runs locally, no API key needed")
                    service_info.append("✓ Best for Indic languages (Telugu, Kannada, Hindi, Tamil, etc.)")
                elif service == "Google":
                    service_info.append("⚠ **Google Translate**: Requires internet, no API key")
                    service_info.append("⚠ May not support all Indic languages")
                elif service == "Bing":
                    service_info.append("⚠ **Bing Translator**: Requires internet, no API key")
                else:
                    service_info.append(f"ℹ **{service}**: Check if API key is needed")
                
                _envs = []
                for i in range(4):
                    _envs.append(gr.update(visible=False, value=""))
                for i, env in enumerate(translator.envs.items()):
                    label = env[0]
                    value = ConfigManager.get_env_by_translatername(
                        translator, env[0], env[1]
                    )
                    visible = True
                    if hidden_gradio_details:
                        if (
                            "MODEL" not in str(label).upper()
                            and value
                            and hidden_gradio_details
                        ):
                            visible = False
                        # Hidden Keys From Gradio
                        if "API_KEY" in label.upper():
                            value = "***"  # We use "***" Present Real API_KEY
                    _envs[i] = gr.update(
                        visible=visible,
                        label=label,
                        value=value,
                    )
                _envs[-1] = gr.update(visible=translator.CustomPrompt)
                return _envs
            
            service_info_text = gr.Markdown(
                "ℹ **IndicTrans** selected: Runs locally, no API key needed",
                visible=True,
                elem_classes=["service-info"]
            )

            def on_select_filetype(file_type):
                return (
                    gr.update(visible=file_type == "File"),
                    gr.update(visible=file_type == "Link"),
                )

            def on_select_page(choice):
                if choice == "Others":
                    return gr.update(visible=True)
                else:
                    return gr.update(visible=False)

            def on_vfont_change(value):
                ConfigManager.set("PDF2ZH_VFONT", value)
                return value

            output_title = gr.Markdown("## Translated", visible=False)
            output_file_mono = gr.File(
                label="Download Translation (Mono)", visible=False
            )
            output_file_dual = gr.File(
                label="Download Translation (Dual)", visible=False
            )
            recaptcha_response = gr.Textbox(
                label="reCAPTCHA Response", elem_id="verify", visible=False
            )
            recaptcha_box = gr.HTML('<div id="recaptcha-box"></div>')
            translate_btn = gr.Button("Translate", variant="primary")
            cancellation_btn = gr.Button("Cancel", variant="secondary")
            tech_details_tog = gr.Markdown(
                tech_details_string,
                elem_classes=["secondary-text"],
            )
            page_range.select(on_select_page, page_range, page_input)
            service.select(
                on_select_service,
                service,
                envs,
            )
            vfont.change(on_vfont_change, inputs=vfont, outputs=None)
            file_type.select(
                on_select_filetype,
                file_type,
                [file_input, link_input],
                js=(
                    f"""
                    (a,b)=>{{
                        try{{
                            grecaptcha.render('recaptcha-box',{{
                                'sitekey':'{client_key}',
                                'callback':'onVerify'
                            }});
                        }}catch(error){{}}
                        return [a];
                    }}
                    """
                    if flag_demo
                    else ""
                ),
            )

        with gr.Column(scale=2):
            gr.Markdown("## Preview")
            preview = PDF(label="Document Preview", visible=True, height=2000)

    # Event handlers
    file_input.upload(
        lambda x: x,
        inputs=file_input,
        outputs=preview,
    )

    state = gr.State({"session_id": None})

    translate_btn.click(
        translate_file,
        inputs=[
            file_type,
            file_input,
            link_input,
            service,
            lang_from,
            lang_to,
            page_range,
            page_input,
            prompt,
            threads,
            skip_subset_fonts,
            ignore_cache,
            vfont,
            use_babeldoc,
            recaptcha_response,
            state,
            *envs,
        ],
        outputs=[
            output_file_mono,
            preview,
            output_file_dual,
            output_file_mono,
            output_file_dual,
            output_title,
        ],
    ).then(lambda: None, js="()=>{grecaptcha.reset()}" if flag_demo else "")

    cancellation_btn.click(
        stop_translate_file,
        inputs=[state],
    )


def parse_user_passwd(file_path: str) -> tuple:
    """
    Parse the user name and password from the file.

    Inputs:
        - file_path: The file path to read.
    Outputs:
        - tuple_list: The list of tuples of user name and password.
        - content: The content of the file
    """
    tuple_list = []
    content = ""
    if not file_path:
        return tuple_list, content
    if len(file_path) == 2:
        try:
            with open(file_path[1], "r", encoding="utf-8") as file:
                content = file.read()
        except FileNotFoundError:
            print(f"Error: File '{file_path[1]}' not found.")
    try:
        with open(file_path[0], "r", encoding="utf-8") as file:
            tuple_list = [
                tuple(line.strip().split(",")) for line in file if line.strip()
            ]
    except FileNotFoundError:
        print(f"Error: File '{file_path[0]}' not found.")
    return tuple_list, content


def setup_gui(
    share: bool = False, auth_file: list = ["", ""], server_port=7860
) -> None:
    """
    Setup the GUI with the given parameters.

    Inputs:
        - share: Whether to share the GUI.
        - auth_file: The file path to read the user name and password.

    Outputs:
        - None
    """
    user_list, html = parse_user_passwd(auth_file)
    
    # Debug output
    logger.info("=" * 50)
    logger.info("GUI STARTUP DEBUG INFO")
    logger.info(f"Enabled services: {enabled_services[:5]}")
    logger.info(f"IndicTrans available: {'IndicTrans' in enabled_services}")
    logger.info(f"First 10 languages: {list(lang_map.keys())[:10]}")
    logger.info("=" * 50)
    
    if flag_demo:
        demo.launch(server_name="0.0.0.0", max_file_size="5mb", inbrowser=True, show_error=True)
    else:
        if len(user_list) == 0:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True,
                    inbrowser=True,
                    share=share,
                    server_port=server_port,
                )
            except Exception:
                print(
                    "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True,
                        inbrowser=True,
                        share=share,
                        server_port=server_port,
                    )
                except Exception:
                    print(
                        "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                    )
                    demo.launch(
                        debug=True, inbrowser=True, share=True, server_port=server_port
                    )
        else:
            try:
                demo.launch(
                    server_name="0.0.0.0",
                    debug=True,
                    inbrowser=True,
                    share=share,
                    auth=user_list,
                    auth_message=html,
                    server_port=server_port,
                )
            except Exception:
                print(
                    "Error launching GUI using 0.0.0.0.\nThis may be caused by global mode of proxy software."
                )
                try:
                    demo.launch(
                        server_name="127.0.0.1",
                        debug=True,
                        inbrowser=True,
                        share=share,
                        auth=user_list,
                        auth_message=html,
                        server_port=server_port,
                    )
                except Exception:
                    print(
                        "Error launching GUI using 127.0.0.1.\nThis may be caused by global mode of proxy software."
                    )
                    demo.launch(
                        debug=True,
                        inbrowser=True,
                        share=True,
                        auth=user_list,
                        auth_message=html,
                        server_port=server_port,
                    )


# For auto-reloading while developing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    setup_gui()
