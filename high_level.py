"""Functions that can be used for the most common use-cases for pdf2zh.six"""

import asyncio
import io
import os
import re
import sys
import tempfile
import logging
from asyncio import CancelledError
from pathlib import Path
from string import Template
from typing import Any, BinaryIO, List, Optional, Dict

import numpy as np
import requests
import tqdm
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfexceptions import PDFValueError
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pymupdf import Document, Font

from pdf2zh.converter import TranslateConverter
from pdf2zh.doclayout import OnnxModel
from pdf2zh.pdfinterp import PDFPageInterpreterEx

from pdf2zh.config import ConfigManager
from babeldoc.assets.assets import get_font_and_metadata

NOTO_NAME = "noto"

logger = logging.getLogger(__name__)

noto_list = [
    "am",  # Amharic
    "ar",  # Arabic
    "bn",  # Bengali
    "bg",  # Bulgarian
    "chr",  # Cherokee
    "el",  # Greek
    "gu",  # Gujarati
    "iw",  # Hebrew
    "hi",  # Hindi
    "kn",  # Kannada
    "ml",  # Malayalam
    "mr",  # Marathi
    "ru",  # Russian
    "sr",  # Serbian
    "ta",  # Tamil
    "te",  # Telugu
    "th",  # Thai
    "ur",  # Urdu
    "uk",  # Ukrainian
]


def check_files(files: List[str]) -> List[str]:
    files = [
        f for f in files if not f.startswith("http://")
    ]  # exclude online files, http
    files = [
        f for f in files if not f.startswith("https://")
    ]  # exclude online files, https
    missing_files = [file for file in files if not os.path.exists(file)]
    return missing_files


def translate_patch(
    inf: BinaryIO,
    pages: Optional[list[int]] = None,
    vfont: str = "",
    vchar: str = "",
    thread: int = 0,
    doc_zh: Document = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    noto_name: str = "",
    noto: Font = None,
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    ignore_cache: bool = False,
    **kwarg: Any,
) -> None:
    rsrcmgr = PDFResourceManager()
    layout = {}
    device = TranslateConverter(
        rsrcmgr,
        vfont,
        vchar,
        thread,
        layout,
        lang_in,
        lang_out,
        service,
        noto_name,
        noto,
        envs,
        prompt,
        ignore_cache,
    )

    assert device is not None
    obj_patch = {}
    interpreter = PDFPageInterpreterEx(rsrcmgr, device, obj_patch)
    if pages:
        total_pages = len(pages)
    else:
        total_pages = doc_zh.page_count

    parser = PDFParser(inf)
    doc = PDFDocument(parser)
    with tqdm.tqdm(total=total_pages) as progress:
        for pageno, page in enumerate(PDFPage.create_pages(doc)):
            if cancellation_event and cancellation_event.is_set():
                raise CancelledError("task cancelled")
            if pages and (pageno not in pages):
                continue
            progress.update()
            if callback:
                callback(progress)
            page.pageno = pageno
            pix = doc_zh[page.pageno].get_pixmap()
            image = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )[:, :, ::-1]
            page_layout = model.predict(image, imgsz=int(pix.height / 32) * 32)[0]
            # kdtree 是不可能 kdtree 的，不如直接渲染成图片，用空间换时间
            box = np.ones((pix.height, pix.width))
            h, w = box.shape
            vcls = ["abandon", "figure", "table", "isolate_formula", "formula_caption"]
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] not in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = i + 2
            for i, d in enumerate(page_layout.boxes):
                if page_layout.names[int(d.cls)] in vcls:
                    x0, y0, x1, y1 = d.xyxy.squeeze()
                    x0, y0, x1, y1 = (
                        np.clip(int(x0 - 1), 0, w - 1),
                        np.clip(int(h - y1 - 1), 0, h - 1),
                        np.clip(int(x1 + 1), 0, w - 1),
                        np.clip(int(h - y0 + 1), 0, h - 1),
                    )
                    box[y0:y1, x0:x1] = 0
            layout[page.pageno] = box
            # 新建一个 xref 存放新指令流
            page.page_xref = doc_zh.get_new_xref()  # hack 插入页面的新 xref
            doc_zh.update_object(page.page_xref, "<<>>")
            doc_zh.update_stream(page.page_xref, b"")
            doc_zh[page.pageno].set_contents(page.page_xref)
            interpreter.process_page(page)

    device.close()
    return obj_patch


def translate_stream(
    stream: bytes,
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    **kwarg: Any,
):
    font_list = [("tiro", None)]

    font_path = download_remote_fonts(lang_out.lower())
    noto_name = NOTO_NAME
    noto = Font(noto_name, font_path)
    font_list.append((noto_name, font_path))

    doc_en = Document(stream=stream)
    stream = io.BytesIO()
    doc_en.save(stream)
    doc_zh = Document(stream=stream)
    page_count = doc_zh.page_count
    # font_list = [("GoNotoKurrent-Regular.ttf", font_path), ("tiro", None)]
    font_id = {}
    for page in doc_zh:
        for font in font_list:
            font_id[font[0]] = page.insert_font(font[0], font[1])
    xreflen = doc_zh.xref_length()
    for xref in range(1, xreflen):
        for label in ["Resources/", ""]:  # 可能是基于 xobj 的 res
            try:  # xref 读写可能出错
                font_res = doc_zh.xref_get_key(xref, f"{label}Font")
                target_key_prefix = f"{label}Font/"
                if font_res[0] == "xref":
                    resource_xref_id = re.search("(\\d+) 0 R", font_res[1]).group(1)
                    xref = int(resource_xref_id)
                    font_res = ("dict", doc_zh.xref_object(xref))
                    target_key_prefix = ""

                if font_res[0] == "dict":
                    for font in font_list:
                        target_key = f"{target_key_prefix}{font[0]}"
                        font_exist = doc_zh.xref_get_key(xref, target_key)
                        if font_exist[0] == "null":
                            doc_zh.xref_set_key(
                                xref,
                                target_key,
                                f"{font_id[font[0]]} 0 R",
                            )
            except Exception:
                pass

    fp = io.BytesIO()

    doc_zh.save(fp)
    obj_patch: dict = translate_patch(fp, **locals())

    for obj_id, ops_new in obj_patch.items():
        # ops_old=doc_en.xref_stream(obj_id)
        # print(obj_id)
        # print(ops_old)
        # print(ops_new.encode())
        doc_zh.update_stream(obj_id, ops_new.encode())

    doc_en.insert_file(doc_zh)
    for id in range(page_count):
        doc_en.move_page(page_count + id, id * 2 + 1)
    
    # Font subsetting can hang for hours on CPU - skip it by default
    import logging
    log = logging.getLogger(__name__)
    if not skip_subset_fonts:
        log.warning("⚠ Font subsetting enabled - this can take HOURS on CPU and may hang!")
        log.warning("⚠ Consider enabling 'Skip font subsetting' option in GUI for faster processing")
        try:
            log.info("Subsetting fonts for mono PDF... (this may take a very long time)")
            doc_zh.subset_fonts(fallback=True)
            log.info("Subsetting fonts for dual PDF... (this may take a very long time)")
            doc_en.subset_fonts(fallback=True)
            log.info("Font subsetting complete")
        except Exception as e:
            log.error(f"Font subsetting failed: {e} - continuing without it")
            # Continue without subsetting if it fails
    else:
        log.info("✓ Font subsetting skipped for faster processing (recommended for CPU)")
    
    log.info("Writing final PDF files...")
    return (
        doc_zh.write(deflate=True, garbage=3, use_objstms=1),
        doc_en.write(deflate=True, garbage=3, use_objstms=1),
    )


def convert_to_pdfa(input_path, output_path):
    """
    Convert PDF to PDF/A format

    Args:
        input_path: Path to source PDF file
        output_path: Path to save PDF/A file
    """
    from pikepdf import Dictionary, Name, Pdf

    # Open the PDF file
    pdf = Pdf.open(input_path)

    # Add PDF/A conformance metadata
    metadata = {
        "pdfa_part": "2",
        "pdfa_conformance": "B",
        "title": pdf.docinfo.get("/Title", ""),
        "author": pdf.docinfo.get("/Author", ""),
        "creator": "PDF Math Translate",
    }

    with pdf.open_metadata() as meta:
        meta.load_from_docinfo(pdf.docinfo)
        meta["pdfaid:part"] = metadata["pdfa_part"]
        meta["pdfaid:conformance"] = metadata["pdfa_conformance"]

    # Create OutputIntent dictionary
    output_intent = Dictionary(
        {
            "/Type": Name("/OutputIntent"),
            "/S": Name("/GTS_PDFA1"),
            "/OutputConditionIdentifier": "sRGB IEC61966-2.1",
            "/RegistryName": "http://www.color.org",
            "/Info": "sRGB IEC61966-2.1",
        }
    )

    # Add output intent to PDF root
    if "/OutputIntents" not in pdf.Root:
        pdf.Root.OutputIntents = [output_intent]
    else:
        pdf.Root.OutputIntents.append(output_intent)

    # Save as PDF/A
    pdf.save(output_path, linearize=True)
    pdf.close()


def translate(
    files: list[str],
    output: str = "",
    pages: Optional[list[int]] = None,
    lang_in: str = "",
    lang_out: str = "",
    service: str = "",
    thread: int = 0,
    vfont: str = "",
    vchar: str = "",
    callback: object = None,
    compatible: bool = False,
    cancellation_event: asyncio.Event = None,
    model: OnnxModel = None,
    envs: Dict = None,
    prompt: Template = None,
    skip_subset_fonts: bool = False,
    ignore_cache: bool = False,
    **kwarg: Any,
):
    if not files:
        raise PDFValueError("No files to process.")

    missing_files = check_files(files)

    if missing_files:
        print("The following files do not exist:", file=sys.stderr)
        for file in missing_files:
            print(f"  {file}", file=sys.stderr)
        raise PDFValueError("Some files do not exist.")

    result_files = []

    for file in files:
        if type(file) is str and (
            file.startswith("http://") or file.startswith("https://")
        ):
            print("Online files detected, downloading...")
            try:
                r = requests.get(file, allow_redirects=True)
                if r.status_code == 200:
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp_file:
                        print(f"Writing the file: {file}...")
                        tmp_file.write(r.content)
                        file = tmp_file.name
                else:
                    r.raise_for_status()
            except Exception as e:
                raise PDFValueError(
                    f"Errors occur in downloading the PDF file. Please check the link(s).\nError:\n{e}"
                )
        filename = os.path.splitext(os.path.basename(file))[0]

        # If the commandline has specified converting to PDF/A format
        # --compatible / -cp
        if compatible:
            with tempfile.NamedTemporaryFile(
                suffix="-pdfa.pdf", delete=False
            ) as tmp_pdfa:
                print(f"Converting {file} to PDF/A format...")
                convert_to_pdfa(file, tmp_pdfa.name)
                doc_raw = open(tmp_pdfa.name, "rb")
                os.unlink(tmp_pdfa.name)
        else:
            doc_raw = open(file, "rb")
        s_raw = doc_raw.read()
        doc_raw.close()

        temp_dir = Path(tempfile.gettempdir())
        file_path = Path(file)
        try:
            if file_path.exists() and file_path.resolve().is_relative_to(
                temp_dir.resolve()
            ):
                file_path.unlink(missing_ok=True)
                logger.debug(f"Cleaned temp file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean temp file {file_path}", exc_info=True)

        s_mono, s_dual = translate_stream(
            s_raw,
            **locals(),
        )
        
        # Log that translation stream completed
        logger.info(f"✓ Translation stream completed for {filename}")
        logger.info("Writing PDF files to disk...")
        
        file_mono = Path(output) / f"{filename}-mono.pdf"
        file_dual = Path(output) / f"{filename}-dual.pdf"
        doc_mono = open(file_mono, "wb")
        doc_dual = open(file_dual, "wb")
        doc_mono.write(s_mono)
        doc_dual.write(s_dual)
        doc_mono.close()
        doc_dual.close()
        
        logger.info(f"✓ PDF files written: {file_mono.name}, {file_dual.name}")
        result_files.append((str(file_mono), str(file_dual)))

    return result_files


def download_remote_fonts(lang: str):
    lang = lang.lower()
    LANG_NAME_MAP = {
        **{la: "GoNotoKurrent-Regular.ttf" for la in noto_list},
        **{
            la: f"SourceHanSerif{region}-Regular.ttf"
            for region, langs in {
                "CN": ["zh-cn", "zh-hans", "zh"],
                "TW": ["zh-tw", "zh-hant"],
                "JP": ["ja"],
                "KR": ["ko"],
            }.items()
            for la in langs
        },
        # Indic languages - use Noto fonts which properly support diacritics
        "te": "NotoSansTelugu-Regular.ttf",  # Telugu
        "tel": "NotoSansTelugu-Regular.ttf",
        "telugu": "NotoSansTelugu-Regular.ttf",
        "kn": "NotoSansKannada-Regular.ttf",  # Kannada
        "kan": "NotoSansKannada-Regular.ttf",
        "kannada": "NotoSansKannada-Regular.ttf",
        "hi": "NotoSansDevanagari-Regular.ttf",  # Hindi
        "hin": "NotoSansDevanagari-Regular.ttf",
        "hindi": "NotoSansDevanagari-Regular.ttf",
        "ta": "NotoSansTamil-Regular.ttf",  # Tamil
        "tam": "NotoSansTamil-Regular.ttf",
        "tamil": "NotoSansTamil-Regular.ttf",
        "mr": "NotoSansDevanagari-Regular.ttf",  # Marathi (uses Devanagari)
        "mar": "NotoSansDevanagari-Regular.ttf",
        "marathi": "NotoSansDevanagari-Regular.ttf",
        "gu": "NotoSansGujarati-Regular.ttf",  # Gujarati
        "guj": "NotoSansGujarati-Regular.ttf",
        "gujarati": "NotoSansGujarati-Regular.ttf",
        "bn": "NotoSansBengali-Regular.ttf",  # Bengali
        "ben": "NotoSansBengali-Regular.ttf",
        "bengali": "NotoSansBengali-Regular.ttf",
        "or": "NotoSansOriya-Regular.ttf",  # Odia (Oriya)
        "odi": "NotoSansOriya-Regular.ttf",
        "odia": "NotoSansOriya-Regular.ttf",
        "pa": "NotoSansGurmukhi-Regular.ttf",  # Punjabi (Gurmukhi script)
        "pan": "NotoSansGurmukhi-Regular.ttf",
        "punjabi": "NotoSansGurmukhi-Regular.ttf",
        "ml": "NotoSansMalayalam-Regular.ttf",  # Malayalam
        "mal": "NotoSansMalayalam-Regular.ttf",
        "malayalam": "NotoSansMalayalam-Regular.ttf",
        "as": "NotoSansBengali-Regular.ttf",  # Assamese (uses Bengali script)
        "asm": "NotoSansBengali-Regular.ttf",
        "assamese": "NotoSansBengali-Regular.ttf",
    }
    font_name = LANG_NAME_MAP.get(lang, "GoNotoKurrent-Regular.ttf")
    
    # Log font selection for Indic languages
    if lang in ["te", "kn", "hi", "ta", "mr", "gu", "bn", "or", "pa", "ml", "as"]:
        logger.info(f"Selected Indic font for {lang}: {font_name}")

    # Check for local fonts first (for Indic languages)
    # Prioritize Anek Telugu as it typically has better diacritic support
    font_path = None
    local_font_paths = {
        "te": [
            # Anek Telugu first (better diacritic support)
            r"C:\Users\airot\Downloads\Anek_Telugu,Noto_Sans_Telugu (1)\Anek_Telugu\static\AnekTelugu-Regular.ttf",
            r"C:\Users\airot\Downloads\Anek_Telugu,Noto_Sans_Telugu (1)\Anek_Telugu\AnekTelugu-VariableFont_wdth,wght.ttf",
            # Noto Sans Telugu as fallback
            r"C:\Users\airot\Downloads\Anek_Telugu,Noto_Sans_Telugu (1)\Noto_Sans_Telugu\static\NotoSansTelugu-Regular.ttf",
            r"C:\Users\airot\Downloads\Anek_Telugu,Noto_Sans_Telugu (1)\Noto_Sans_Telugu\NotoSansTelugu-VariableFont_wdth,wght.ttf",
        ],
    }
    
    # Try local fonts first - prioritize fonts with better diacritic support
    if lang == "te" and "te" in local_font_paths:
        for local_path in local_font_paths["te"]:
            if Path(local_path).exists():
                font_name_display = Path(local_path).name
                logger.info(f"✓ Using local Telugu font: {font_name_display} (full path: {local_path})")
                font_path = local_path
                break
        if not font_path:
            logger.info("Local Telugu fonts not found, will try system fonts...")
    
    # If no local font, try downloading from system
    if not font_path or not Path(font_path).exists():
        docker_path = ConfigManager.get("NOTO_FONT_PATH", Path("/app", font_name).as_posix())
        if Path(docker_path).exists():
            font_path = docker_path
            logger.info(f"Using configured font path: {font_path}")
        else:
            logger.info(f"Attempting to get Indic font: {font_name}...")
            try:
                result = get_font_and_metadata(font_name)
                if result is not None and len(result) >= 2:
                    font_path, _ = result
                    font_path = font_path.as_posix() if font_path else None
                    if font_path and Path(font_path).exists():
                        logger.info(f"✓ Indic font found: {font_path}")
                    else:
                        raise ValueError(f"Font path invalid: {font_path}")
                else:
                    raise ValueError(f"Font {font_name} not available in registry")
            except Exception as e:
                logger.warning(f"⚠ Could not get specific Indic font '{font_name}': {e}")
                # Fallback strategy for Indic languages
                if lang in ["te", "kn", "hi", "ta", "mr", "gu", "bn", "or", "pa", "ml", "as"]:
                    # Try generic NotoSans which supports multiple Indic scripts
                    logger.info("Trying fallback: NotoSans-Regular.ttf (supports multiple Indic scripts)")
                    try:
                        result = get_font_and_metadata("NotoSans-Regular.ttf")
                        if result is not None and len(result) >= 2 and result[0]:
                            font_path = result[0].as_posix()
                            if Path(font_path).exists():
                                logger.info(f"✓ Using fallback font: {font_path}")
                            else:
                                raise ValueError("Fallback font path invalid")
                        else:
                            raise ValueError("Fallback font not available")
                    except Exception as e2:
                        logger.warning(f"⚠ Fallback font also failed: {e2}")
                        # Last resort: use GoNotoKurrent (but warn it may not support diacritics)
                        logger.warning("⚠ Using GoNotoKurrent-Regular.ttf as last resort (may not support all Indic diacritics)")
                        try:
                            font_path = download_remote_fonts("en")  # Use default font
                        except Exception:
                            # Absolute last resort - use a hardcoded path
                            logger.error("All font loading methods failed!")
                            raise ValueError("Could not load any font for translation")
                else:
                    # For non-Indic languages, use default
                    font_path = download_remote_fonts("en")

    # Ensure font_path is set
    if not font_path:
        logger.error(f"Font path not set for language: {lang}")
        raise ValueError(f"Could not determine font path for language: {lang}")
    
    if not Path(font_path).exists():
        logger.error(f"Font file does not exist: {font_path}")
        raise ValueError(f"Font file does not exist: {font_path}")

    logger.info(f"Using font for {lang}: {font_path}")

    return font_path
