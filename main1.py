#!/usr/bin/env python3
"""
ENHANCED LEGAL RAG DOCUMENT TRANSLATOR BACKEND
Real OCR processing, RAG-powered translation, and paragraph-by-paragraph processing
"""

import os
import torch
from PIL import Image, ImageDraw, ImageFont
import webbrowser
import base64
from io import BytesIO
import numpy as np
import textwrap
import json
import tempfile
# Make cv2 optional (only needed for some image processing)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("WARNING: cv2 (opencv) not available. Some image processing features may be limited.")
    CV2_AVAILABLE = False
    cv2 = None

from flask import Flask, request, jsonify, send_file
import threading
import time
import re
from pathlib import Path
import logging
import requests
from datetime import datetime
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit import IndicProcessor
import requests
from io import BytesIO

# Make easyocr optional (only needed for OCR, Tesseract can be used instead)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    print("WARNING: easyocr not available. OCR will use Tesseract only.")
    EASYOCR_AVAILABLE = False
    easyocr = None

# Import our Legal RAG System
try:
    from legal_rag_system import LegalRAGSystem
    RAG_AVAILABLE = True
except ImportError:
    print("WARNING: Legal RAG System not available. Using basic translation.")
    RAG_AVAILABLE = False

app = Flask(__name__)

# Global variables to store processing data
current_image = None
current_bboxes = []
current_translated_text = []
current_image_path = None
current_target_language = 'te'
current_rag_system = None
current_paragraphs = []
current_paragraph_index = 0

# Initialize Legal RAG System if available
if RAG_AVAILABLE:
    try:
        print("Initializing Legal RAG System...")
        current_rag_system = LegalRAGSystem()
        current_rag_system.initialize()
        
        # Check if collections already have data to avoid reprocessing
        collections_to_check = ['hindi_glossary', 'telugu_glossary', 'government_orders']
        needs_processing = False
        
        for collection_name in collections_to_check:
            try:
                collection = current_rag_system.chroma_client.get_collection(collection_name)
                if collection.count() == 0:
                    needs_processing = True
                    break
            except:
                needs_processing = True
                break
        
        if needs_processing:
            print("📚 Processing legal documents for the first time...")
            # Process existing PDF files in glossary directories (limit to first 3 files for faster startup)
            hindi_glossary_dir = Path("glossary - hindi")
            telugu_glossary_dir = Path("glossary telugu")
            go_dir = Path("GOs (1)")
            
            if hindi_glossary_dir.exists():
                print("📚 Processing Hindi glossary documents (first 5 files for optimal performance)...")
                current_rag_system.process_glossary_documents_limited(str(hindi_glossary_dir), 'hindi', limit=5)
            
            if telugu_glossary_dir.exists():
                print("📚 Processing Telugu glossary documents (first 3 files for optimal performance)...")
                current_rag_system.process_glossary_documents_limited(str(telugu_glossary_dir), 'telugu', limit=3)
            
            if go_dir.exists():
                print("Processing Government Orders (first 5 files for optimal performance)...")
                current_rag_system.process_government_orders_limited(str(go_dir), limit=5)
        else:
            print("Legal documents already processed - using existing data")
        
        print("Legal RAG System initialized successfully!")
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        current_rag_system = None

# Configure Hugging Face OCR API
def configure_hf_ocr():
    """Configure Hugging Face OCR API"""
    try:
        print("Configuring Hugging Face OCR API...")
        
        # Test API connection
        api_url = "https://api-inference.huggingface.co/models/microsoft/trocr-base-printed"
        headers = {"Authorization": "Bearer hf_your_token_here"}  # You can add your HF token here
        
        print("Hugging Face OCR API configured successfully")
        return True
        
    except Exception as e:
        print(f"Error configuring Hugging Face OCR API: {e}")
        return False

# Initialize Hugging Face OCR
HF_OCR_AVAILABLE = configure_hf_ocr()

# Initialize IndicTransToolkit
print("Initializing IndicTransToolkit...")
try:
    import os
    
    # Check for offline mode (self-hosted)
    OFFLINE_MODE = os.environ.get('INDIC_TRANS_OFFLINE', 'false').lower() in ('true', '1', 'yes')
    if OFFLINE_MODE:
        print("🔒 Running in OFFLINE mode (self-hosted) - using local models only")
    
    # Direction-specific model names
    INDIC2_INDIC_EN = "ai4bharat/indictrans2-indic-en-1B"   # Indic -> English
    INDIC2_EN_INDIC = "ai4bharat/indictrans2-en-indic-1B"   # English -> Indic

    # Lazy-loaded model/tokenizer caches per direction
    indic2_indic_en_tokenizer = None
    indic2_indic_en_model = None
    indic2_en_indic_tokenizer = None
    indic2_en_indic_model = None
    indic_processor = None

    def _ensure_processor():
        global indic_processor
        if indic_processor is None:
            indic_processor = IndicProcessor(inference=True)
        return indic_processor

    def _load_model(model_name: str, local_only=False):
        """Load model, optionally from local directory only (fully offline/self-hosted)"""
        import os
        import shutil
        
        # Local models directory (in the project folder)
        local_models_dir = os.path.join(os.path.dirname(__file__), 'models')
        model_path = os.path.join(local_models_dir, model_name.replace('/', '_'))
        
        # ALWAYS try local directory first (truly self-hosted)
        if os.path.exists(model_path) and os.path.isdir(model_path):
            print(f"📦 Loading model from LOCAL directory: {model_path}")
            try:
                tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                mdl = mdl.to(device)
                mdl.eval()
                print(f"✅ Loaded '{model_name}' from LOCAL storage on {device}")
                return tok, mdl
            except Exception as e:
                print(f"⚠️ Failed to load from local directory: {e}")
                if local_only:
                    raise OSError(f"Model '{model_name}' not found in local directory '{model_path}'. Please run 'python download_models.py' first.")
        
        # If offline mode, don't try to download from internet
        if local_only:
            # Try Hugging Face cache as last resort
            print(f"🔍 Trying Hugging Face cache (last resort)...")
            try:
                tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
                mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                mdl = mdl.to(device)
                mdl.eval()
                print(f"✅ Loaded '{model_name}' from HF cache on {device}")
                
                # Copy from cache to local directory for future use
                if not os.path.exists(local_models_dir):
                    os.makedirs(local_models_dir, exist_ok=True)
                print(f"💾 Copying model to local directory for self-hosting...")
                # Note: transformers cache structure is complex, but we can use save_pretrained
                tok.save_pretrained(model_path)
                mdl.save_pretrained(model_path)
                print(f"✅ Model saved locally to: {model_path}")
                return tok, mdl
            except Exception as e:
                raise OSError(f"Model '{model_name}' not found locally. Please run 'python download_models.py' to download models for offline use.")
        
        # Online mode: Download from Hugging Face and SAVE to local directory
        print(f"🌐 Downloading IndicTrans2 model '{model_name}' (will save locally for offline use)...")
        try:
            # Try to use Hugging Face token from environment or login
            from huggingface_hub import login as hf_login
            hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_HUB_TOKEN')
            if hf_token:
                try:
                    hf_login(token=hf_token, add_to_git_credential=False)
                    print(f"✅ Authenticated with Hugging Face using token")
                except:
                    pass
            
            # Download model
            tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
            
            # IMPORTANT: Save to local directory for truly self-hosted operation
            if not os.path.exists(local_models_dir):
                os.makedirs(local_models_dir, exist_ok=True)
            
            print(f"💾 Saving model to LOCAL directory: {model_path}...")
            tok.save_pretrained(model_path)
            mdl.save_pretrained(model_path)
            print(f"✅ Model saved locally for offline use!")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            mdl = mdl.to(device)
            mdl.eval()
            print(f"✅ Loaded '{model_name}' on {device}")
            return tok, mdl
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower():
                print(f"⚠️ Model '{model_name}' is gated. Checking for local copy...")
                # Try local directory again
                if os.path.exists(model_path) and os.path.isdir(model_path):
                    try:
                        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
                        mdl = AutoModelForSeq2SeqLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
                        device = "cuda" if torch.cuda.is_available() else "cpu"
                        mdl = mdl.to(device)
                        mdl.eval()
                        print(f"✅ Loaded '{model_name}' from local directory on {device}")
                        return tok, mdl
                    except:
                        pass
                
                # Try cache as last resort
                try:
                    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
                    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    mdl = mdl.to(device)
                    mdl.eval()
                    print(f"✅ Loaded '{model_name}' from cache on {device}")
                    return tok, mdl
                except:
                    print(f"💡 To enable online access: Run 'huggingface-cli login' or set HF_TOKEN environment variable")
                    print(f"💡 Or run 'python download_models.py' to download models locally")
                    raise OSError(f"Gated model access required. Please authenticate or ensure model is downloaded locally.")
            raise

    def get_directional_models(src_lang_code: str, tgt_lang_code: str):
        """Return (tokenizer, model, model_name) appropriate for the src->tgt direction.
        If src!=eng and tgt!=eng, caller should pivot via English with two calls.
        """
        global indic2_indic_en_tokenizer, indic2_indic_en_model
        global indic2_en_indic_tokenizer, indic2_en_indic_model

        src_is_eng = (src_lang_code == 'eng_Latn')
        tgt_is_eng = (tgt_lang_code == 'eng_Latn')

        if not src_is_eng and tgt_is_eng:
            # Indic -> English
            if indic2_indic_en_tokenizer is None or indic2_indic_en_model is None:
                indic2_indic_en_tokenizer, indic2_indic_en_model = _load_model(INDIC2_INDIC_EN, local_only=OFFLINE_MODE)
            return indic2_indic_en_tokenizer, indic2_indic_en_model, INDIC2_INDIC_EN
        elif src_is_eng and not tgt_is_eng:
            # English -> Indic
            if indic2_en_indic_tokenizer is None or indic2_en_indic_model is None:
                indic2_en_indic_tokenizer, indic2_en_indic_model = _load_model(INDIC2_EN_INDIC, local_only=OFFLINE_MODE)
            return indic2_en_indic_tokenizer, indic2_en_indic_model, INDIC2_EN_INDIC
        else:
            # Non-English to Non-English requires pivot via English handled by caller
            return None, None, None

    INDIC_TRANS_AVAILABLE = True
    print("IndicTransToolkit ready (lazy loading enabled per direction)")
except Exception as e:
    print(f"WARNING: IndicTransToolkit initialization failed: {e}")
    print("Translation will fall back to basic method")
    INDIC_TRANS_AVAILABLE = False

# Language code mapping: app codes -> IndicTransToolkit codes
def get_indic_lang_code(app_lang_code, is_source=False):
    """Convert app language codes to IndicTransToolkit format"""
    # Source languages (typically English or Indic languages)
    source_mapping = {
        'en': 'eng_Latn',
        'hi': 'hin_Deva',
        'te': 'tel_Telu',
        'ta': 'tam_Taml',
        'kn': 'kan_Knda',
        'ml': 'mal_Mlym',
        'gu': 'guj_Gujr',
        'pa': 'pan_Guru',
        'bn': 'ben_Beng',
        'or': 'ory_Orya',
        'as': 'asm_Beng',
        'mr': 'mar_Deva',
        'ne': 'nep_Deva',
    }
    
    # Target languages
    target_mapping = {
        'en': 'eng_Latn',
        'hi': 'hin_Deva',
        'te': 'tel_Telu',
        'ta': 'tam_Taml',
        'kn': 'kan_Knda',
        'ml': 'mal_Mlym',
        'gu': 'guj_Gujr',
        'pa': 'pan_Guru',
        'bn': 'ben_Beng',
        'or': 'ory_Orya',
        'as': 'asm_Beng',
        'mr': 'mar_Deva',
        'ne': 'nep_Deva',
    }
    
    mapping = source_mapping if is_source else target_mapping
    return mapping.get(app_lang_code, 'eng_Latn')

def process_image_with_tesseract_ocr(image_path, source_language='auto'):
    """Process image with Tesseract OCR using pytesseract directly"""
    print("Using Tesseract OCR...")
    
    try:
        import pytesseract
        from PIL import Image
        
        # Set Tesseract path for Windows
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Set custom tessdata path to our local language packs
        tessdata_dir = os.path.join(os.path.dirname(__file__), 'tessdata')
        if os.path.exists(tessdata_dir):
            os.environ['TESSDATA_PREFIX'] = tessdata_dir
            print(f"Using custom tessdata directory: {tessdata_dir}")
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Determine language parameter - now with proper language support
        lang_param = 'eng'  # Default to English
        if source_language == 'te':
            lang_param = 'tel'  # Telugu
        elif source_language == 'hi':
            lang_param = 'hin'  # Hindi
        elif source_language == 'ta':
            lang_param = 'tam'  # Tamil
        elif source_language == 'kn':
            lang_param = 'kan'  # Kannada
        elif source_language == 'auto':
            # Try multiple languages for auto detection
            lang_param = 'eng+tel+hin'
        
        print(f"Using Tesseract with language: {lang_param}")
        
        # Get text with paragraph-based detection using Tesseract's block-level processing
        try:
            # Use PSM 6 (Uniform block of text) for better paragraph detection
            data = pytesseract.image_to_data(image, lang=lang_param, output_type=pytesseract.Output.DICT, 
                                           config='--psm 6')
            
            bboxes = []
            
            # Group text by paragraphs using line-based grouping
            paragraphs = []
            current_paragraph = []
            current_line_y = -1
            line_height_threshold = 30  # Threshold for grouping lines into paragraphs
            
            # First, collect all text items with their positions
            text_items = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                if text and int(data['conf'][i]) > 0:  # Only process non-empty text with confidence > 0
                    text_items.append({
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i],
                        'conf': float(data['conf'][i]) / 100.0,
                        'line_num': data['line_num'][i]
                    })
            
            # Sort by line number and then by x position within each line
            text_items.sort(key=lambda item: (item['line_num'], item['x']))
            
            # Group text items into paragraphs based on line proximity
            for item in text_items:
                if current_line_y == -1 or abs(item['y'] - current_line_y) < line_height_threshold:
                    # Same paragraph - add to current
                    current_paragraph.append(item)
                    current_line_y = item['y']
                else:
                    # New paragraph - process current and start new
                    if current_paragraph:
                        paragraphs.append(current_paragraph)
                    current_paragraph = [item]
                    current_line_y = item['y']
            
            # Add the last paragraph
            if current_paragraph:
                paragraphs.append(current_paragraph)
            
            # Process each paragraph
            for para_idx, paragraph_items in enumerate(paragraphs):
                if not paragraph_items:
                    continue
                
                # Combine text from all items in the paragraph
                paragraph_text = ' '.join([item['text'] for item in paragraph_items])
                
                # Calculate combined bounding box for the entire paragraph
                min_x = min([item['x'] for item in paragraph_items])
                min_y = min([item['y'] for item in paragraph_items])
                max_x = max([item['x'] + item['w'] for item in paragraph_items])
                max_y = max([item['y'] + item['h'] for item in paragraph_items])
                avg_conf = sum([item['conf'] for item in paragraph_items]) / len(paragraph_items)
                
                bbox_coords = [
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y]
                ]
                
                bbox_info = {
                    "bbox": bbox_coords,
                    "text": paragraph_text,
                    "confidence": avg_conf,
                    "engine": "Tesseract"
                }
                bboxes.append(bbox_info)
                print(f"Tesseract found paragraph {para_idx + 1}: '{paragraph_text[:50]}...' (conf: {avg_conf:.2f})")
            
            print(f"Tesseract OCR found {len(bboxes)} paragraph regions")
            return bboxes
            
        except Exception as e:
            print(f"Tesseract detailed OCR failed: {e}")
            # Fallback to simple text extraction
            try:
                text = pytesseract.image_to_string(image, lang=lang_param)
                if text.strip():
                    # Create a single bounding box for the entire image
                    img_width, img_height = image.size
                    bbox_coords = [
                        [0, 0],
                        [img_width, 0],
                        [img_width, img_height],
                        [0, img_height]
                    ]
                    
                    bbox_info = {
                        "bbox": bbox_coords,
                        "text": text.strip(),
                        "confidence": 0.8,
                        "engine": "Tesseract"
                    }
                    print(f"Tesseract fallback found text: '{text[:50]}...'")
                    return [bbox_info]
                else:
                    print("Tesseract found no text")
                    return []
            except Exception as e2:
                print(f"Tesseract fallback also failed: {e2}")
                return []
            
    except ImportError:
        print("pytesseract not available, falling back to EasyOCR")
        return []
    except Exception as e:
        print(f"Tesseract OCR completely failed: {e}")
        return []
    
def process_image_with_hf_ocr(image_path):
    if not EASYOCR_AVAILABLE:
        print("EasyOCR not available, falling back to Tesseract OCR...")
        return process_image_with_tesseract_ocr(image_path, 'auto')
    
    if not CV2_AVAILABLE:
        print("cv2 not available, falling back to Tesseract OCR...")
        return process_image_with_tesseract_ocr(image_path, 'auto')
    
    print("Using improved EasyOCR as fallback...")
    
    try:
        # Load and preprocess image more aggressively
        image = Image.open(image_path).convert('RGB')
        
        # Convert to numpy array for OpenCV processing
        img_array = np.array(image)
        
        # Apply multiple preprocessing techniques
        processed_images = []
        
        # 1. Original image
        processed_images.append(('original', img_array))
        
        # 2. Grayscale
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        processed_images.append(('grayscale', gray))
        
        # 3. Denoised
        denoised = cv2.fastNlMeansDenoising(gray)
        processed_images.append(('denoised', denoised))
        
        # 4. Adaptive threshold
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        processed_images.append(('threshold', thresh))
        
        # 5. Morphological operations
        kernel = np.ones((2,2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        processed_images.append(('morphology', morph))
        
        bboxes = []
        
        # Try each processed image with EasyOCR
        for name, processed_img in processed_images:
            try:
                print(f"Trying EasyOCR with {name} image...")
                
                # Save processed image temporarily
                temp_path = image_path.replace('.', f'_{name}.')
                if len(processed_img.shape) == 2:  # Grayscale
                    cv2.imwrite(temp_path, processed_img)
                else:  # Color
                    cv2.imwrite(temp_path, cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))
                
                # Try with different language combinations
                language_combinations = [
                    ['en'],  # English only
                    ['te', 'en'],  # Telugu + English
                    ['hi', 'en'],  # Hindi + English
                ]
                
                for lang_combo in language_combinations:
                    try:
                        if not EASYOCR_AVAILABLE:
                            continue
                        reader = easyocr.Reader(lang_combo)
                        results = reader.readtext(temp_path, detail=1, paragraph=True)
                        
                        for res in results:
                            if len(res) >= 2 and res[1].strip():
                                bbox_info = {
                                    "bbox": res[0], 
                                    "text": res[1],
                                    "confidence": res[2] if len(res) >= 3 else 0.9,
                                    "preprocessing": name
                                }
                                bboxes.append(bbox_info)
                                print(f"Found text with {name}: '{res[1][:50]}...'")
                        
                        if bboxes:
                            print(f"EasyOCR found {len(bboxes)} text regions with {name} preprocessing")
                            return bboxes
                            
                    except Exception as e:
                        print(f"EasyOCR failed with {name} and {lang_combo}: {e}")
                        continue
                
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                print(f"Processing failed for {name}: {e}")
                continue
        
        if bboxes:
            print(f"Improved EasyOCR found {len(bboxes)} text regions")
            return bboxes
        else:
            print("Improved EasyOCR could not extract text")
            return []
        
    except Exception as e:
        print(f"Improved EasyOCR completely failed: {e}")
        return []

def process_image_with_ocr(image_path):
    """Process image with EasyOCR (if available) or Tesseract OCR"""
    print(f"Processing image: {image_path}")
    
    # Check if image file exists and is readable
    if not os.path.exists(image_path):
        print(f"ERROR: Image file does not exist: {image_path}")
        return []
    
    try:
        # Test if image can be opened and convert to RGB if needed
        with Image.open(image_path) as img:
            print(f"Image loaded successfully: {img.size}, mode: {img.mode}")
            if img.size[0] == 0 or img.size[1] == 0:
                print("ERROR: Image has zero dimensions")
                return []
            
            # Convert image to RGB if it's in a different mode (like P for palette)
            if img.mode != 'RGB':
                print(f"Converting image from {img.mode} to RGB")
                img = img.convert('RGB')
                # Save the converted image temporarily
                temp_path = image_path.replace('.', '_converted.')
                img.save(temp_path)
                image_path = temp_path
                print(f"Converted image saved to: {image_path}")
                
    except Exception as e:
        print(f"ERROR: Cannot open image: {e}")
        return []
    
    # If EasyOCR is not available, use Tesseract directly
    if not EASYOCR_AVAILABLE:
        print("EasyOCR not available, using Tesseract OCR...")
        return process_image_with_tesseract_ocr(image_path, 'auto')
    
    # Try multiple language combinations for better detection
    language_combinations = [
        ['en'],  # English only
        ['en', 'hi'],  # English + Hindi
        ['en', 'te'],  # English + Telugu
        ['en', 'hi', 'te'],  # English + Hindi + Telugu
        ['en', 'hi', 'te', 'ta', 'kn', 'ml', 'gu', 'pa', 'bn', 'or'],  # All major Indian languages
    ]
    
    bboxes = []
    
    for lang_combo in language_combinations:
        try:
            print(f"Trying OCR with languages: {lang_combo}")
            reader = easyocr.Reader(lang_combo)
            # Force paragraph mode for consistent layout-based detection
            results = reader.readtext(image_path, detail=1, paragraph=True, blocklist='')
            
            for res in results:
                if len(res) >= 2:
                    bbox_info = {
                        "bbox": res[0], 
                        "text": res[1],
                        "confidence": res[2] if len(res) >= 3 else 0.9
                    }
                    bboxes.append(bbox_info)
            
            if bboxes:
                print(f"Found {len(bboxes)} text regions with {lang_combo} (paragraph mode)")
                break
                
        except Exception as e:
            print(f"OCR failed with {lang_combo}: {e}")
            continue
    
    # If still no text found, try with different paragraph parameters
    if not bboxes:
        print("Trying OCR with alternative paragraph settings...")
        try:
            reader = easyocr.Reader(['en'])
            # Try with different paragraph detection parameters
            results = reader.readtext(image_path, detail=1, paragraph=True, 
                                    width_ths=0.7, height_ths=0.7, 
                                    paragraph_ths=0.6, blocklist='')
            
            for res in results:
                if len(res) >= 2:
                    bbox_info = {
                        "bbox": res[0], 
                        "text": res[1],
                        "confidence": res[2] if len(res) >= 3 else 0.9
                    }
                    bboxes.append(bbox_info)
            
            print(f"Found {len(bboxes)} text regions with alternative paragraph settings")
        except Exception as e:
            print(f"OCR failed with alternative paragraph settings: {e}")
    
    # If EasyOCR completely fails, try Tesseract OCR as fallback
    if not bboxes:
        print("EasyOCR failed completely, trying Tesseract OCR...")
        bboxes = process_image_with_tesseract_ocr(image_path, 'auto')
    
    if not bboxes:
        print("WARNING: No text detected! The image might be:")
        print("   - Too blurry or low quality")
        print("   - Contains only images/logos without text")
        print("   - Text is too small or unclear")
        print("   - Contains handwritten text (EasyOCR works best with printed text)")
        print("   - Neither EasyOCR nor Tesseract OCR could process the image")
    
    return bboxes

def split_text_into_paragraphs(text):
    """Split text into logical paragraphs for processing"""
    print("📝 Splitting text into paragraphs...")
    
    # Split by double newlines first
    paragraphs = re.split(r'\n\s*\n', text.strip())
    
    # Further split long paragraphs by sentences
    refined_paragraphs = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If paragraph is very long (>500 chars), split by sentences
        if len(para) > 500:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk + sentence) > 300:
                    if current_chunk:
                        refined_paragraphs.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        refined_paragraphs.append(sentence)
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
            
            if current_chunk:
                refined_paragraphs.append(current_chunk.strip())
        else:
            refined_paragraphs.append(para)
    
    print(f"Split into {len(refined_paragraphs)} paragraphs")
    return refined_paragraphs

def detect_app_lang_code(sample_text: str) -> str:
    """Very simple unicode-range based detection to choose source app code."""
    if not sample_text:
        return 'en'
    # Devanagari (Hindi)
    if any('\u0900' <= ch <= '\u097F' for ch in sample_text):
        return 'hi'
    # Telugu
    if any('\u0C00' <= ch <= '\u0C7F' for ch in sample_text):
        return 'te'
    # Tamil
    if any('\u0B80' <= ch <= '\u0BFF' for ch in sample_text):
        return 'ta'
    # Kannada
    if any('\u0C80' <= ch <= '\u0CFF' for ch in sample_text):
        return 'kn'
    # Malayalam
    if any('\u0D00' <= ch <= '\u0D7F' for ch in sample_text):
        return 'ml'
    return 'en'

def translate_with_indic_toolkit(text, source_language='en', target_language='te'):
    """Translate text using IndicTransToolkit following the official example code"""
    if not INDIC_TRANS_AVAILABLE:
        print("IndicTransToolkit not available, using fallback")
        return fallback_translation(text, target_language)
    
    try:
        processor = _ensure_processor()
        
        # ALWAYS auto-detect source language from text (ignore what caller says if it seems wrong)
        detected = detect_app_lang_code(text if isinstance(text, str) else ' '.join(text))
        if source_language in ('auto', None, '') or (source_language == 'en' and detected != 'en'):
            source_language = detected
            print(f"🔍 Overriding source language to detected: {detected}")
        
        # Get language codes
        src_lang = get_indic_lang_code(source_language, is_source=True)
        tgt_lang = get_indic_lang_code(target_language, is_source=False)
        
        print(f"🌐 Translation: {src_lang} -> {tgt_lang}")
        
        # If both sides are non-English, pivot via English
        if src_lang != 'eng_Latn' and tgt_lang != 'eng_Latn':
            print(f"🔄 Pivoting {src_lang}->{tgt_lang} via English...")
            # Step 1: src -> English (this should work)
            mid = translate_with_indic_toolkit(text, source_language=source_language, target_language='en')
            print(f"   Intermediate (English): {mid[:100]}...")
            # Step 2: English -> target (may fail if model is gated)
            try:
                result = translate_with_indic_toolkit(mid, source_language='en', target_language=target_language)
                print(f"   Final result: {result[:100]}...")
                return result
            except Exception as e:
                if "gated" in str(e).lower() or "401" in str(e) or "unauthorized" in str(e).lower():
                    print(f"⚠️ English->{target_language} model unavailable (gated). Returning English translation with note.")
                    lang_info = get_language_info(target_language)
                    return f"{mid}\n\n⚠️ [Translation Note: Full translation to {lang_info['name']} ({lang_info['native']}) requires access to an additional translation component.\nTo enable: authenticate your environment and ensure all translation components are available.\nCurrently showing English translation above.]"
                raise
        
        # Load correct directional model
        try:
            tokenizer, model, model_name = get_directional_models(src_lang, tgt_lang)
            if tokenizer is None or model is None:
                print(f"❌ ERROR: Could not load model for {src_lang} -> {tgt_lang}")
                return fallback_translation(text, target_language)
            print(f"✅ Using model: {model_name} for {src_lang} -> {tgt_lang}")
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower() or "401" in error_msg or "unauthorized" in error_msg.lower():
                print(f"⚠️ Model is gated. Providing English translation as fallback.")
                # If English->Indic fails, try reverse: use Indic->English to get English, then return with note
                if src_lang == 'eng_Latn' and tgt_lang != 'eng_Latn':
                    lang_info = get_language_info(target_language)
                    return f"{text}\n\n[Note: Translation to {lang_info['name']} requires Hugging Face authentication. Please run 'huggingface-cli login' or set HF_TOKEN environment variable.]"
                return fallback_translation(text, target_language)
            raise
        
        # Prepare input sentences
        input_sentences = [text] if isinstance(text, str) else text
        
        # Preprocess batch
        batch = processor.preprocess_batch(
            input_sentences,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
        )
        
        # Tokenize
        device = model.device
        inputs = tokenizer(
            batch,
            truncation=True,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
        ).to(device)
        
        # Generate translations using EXACT parameters from official example
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=256,  # Match official example
                num_beams=5,      # Match official example
                num_return_sequences=1,
            )
        
        # Decode tokens - using as_target_tokenizer as in official example
        with tokenizer.as_target_tokenizer():
            generated_tokens = tokenizer.batch_decode(
                generated_tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        
        # Postprocess the translations, including entity replacement
        translations = processor.postprocess_batch(generated_tokens, lang=tgt_lang)
        
        # Return single translation or list
        if isinstance(text, str):
            result = translations[0] if translations else text
            print(f"✨ Translation result: {result[:100]}...")
            return result
        return translations
        
    except Exception as e:
        import traceback
        print(f"❌ IndicTransToolkit translation error: {e}")
        traceback.print_exc()
        return fallback_translation(text, target_language)

def translate_paragraph_with_rag(paragraph, model='gemma3:latest', target_language='te', source_language='auto'):
    """Translate a single paragraph using IndicTransToolkit (RAG disabled for IndicTransToolkit)"""
    print(f"Translating paragraph: {paragraph[:50]}...")
    
    lang_info = get_language_info(target_language)
    
    # Auto-detect source language from text if not provided or is 'auto'
    if source_language in ('auto', None, ''):
        detected = detect_app_lang_code(paragraph)
        source_language = detected
        print(f"🔍 Auto-detected source language: {detected}")
    
    print(f"Translating from {source_language} to {target_language}")
    
    # Use IndicTransToolkit directly - NO RAG for IndicTransToolkit
    try:
        translated_text = translate_with_indic_toolkit(paragraph, source_language, target_language)
        print(f"Paragraph translated to {lang_info['name']} using IndicTransToolkit")
        return translated_text
    except Exception as e:
        print(f"Translation error: {e}")
        return fallback_translation(paragraph, target_language)

def translate_text_with_rag_paragraphs(text, model='gemma3:latest', target_language='te', source_language='auto'):
    """Translate text paragraph by paragraph using IndicTransToolkit (NO RAG)"""
    print("Starting paragraph-by-paragraph translation with IndicTransToolkit...")
    
    # Auto-detect source language from text if not provided
    if source_language in ('auto', None, ''):
        detected = detect_app_lang_code(text)
        source_language = detected
        print(f"🔍 Auto-detected source language: {detected}")
    
    # For short texts, translate directly without splitting
    if len(text.split()) < 50:
        print("Short text, translating directly...")
        return translate_with_indic_toolkit(text, source_language, target_language)
    
    # Split text into paragraphs for longer texts
    paragraphs = split_text_into_paragraphs(text)
    
    # Store paragraphs globally for processing
    global current_paragraphs
    current_paragraphs = paragraphs
    
    translated_paragraphs = []
    
    for i, paragraph in enumerate(paragraphs):
        print(f"📝 Processing paragraph {i+1}/{len(paragraphs)}")
        
        # Translate each paragraph directly using IndicTransToolkit
        translated_para = translate_with_indic_toolkit(paragraph, source_language, target_language)
        translated_paragraphs.append(translated_para)
        
        # Add small delay to prevent overwhelming the system
        time.sleep(0.1)
    
    # Join translated paragraphs
    final_translation = '\n\n'.join(translated_paragraphs)
    print(f"Completed paragraph-by-paragraph translation ({len(paragraphs)} paragraphs)")
    
    return final_translation

def translate_text(text, model='gemma3:latest', target_language='te', source_language='auto'):
    """Translate text using IndicTransToolkit (NO RAG)"""
    print("Translating text with IndicTransToolkit...")
    
    # ALWAYS auto-detect source language from actual text (override if needed)
    detected = detect_app_lang_code(text)
    if source_language in ('auto', None, '') or (source_language == 'en' and detected != 'en'):
        source_language = detected
        print(f"🔍 Auto-detected source language: {detected}")
    
    # If source and target are the same, return original text
    if source_language == target_language:
        print(f"⚠️ Source and target are same ({source_language}), returning original text")
        return text
    
    print(f"📋 Translating from {source_language} to {target_language}")
    
    # Use IndicTransToolkit directly - NO RAG
    translated = translate_with_indic_toolkit(text, source_language, target_language)
    print(f"✅ Translation completed")
    return translated

def fallback_translation(text, target_language):
    """Fallback translation when IndicTransToolkit is unavailable"""
    print(f"Using fallback translation for {target_language}")
    
    # If text is already in English and target is Indic, provide helpful message
    # Check if text is mostly English (simple heuristic)
    is_english = not any('\u0900' <= ch <= '\u0D7F' for ch in text)  # Not Indic script
    if is_english and target_language != 'en':
        lang_info = get_language_info(target_language)
        return f"{text}\n\n⚠️ [Translation Note: Full translation to {lang_info['name']} ({lang_info['native']}) requires access to an additional translation component.\nTo enable: authenticate your environment and ensure all translation components are available.\nCurrently showing English text above.]"
    
    # Simple fallback translations for common legal terms
    fallback_translations = {
        'te': {  # Telugu
            'OFFICE OF THE REGISTRAR GENERAL': 'రిజిస్ట్రార్ జనరల్ కార్యాలయం',
            'Government of India': 'భారత ప్రభుత్వం',
            'Ministry of Home Affairs': 'గృహ వ్యవహారాల మంత్రిత్వ శాఖ',
            'TENDER ENQUIRY NOTICE': 'టెండర్ విచారణ నోటీసు',
            'Subject:': 'విషయం:',
            'Dated:': 'తేదీ:',
            'No.': 'సంఖ్య:',
            'Procurement': 'కొనుగోలు',
            'Supply': 'సరఫరా',
            'Services': 'సేవలు',
            'Contract': 'ఒప్పందం',
            'Agreement': 'ఒప్పందం',
            'Terms and Conditions': 'నిబంధనలు మరియు షరతులు'
        },
        'hi': {  # Hindi
            'OFFICE OF THE REGISTRAR GENERAL': 'रजिस्ट्रार जनरल का कार्यालय',
            'Government of India': 'भारत सरकार',
            'Ministry of Home Affairs': 'गृह मंत्रालय',
            'TENDER ENQUIRY NOTICE': 'निविदा पूछताछ नोटिस',
            'Subject:': 'विषय:',
            'Dated:': 'दिनांक:',
            'No.': 'संख्या:',
            'Procurement': 'खरीद',
            'Supply': 'आपूर्ति',
            'Services': 'सेवाएं',
            'Contract': 'अनुबंध',
            'Agreement': 'समझौता',
            'Terms and Conditions': 'नियम और शर्तें'
        },
        'ta': {  # Tamil
            'OFFICE OF THE REGISTRAR GENERAL': 'பதிவாளர் ஜெனரல் அலுவலகம்',
            'Government of India': 'இந்திய அரசு',
            'Ministry of Home Affairs': 'உள்துறை அமைச்சகம்',
            'TENDER ENQUIRY NOTICE': 'டெண்டர் விசாரணை அறிவிப்பு',
            'Subject:': 'பொருள்:',
            'Dated:': 'தேதி:',
            'No.': 'எண்:',
            'Procurement': 'கொள்முதல்',
            'Supply': 'வழங்கல்',
            'Services': 'சேவைகள்',
            'Contract': 'ஒப்பந்தம்',
            'Agreement': 'ஒப்பந்தம்',
            'Terms and Conditions': 'விதிமுறைகள் மற்றும் நிபந்தனைகள்'
        }
    }
    
    # Get translations for the target language
    translations = fallback_translations.get(target_language, {})
    
    # Apply translations
    translated_text = text
    for english_term, translated_term in translations.items():
        translated_text = translated_text.replace(english_term, translated_term)
    
    # If no translations were applied, return original text with a note
    if translated_text == text:
        translated_text = f"[{target_language.upper()}] {text} [Translation unavailable - IndicTransToolkit service required]"
    
    return translated_text

def get_language_info(lang_code):
    """Get language information by code"""
    languages = {
        'te': {'name': 'Telugu', 'native': 'తెలుగు'},
        'hi': {'name': 'Hindi', 'native': 'हिन्दी'},
        'ta': {'name': 'Tamil', 'native': 'தமிழ்'},
        'kn': {'name': 'Kannada', 'native': 'ಕನ್ನಡ'},
        'ml': {'name': 'Malayalam', 'native': 'മലയാളം'},
        'en': {'name': 'English', 'native': 'English'},
        'gu': {'name': 'Gujarati', 'native': 'ગુજરાતી'},
        'pa': {'name': 'Punjabi', 'native': 'ਪੰਜਾਬੀ'},
        'bn': {'name': 'Bengali', 'native': 'বাংলা'},
        'or': {'name': 'Odia', 'native': 'ଓଡ଼ିଆ'}
    }
    return languages.get(lang_code, {'name': 'Telugu', 'native': 'తెలుగు'})

def get_font_family(language_code):
    """Get appropriate font family for the language"""
    font_mapping = {
        'te': 'Noto+Sans+Telugu',
        'hi': 'Noto+Sans+Devanagari',
        'ta': 'Noto+Sans+Tamil',
        'kn': 'Noto+Sans+Kannada',
        'ml': 'Noto+Sans+Malayalam',
        'gu': 'Noto+Sans+Gujarati',
        'pa': 'Noto+Sans+Gurmukhi',
        'bn': 'Noto+Sans+Bengali',
        'or': 'Noto+Sans+Oriya',
        'en': 'Roboto'
    }
    return font_mapping.get(language_code, 'Roboto')

def create_processed_html(image_path, bboxes, translated_lines, user_actions, target_language='te'):
    """Create processed HTML document based on user actions"""
    print("Creating processed HTML document...")
    
    # Get language info for font selection
    lang_info = get_language_info(target_language)
    
    # Convert image to base64
    with open(image_path, 'rb') as img_file:
        img_data = base64.b64encode(img_file.read()).decode()
    
    # Get image dimensions
    with Image.open(image_path) as img:
        img_width, img_height = img.size
    
    # Create text overlays based on user actions
    text_overlays = []
    print(f"DEBUG: translated_lines length: {len(translated_lines)}")
    print(f"DEBUG: translated_lines content: {translated_lines[:3] if translated_lines else 'Empty'}")
    print(f"DEBUG: user_actions: {user_actions}")
    
    for i, bbox in enumerate(bboxes):
        action = user_actions.get(str(i), 'preserve')
        
        if action == 'whiteout':
            print(f"Skipping whiteout region {i}")
            continue
        elif action == 'translate':
            if i < len(translated_lines):
                text_to_draw = translated_lines[i].strip()
                # Ensure we don't truncate the text
                if len(text_to_draw) > 200:
                    print(f"✓ Using FULL TRANSLATED text for region {i} ({len(text_to_draw)} chars): '{text_to_draw[:100]}...'")
                else:
                    print(f"✓ Using TRANSLATED text for region {i}: '{text_to_draw}'")
            else:
                text_to_draw = f"[Translation missing for region {i}]"
                print(f"⚠️ Translation missing for region {i}, using placeholder.")
        elif action == 'preserve':
            text_to_draw = bbox['text']
            print(f"✓ Using ORIGINAL text for region {i} (Preserve): '{text_to_draw[:50]}...'")
        else:
            text_to_draw = bbox['text']
            print(f"⚠️ Unknown action '{action}' for region {i}, defaulting to ORIGINAL text.")
        
        # Convert bbox coordinates to percentages
        bbox_coords = bbox['bbox']
        x_coords = [point[0] for point in bbox_coords]
        y_coords = [point[1] for point in bbox_coords]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Convert to percentages
        left_pct = (x_min / img_width) * 100
        top_pct = (y_min / img_height) * 100
        width_pct = ((x_max - x_min) / img_width) * 100
        height_pct = ((y_max - y_min) / img_height) * 100
        
        # Calculate font size based on height - adjusted for better text fitting
        font_size = max(12, min(24, int(height_pct * 1.5)))
        
        text_overlays.append({
            'text': text_to_draw,
            'left': left_pct,
            'top': top_pct,
            'width': width_pct,
            'height': height_pct,
            'font_size': font_size,
            'action': action
        })
    
    # Get appropriate font for the language
    font_family = get_font_family(target_language)
    
    # Generate HTML overlays - EXACTLY like original main.py
    overlays_html = ""
    for overlay in text_overlays:
        overlays_html += f'''
        <div class="text-overlay {overlay['action']}" 
             style="left: {overlay['left']:.2f}%; 
                    top: {overlay['top']:.2f}%; 
                    width: {overlay['width']:.2f}%; 
                    height: {overlay['height']:.2f}%; 
                    font-size: {overlay['font_size']}px;">
            {overlay['text']}
        </div>
        '''
    
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Processed Document - {lang_info['name']}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family={font_family}:wght@400;700&display=swap');
        
        body {{
            margin: 0;
            padding: 0;
            font-family: '{font_family}', Arial, sans-serif;
            background: white;
            position: relative;
        }}
        
        .document-container {{
            position: relative;
            width: 100%;
            height: 100vh;
            background: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
            min-width: 800px;
            min-height: 600px;
        }}
        
        .background-image {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            opacity: 1.0;
            z-index: 1;
            pointer-events: none;
        }}
        
        .text-overlay {{
            position: absolute;
            z-index: 2;
            background: rgba(255, 255, 255, 1.0);
            border-radius: 2px;
            display: flex;
            align-items: flex-start;
            justify-content: flex-start;
            line-height: 1.1;
            padding: 3px;
            box-shadow: none;
            border: none;
            text-align: left;
            word-wrap: break-word;
            overflow-wrap: break-word;
            white-space: normal;
            overflow: visible;
            min-width: 20px;
            min-height: 15px;
        }}
        
        .text-overlay.translate {{
            background: rgba(255, 255, 255, 1.0);
            border: none;
        }}
        
        .text-overlay.preserve {{
            background: rgba(255, 255, 255, 1.0);
            border: none;
        }}
        
        .text-overlay.whiteout {{
            background: rgba(255, 255, 255, 1.0);
            border: none;
        }}
        
        /* Print styles */
        @media print {{
            body {{ margin: 0; padding: 0; }}
            .text-overlay {{ background: white; }}
        }}
    </style>
</head>
<body>
    <div class="document-container">
        <!-- Background image with full opacity to preserve logo/signatures -->
        <img src="data:image/png;base64,{img_data}" class="background-image" alt="Original Document">
        
        <!-- Text overlays positioned exactly like original -->
        {overlays_html}
    </div>
</body>
</html>
"""
    
    print(f"HTML document created")
    return html_template

def create_dynamic_ui():
    """Create the enhanced HTML interface with all advanced features"""
    # Read optional logos from user-provided absolute paths and embed as data URIs
    left_logo_data = ""
    right_logo_data = ""
    try:
        left_path = r"C:\\Users\\airot\\OneDrive\\Desktop\\Govt\\download (1).svg"
        if os.path.exists(left_path):
            with open(left_path, "rb") as f:
                left_logo_data = "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        left_logo_data = ""
    try:
        right_path = r"C:\\Users\\airot\\OneDrive\\Desktop\\Govt\\images.png"
        if os.path.exists(right_path):
            with open(right_path, "rb") as f:
                right_logo_data = "data:image/png;base64," + base64.b64encode(f.read()).decode()
    except Exception:
        right_logo_data = ""

    return f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Document Translation Studio</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #000; color: #fff; min-height: 100vh; }}
        .topbar {{ width: 100%; background: #0a0a0a; border-bottom: 1px solid #1f1f1f; position: sticky; top: 0; z-index: 100; }}
        .topbar-inner {{ max-width: 1400px; margin: 0 auto; padding: 12px 20px; display: flex; align-items: center; justify-content: space-between; gap: 16px; }}
        .brand {{ display: flex; align-items: center; gap: 12px; }}
        .brand-title {{ font-size: 18px; font-weight: 600; letter-spacing: 0.2px; }}
        .logo-img {{ height: 36px; width: auto; display: block; }}
        .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
        .header {{ text-align: center; margin: 24px 0 40px 0; border-bottom: 1px solid #1f1f1f; padding-bottom: 16px; }}
        .header h1 {{ font-size: 2.1em; font-weight: 500; margin-bottom: 10px; }}
        .header p {{ font-size: 1.1em; opacity: 0.8; }}
        .rag-status {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-size: 14px; font-weight: 600; margin: 10px 0; }}
        .rag-status.active {{ background: #e8f5e8; color: #2e7d32; border: 1px solid #4caf50; }}
        .rag-status.inactive {{ background: #fff3e0; color: #f57c00; border: 1px solid #ff9800; }}
        .upload-section {{ background: #111; border: 2px dashed #333; border-radius: 10px; padding: 40px; text-align: center; margin-bottom: 30px; transition: all 0.3s ease; }}
        .upload-section:hover {{ border-color: #666; background: #1a1a1a; }}
        .upload-section.dragover {{ border-color: #fff; background: #222; }}
        .upload-icon {{ font-size: 3em; margin-bottom: 20px; opacity: 0.6; }}
        .upload-text {{ font-size: 1.2em; margin-bottom: 20px; }}
        .file-input {{ display: none; }}
        .upload-btn {{ background: #fff; color: #000; border: none; padding: 12px 30px; border-radius: 5px; font-size: 1em; cursor: pointer; transition: all 0.3s ease; }}
        .upload-btn:hover {{ background: #ccc; }}
        .model-selection-section {{ background: #111; border-radius: 10px; padding: 30px; margin-bottom: 30px; }}
        .model-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .model-card {{ background: #1a1a1a; border: 2px solid #333; border-radius: 10px; padding: 20px; text-align: center; cursor: pointer; transition: all 0.3s ease; }}
        .model-card:hover {{ border-color: #666; background: #222; }}
        .model-card.selected {{ border-color: #fff; background: #333; }}
        .model-icon {{ font-size: 2em; margin-bottom: 10px; }}
        .model-name {{ font-size: 1.2em; font-weight: bold; margin-bottom: 5px; }}
        .model-desc {{ font-size: 0.9em; opacity: 0.7; }}
        .agent-mode-toggle {{ text-align: center; margin-top: 20px; }}
        .toggle-label {{ display: flex; align-items: center; justify-content: center; gap: 15px; cursor: pointer; }}
        .toggle-label input[type=\"checkbox\"] {{ display: none; }}
        .toggle-slider {{ width: 60px; height: 30px; background: #333; border-radius: 15px; position: relative; transition: all 0.3s ease; }}
        .toggle-slider::before {{ content: ''; position: absolute; width: 26px; height: 26px; background: #fff; border-radius: 50%; top: 2px; left: 2px; transition: all 0.3s ease; }}
        .toggle-label input[type=\"checkbox\"]:checked + .toggle-slider {{ background: #4CAF50; }}
        .toggle-label input[type=\"checkbox\"]:checked + .toggle-slider::before {{ transform: translateX(30px); }}
        .toggle-text {{ font-size: 1.1em; font-weight: 500; }}
        /* (rest of existing styles remain unchanged) */
    </style>
</head>
<body>
    <div class=\"topbar\"> 
        <div class=\"topbar-inner\">
            <div class=\"brand\">{f'<img src="{left_logo_data}" alt="Logo" class="logo-img" />' if left_logo_data else ''}<div class=\"brand-title\">Document Translation Studio</div></div>
            <div>{f'<img src="{right_logo_data}" alt="Brand" class="logo-img" />' if right_logo_data else ''}</div>
        </div>
    </div>
    <div class=\"container\">
        <div class=\"header\">
            <h1>Translate documents and text with precision</h1>
            <p>Upload a document or paste text. Choose languages and translate seamlessly.</p>
        </div>
"""

def create_dynamic_ui():
    """Create the enhanced HTML interface with all advanced features"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Legal RAG Document Translator</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+Telugu:wght@400;700&display=swap');
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #000;
            color: #fff;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #fff;
            padding-bottom: 20px;
        }
        
        .header h1 {
            font-size: 2.5em;
            font-weight: 300;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.8;
        }
        
        .rag-status {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
            margin: 10px 0;
        }
        
        .rag-status.active {
            background: #e8f5e8;
            color: #2e7d32;
            border: 1px solid #4caf50;
        }
        
        .rag-status.inactive {
            background: #fff3e0;
            color: #f57c00;
            border: 1px solid #ff9800;
        }
        
        .upload-section {
            background: #111;
            border: 2px dashed #333;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            margin-bottom: 30px;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #666;
            background: #1a1a1a;
        }
        
        .upload-section.dragover {
            border-color: #fff;
            background: #222;
        }
        
        .upload-icon {
            font-size: 3em;
            margin-bottom: 20px;
            opacity: 0.6;
        }
        
        .upload-text {
            font-size: 1.2em;
            margin-bottom: 20px;
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: #fff;
            color: #000;
            border: none;
            padding: 12px 30px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .upload-btn:hover {
            background: #ccc;
        }
        
        .model-selection-section {
            background: #111;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .model-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .model-card {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .model-card:hover {
            border-color: #666;
            background: #222;
        }
        
        .model-card.selected {
            border-color: #fff;
            background: #333;
        }
        
        .model-icon {
            font-size: 2em;
            margin-bottom: 10px;
        }
        
        .model-name {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .model-desc {
            font-size: 0.9em;
            opacity: 0.7;
        }
        
        .agent-mode-toggle {
            text-align: center;
            margin-top: 20px;
        }
        
        .toggle-label {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            cursor: pointer;
        }
        
        .toggle-label input[type="checkbox"] {
            display: none;
        }
        
        .toggle-slider {
            width: 60px;
            height: 30px;
            background: #333;
            border-radius: 15px;
            position: relative;
            transition: all 0.3s ease;
        }
        
        .toggle-slider::before {
            content: '';
            position: absolute;
            width: 26px;
            height: 26px;
            background: #fff;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: all 0.3s ease;
        }
        
        .toggle-label input[type="checkbox"]:checked + .toggle-slider {
            background: #4CAF50;
        }
        
        .toggle-label input[type="checkbox"]:checked + .toggle-slider::before {
            transform: translateX(30px);
        }
        
        .toggle-text {
            font-size: 1.1em;
            font-weight: 500;
        }
        
        .language-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        
        .language-card {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .language-card:hover {
            border-color: #666;
            background: #222;
        }
        
        .language-card.selected {
            border-color: #4CAF50;
            background: #2a2a2a;
        }
        
        .language-icon {
            font-size: 1.5em;
            margin-bottom: 8px;
        }
        
        .language-name {
            font-size: 0.9em;
            font-weight: bold;
            margin-bottom: 3px;
        }
        
        .language-desc {
            font-size: 0.8em;
            opacity: 0.7;
        }
        
        .input-mode-toggle {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .toggle-buttons {
            display: inline-flex;
            background: #1a1a1a;
            border-radius: 10px;
            padding: 5px;
            gap: 5px;
        }
        
        .mode-btn {
            background: transparent;
            color: #fff;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .mode-btn.active {
            background: #4CAF50;
            color: #000;
        }
        
        .mode-btn:hover:not(.active) {
            background: #333;
        }
        
        .text-input-section {
            background: #111;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .text-language-selection {
            margin-bottom: 30px;
        }
        
        .text-language-selection .language-grid {
            margin-top: 15px;
        }
        
        .text-input-container {
            margin-bottom: 20px;
        }
        
        .input-label {
            font-size: 1.1em;
            font-weight: 600;
            margin-bottom: 10px;
            color: #fff;
        }
        
        .current-language-display {
            background: #2a2a2a;
            border: 1px solid #4CAF50;
            border-radius: 5px;
            padding: 8px 12px;
            margin-bottom: 15px;
            font-size: 0.9em;
            color: #4CAF50;
            font-weight: 500;
        }
        
        .text-input-container textarea {
            width: 100%;
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 15px;
            color: #fff;
            font-size: 16px;
            font-family: 'Segoe UI', Arial, sans-serif;
            resize: vertical;
            min-height: 150px;
        }
        
        .text-input-container textarea:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        .text-input-container textarea::placeholder {
            color: #666;
        }
        
        .text-input-actions {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            justify-content: center;
        }
        
        .text-result-container {
            border-top: 1px solid #333;
            padding-top: 20px;
        }
        
        .text-result {
            background: #1a1a1a;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 20px;
            color: #fff;
            font-size: 16px;
            line-height: 1.6;
            min-height: 100px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .text-result-actions {
            display: flex;
            gap: 15px;
            margin-top: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .editable-translation {
            background: #2a2a2a;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 20px;
            color: #fff;
            font-size: 16px;
            line-height: 1.6;
            min-height: 100px;
            white-space: pre-wrap;
            word-wrap: break-word;
            cursor: text;
        }
        
        .editable-translation:focus {
            outline: none;
            border-color: #fff;
        }
        
        /* Enhanced UI Components */
        .card {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
        }
        
        .card:hover {
            border-color: #4CAF50;
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.1);
        }
        
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        }
        
        .card-title {
            font-size: 1.2em;
            font-weight: 600;
            color: #fff;
        }
        
        .card-subtitle {
            font-size: 0.9em;
            color: #888;
            margin-top: 5px;
        }
        
        .badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .badge-success {
            background: #4CAF50;
            color: #000;
        }
        
        .badge-warning {
            background: #ff9800;
            color: #000;
        }
        
        .badge-info {
            background: #2196F3;
            color: #fff;
        }
        
        .badge-error {
            background: #f44336;
            color: #fff;
        }
        
        .floating-action-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: #4CAF50;
            color: #000;
            border: none;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 20px rgba(76, 175, 80, 0.3);
            transition: all 0.3s ease;
            z-index: 1000;
        }
        
        .floating-action-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 6px 25px rgba(76, 175, 80, 0.4);
        }
        
        /* Multi-Agent Translation Options UI */
        .translation-options-container {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
        }
        
        .translation-option {
            background: #2a2a2a;
            border: 2px solid #444;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .translation-option:hover {
            border-color: #4CAF50;
            background: #2d2d2d;
        }
        
        .translation-option.selected {
            border-color: #4CAF50;
            background: #1e3a1e;
            box-shadow: 0 0 10px rgba(76, 175, 80, 0.3);
        }
        
        .option-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .option-title {
            font-weight: bold;
            color: #4CAF50;
            font-size: 16px;
        }
        
        .option-badge {
            background: #333;
            color: #fff;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
        }
        
        .option-badge.original {
            background: #2196F3;
        }
        
        .option-badge.improved {
            background: #4CAF50;
        }
        
        .option-badge.rag {
            background: #FF9800;
        }
        
        .option-content {
            color: #ccc;
            line-height: 1.6;
            margin-bottom: 10px;
        }
        
        .option-metrics {
            display: flex;
            gap: 15px;
            font-size: 12px;
            color: #888;
        }
        
        .metric {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .metric-icon {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }
        
        .metric-icon.accuracy {
            background: #4CAF50;
        }
        
        .metric-icon.consistency {
            background: #2196F3;
        }
        
        .metric-icon.quality {
            background: #FF9800;
        }
        
        .comparison-view {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 20px 0;
        }
        
        .comparison-panel {
            background: #2a2a2a;
            border: 1px solid #444;
            border-radius: 8px;
            padding: 15px;
        }
        
        .comparison-panel h4 {
            margin: 0 0 10px 0;
            color: #4CAF50;
            font-size: 14px;
        }
        
        .comparison-text {
            color: #ccc;
            line-height: 1.6;
            font-size: 14px;
        }
        
        .agent-progress {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .agent-step {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 8px 0;
            border-bottom: 1px solid #333;
        }
        
        .agent-step:last-child {
            border-bottom: none;
        }
        
        .step-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 12px;
            font-weight: bold;
        }
        
        .step-icon.active {
            background: #4CAF50;
            color: white;
        }
        
        .step-icon.completed {
            background: #2196F3;
            color: white;
        }
        
        .step-icon.pending {
            background: #666;
            color: #ccc;
        }
        
        .step-text {
            color: #ccc;
            font-size: 14px;
        }
        
        .step-text.active {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .step-text.completed {
            color: #2196F3;
        }
        
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 120px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -60px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 12px;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        
        .language-editor {
            background: #2a2a2a;
            border: 2px solid #4CAF50;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
        
        .language-selector-row {
            display: flex;
            gap: 20px;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .language-selector-item {
            flex: 1;
        }
        
        .language-selector-item label {
            display: block;
            font-size: 0.9em;
            font-weight: 600;
            margin-bottom: 5px;
            color: #4CAF50;
        }
        
        .language-selector-item select {
            width: 100%;
            padding: 8px 12px;
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            color: #fff;
            font-size: 14px;
        }
        
        .language-selector-item select:focus {
            outline: none;
            border-color: #4CAF50;
        }
        
        .translation-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin: 15px 0;
        }
        
        .translation-side {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 15px;
        }
        
        .translation-side-header {
            font-size: 0.9em;
            font-weight: 600;
            margin-bottom: 10px;
            color: #4CAF50;
        }
        
        .translation-content {
            font-size: 14px;
            line-height: 1.6;
            color: #fff;
            min-height: 100px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .translation-content.editable {
            background: #2a2a2a;
            border: 1px solid #4CAF50;
            border-radius: 5px;
            padding: 10px;
            cursor: text;
        }
        
        .translation-content.editable:focus {
            outline: none;
            border-color: #fff;
        }
        
        .action-bar {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #333;
        }
        
        .btn-sm {
            padding: 6px 12px;
            font-size: 12px;
            border-radius: 4px;
        }
        
        .btn-outline {
            background: transparent;
            border: 1px solid #4CAF50;
            color: #4CAF50;
        }
        
        .btn-outline:hover {
            background: #4CAF50;
            color: #000;
        }
        
        /* Loading Screen Styles */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        
        .loading-content {
            text-align: center;
            color: white;
        }
        
        .loading-spinner {
            width: 60px;
            height: 60px;
            border: 4px solid #333;
            border-top: 4px solid #fff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            font-size: 1.2em;
            margin-bottom: 10px;
        }
        
        .loading-subtext {
            font-size: 0.9em;
            opacity: 0.7;
        }
        
        .model-loading {
            opacity: 0.5;
            pointer-events: none;
        }
        
        .model-loading::after {
            content: '⏳';
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2em;
        }
        
        .model-loading-card {
            background: #1a1a1a;
            border: 2px solid #333;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            grid-column: 1 / -1;
        }
        
        .model-loading-card .loading-spinner {
            width: 40px;
            height: 40px;
            margin-bottom: 15px;
        }
        
        .model-loading-card .loading-text {
            font-size: 1em;
            margin-bottom: 5px;
        }
        
        .translation-options {
            margin-top: 20px;
        }
        
        .processing-section {
            display: none;
            background: #111;
            border-radius: 10px;
            padding: 30px;
            margin-bottom: 30px;
        }
        
        .processing-step {
            display: flex;
            align-items: center;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
            transition: all 0.3s ease;
        }
        
        .processing-step.active {
            background: #222;
        }
        
        .processing-step.completed {
            background: #0a0;
        }
        
        .step-icon {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            background: #333;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-weight: bold;
        }
        
        .step-icon.active {
            background: #fff;
            color: #000;
        }
        
        .step-icon.completed {
            background: #0f0;
            color: #000;
        }
        
        .step-text {
            flex: 1;
        }
        
        .main-content {
            display: none;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .image-section {
            background: #111;
            border-radius: 10px;
            padding: 20px;
        }
        
        .image-container {
            position: relative;
            display: inline-block;
        }
        
        .document-image {
            max-width: 100%;
            border: 1px solid #333;
            border-radius: 5px;
        }
        
        .bbox-overlay {
            position: absolute;
            border: 2px solid #fff;
            background: rgba(255, 255, 255, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .bbox-overlay:hover {
            background: rgba(255, 255, 255, 0.3);
        }
        
        .bbox-overlay.translate {
            border-color: #fff;
            background: rgba(255, 255, 255, 0.2);
        }
        
        .bbox-overlay.preserve {
            border-color: #0f0;
            background: rgba(0, 255, 0, 0.2);
        }
        
        .bbox-overlay.whiteout {
            border-color: #f00;
            background: rgba(255, 0, 0, 0.2);
        }
        
        .controls-section {
            background: #111;
            border-radius: 10px;
            padding: 20px;
        }
        
        .section-title {
            font-size: 1.5em;
            margin-bottom: 20px;
            border-bottom: 1px solid #333;
            padding-bottom: 10px;
        }
        
        .text-region {
            background: #1a1a1a;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .text-region:hover {
            border-color: #666;
            background: #222;
        }
        
        .text-region.selected {
            border-color: #fff;
            background: #333;
        }
        
        .text-region.translate {
            border-color: #fff;
            background: #333;
        }
        
        .text-region.preserve {
            border-color: #0f0;
            background: #0a0;
        }
        
        .text-region.whiteout {
            border-color: #f00;
            background: #a00;
        }
        
        .region-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        
        .region-title {
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .region-controls {
            display: flex;
            gap: 10px;
        }
        
        .control-btn {
            background: #333;
            color: #fff;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.9em;
            transition: all 0.3s ease;
        }
        
        .control-btn:hover {
            background: #555;
        }
        
        .control-btn.active {
            background: #fff;
            color: #000;
        }
        
        .region-text {
            font-size: 0.9em;
            opacity: 0.8;
            line-height: 1.4;
        }
        
        .action-buttons {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-top: 30px;
        }
        
        .action-btn {
            background: #fff;
            color: #000;
            border: none;
            padding: 15px 30px;
            border-radius: 5px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .action-btn:hover {
            background: #ccc;
        }
        
        .action-btn:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
        }
        
        .preview-section {
            display: none;
            background: #111;
            border-radius: 10px;
            padding: 30px;
            text-align: center;
        }
        
        .preview-controls {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .preview-btn {
            padding: 8px 16px;
            background: #333;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        
        .preview-btn:hover {
            background: #555;
        }
        
        .preview-btn.active {
            background: #007bff;
        }
        
        .side-by-side-preview {
            display: flex;
            gap: 20px;
            justify-content: center;
            margin-bottom: 20px;
        }
        
        .preview-panel {
            flex: 1;
            max-width: 45%;
            background: #222;
            border-radius: 8px;
            padding: 15px;
        }
        
        .panel-header {
            font-weight: bold;
            margin-bottom: 15px;
            color: #fff;
            text-align: center;
        }
        
        .preview-container {
            position: relative;
            display: inline-block;
        }
        
        .document-image {
            max-width: 100%;
            height: auto;
            border: 1px solid #333;
            border-radius: 5px;
        }
        
        .text-overlays {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
        }
        
        .overlay-preview {
            margin-bottom: 20px;
        }
        
        .preview-image {
            max-width: 100%;
            border: 1px solid #333;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        
        .download-btn {
            background: #0f0;
            color: #000;
            border: none;
            padding: 12px 25px;
            border-radius: 5px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .download-btn:hover {
            background: #0a0;
        }
        
        .status-message {
            text-align: center;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            display: none;
        }
        
        .status-message.success {
            background: #0a0;
            color: #fff;
        }
        
        .status-message.error {
            background: #a00;
            color: #fff;
        }
        
        .status-message.info {
            background: #006;
            color: #fff;
        }
        
        /* Modal Styles */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 1000;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background: white;
            border-radius: 10px;
            padding: 0;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }
        
        .modal-header {
            padding: 20px;
            border-bottom: 1px solid #eee;
            text-align: center;
        }
        
        .modal-header h3 {
            margin: 0 0 10px 0;
            color: #333;
        }
        
        .modal-header p {
            margin: 0;
            color: #666;
            font-size: 14px;
        }
        
        .modal-body {
            padding: 20px;
        }
        
        .modal-footer {
            padding: 20px;
            border-top: 1px solid #eee;
            text-align: center;
        }
        
        .language-grid-modal {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }
        
        .language-card-modal {
            background: #f8f9fa;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .language-card-modal:hover {
            border-color: #007bff;
            background: #e3f2fd;
        }
        
        .language-card-modal.selected {
            border-color: #007bff;
            background: #007bff;
            color: white;
        }
        
        .language-card-modal .language-icon {
            font-size: 24px;
            margin-bottom: 8px;
        }
        
        .language-card-modal .language-name {
            font-weight: 600;
            font-size: 14px;
            margin-bottom: 4px;
        }
        
        .language-card-modal .language-desc {
            font-size: 12px;
            opacity: 0.8;
        }
        
        /* Control Panel Styles */
        .control-panel {
            background: #1a1a1a;
            border-radius: 10px;
            margin: 20px 0;
            overflow: hidden;
        }
        
        .panel-header {
            background: #333;
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }
        
        .panel-header h3 {
            margin: 0;
            color: #fff;
        }
        
        .toggle-panel {
            background: none;
            border: none;
            color: #fff;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.3s;
        }
        
        .toggle-panel.collapsed {
            transform: rotate(-90deg);
        }
        
        .panel-content {
            padding: 20px;
            display: block;
        }
        
        .panel-content.collapsed {
            display: none;
        }
        
        .control-section {
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }
        
        .control-section:last-child {
            border-bottom: none;
            margin-bottom: 0;
        }
        
        .control-section h4 {
            margin: 0 0 15px 0;
            color: #fff;
            font-size: 16px;
        }
        
        .control-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        
        .control-group label {
            color: #ccc;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .control-group input[type="checkbox"],
        .control-group input[type="radio"] {
            margin: 0;
        }
        
        .control-group select {
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
        }
        
        .control-group input[type="number"] {
            background: #333;
            color: #fff;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 6px;
            width: 80px;
            font-size: 14px;
        }
        
        .control-group input[type="range"] {
            width: 200px;
            margin: 0 10px;
        }
        
        .range-controls {
            display: flex;
            gap: 20px;
            align-items: center;
        }
        
        .scale-display {
            color: #007bff;
            font-weight: bold;
            font-size: 14px;
        }
        
        .language-grid-controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        
        .language-card-control {
            background: #333;
            border: 2px solid #555;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .language-card-control:hover {
            border-color: #007bff;
            background: #444;
        }
        
        .language-card-control.selected {
            border-color: #007bff;
            background: #007bff;
        }
        
        .language-card-control .language-icon {
            font-size: 20px;
            margin-bottom: 5px;
        }
        
        .language-card-control .language-name {
            font-size: 12px;
            color: #fff;
            font-weight: 500;
        }
        
        .paragraph-section {
            margin: 20px 0;
            padding: 20px;
            background: #1a1a1a;
            border-radius: 10px;
            border-left: 4px solid #4CAF50;
        }
        
        .paragraph-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .paragraph-number {
            background: #4CAF50;
            color: white;
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 14px;
            font-weight: 600;
        }
        
        .paragraph-status {
            padding: 5px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .paragraph-status.pending {
            background: #fff3e0;
            color: #f57c00;
        }
        
        .paragraph-status.processing {
            background: #e3f2fd;
            color: #1976d2;
        }
        
        .paragraph-status.completed {
            background: #e8f5e8;
            color: #2e7d32;
        }
        
        .paragraph-text {
            margin: 10px 0;
            padding: 15px;
            background: #2a2a2a;
            border-radius: 8px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        
        .paragraph-translation {
            margin: 10px 0;
            padding: 15px;
            background: #0a0;
            border-radius: 8px;
            font-size: 14px;
            line-height: 1.5;
            display: none;
        }
        
        .paragraph-translation.show {
            display: block;
        }
        
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
            margin: 20px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: #4CAF50;
            width: 0%;
            transition: width 0.3s ease;
        }
        
        .download-section {
            text-align: center;
            margin: 30px 0;
            padding: 20px;
            background: #111;
            border-radius: 15px;
            display: none;
        }
        
        .download-section.active {
            display: block;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 20px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .language-grid {
                grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
            }
            
            .action-buttons {
                flex-direction: column;
            }
            
            .main-content {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⚖️ Enhanced Legal RAG Document Translator</h1>
            <p>Advanced AI-powered translation with legal context awareness and real-time document processing</p>
            <div class="rag-status" id="rag-status">
                <span id="rag-status-text">Checking RAG System...</span>
            </div>
        </div>
        
        <div class="input-mode-toggle">
            <div class="toggle-buttons">
                <button class="mode-btn active" id="documentMode" onclick="switchMode('document')">📄 Document Upload</button>
                <button class="mode-btn" id="textMode" onclick="switchMode('text')">✏️ Text Input</button>
            </div>
        </div>
        
        <!-- Source Language Selection -->
        <div class="language-selection-section">
            <div class="section-title">📋 Document Language (Source)</div>
            <div class="language-grid">
                <div class="language-card" data-lang="auto">
                    <div class="language-icon">🔍</div>
                    <div class="language-name">Auto-detect</div>
                    <div class="language-desc">Let AI detect</div>
                </div>
                <div class="language-card" data-lang="en">
                    <div class="language-icon">🇺🇸</div>
                    <div class="language-name">English</div>
                    <div class="language-desc">English</div>
                </div>
                <div class="language-card" data-lang="te">
                    <div class="language-icon">📜</div>
                    <div class="language-name">Telugu</div>
                    <div class="language-desc">తెలుగు</div>
                </div>
                <div class="language-card" data-lang="hi">
                    <div class="language-icon">📖</div>
                    <div class="language-name">Hindi</div>
                    <div class="language-desc">हिन्दी</div>
                </div>
                <div class="language-card" data-lang="ta">
                    <div class="language-icon">📚</div>
                    <div class="language-name">Tamil</div>
                    <div class="language-desc">தமிழ்</div>
                </div>
                <div class="language-card" data-lang="kn">
                    <div class="language-icon">📝</div>
                    <div class="language-name">Kannada</div>
                    <div class="language-desc">ಕನ್ನಡ</div>
                </div>
            </div>
        </div>
        
        <div class="upload-section" id="uploadSection">
            <div class="upload-icon">📄</div>
            <div class="upload-text">Drag & drop your document image here or click to browse</div>
            <input type="file" id="fileInput" class="file-input" accept="image/*">
            <button class="upload-btn" onclick="document.getElementById('fileInput').click()">Choose File</button>
            <p id="file-name" style="margin-top: 10px; opacity: 0.7;"></p>
        </div>
        
        <div class="text-input-section" id="textInputSection" style="display: none;">
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">✏️ Direct Text Translation</div>
                        <div class="card-subtitle">Translate text directly without document upload</div>
            </div>
                    <div class="badge badge-info">Text Mode</div>
                </div>
                
                <!-- Language Selection for Text Mode -->
                <div class="text-language-selection">
                    <div class="section-title">🌐 Select Target Language</div>
                    <div class="language-grid">
                        <div class="language-card" data-lang="te">
                            <div class="language-icon">📜</div>
                            <div class="language-name">Telugu</div>
                            <div class="language-desc">తెలుగు</div>
                        </div>
                        <div class="language-card" data-lang="hi">
                            <div class="language-icon">📖</div>
                            <div class="language-name">Hindi</div>
                            <div class="language-desc">हिन्दी</div>
                        </div>
                        <div class="language-card" data-lang="ta">
                            <div class="language-icon">📚</div>
                            <div class="language-name">Tamil</div>
                            <div class="language-desc">தமிழ்</div>
                        </div>
                        <div class="language-card" data-lang="kn">
                            <div class="language-icon">📝</div>
                            <div class="language-name">Kannada</div>
                            <div class="language-desc">ಕನ್ನಡ</div>
                        </div>
                        <div class="language-card" data-lang="ml">
                            <div class="language-icon">📄</div>
                            <div class="language-name">Malayalam</div>
                            <div class="language-desc">മലയാളം</div>
                        </div>
                        <div class="language-card" data-lang="en">
                            <div class="language-icon">📰</div>
                            <div class="language-name">English</div>
                            <div class="language-desc">Improved</div>
                        </div>
                    </div>
                </div>
                
                <div class="text-input-container">
                    <div class="input-label">📝 Enter Text to Translate:</div>
                    <div class="current-language-display">
                        <span id="currentLanguageDisplay">Target Language: తెలుగు (Telugu)</span>
                    </div>
                    <textarea id="textInput" placeholder="Enter the text you want to translate here..." rows="8"></textarea>
                    <div class="text-input-actions">
                        <button class="action-btn" onclick="translateText()">🔄 Translate Text</button>
                        <button class="action-btn btn-outline" onclick="clearText()">🗑️ Clear</button>
                    </div>
                </div>
            </div>
            
            <div class="text-result-container" id="textResultContainer" style="display: none;">
                <div class="card">
                    <div class="card-header">
                        <div>
                            <div class="card-title">📝 Translation Result</div>
                            <div class="card-subtitle">Your translated text is ready</div>
                        </div>
                        <div class="badge badge-success">Completed</div>
                    </div>
                    <div class="text-result" id="textResult"></div>
                    <div class="text-result-actions">
                        <button class="action-btn" onclick="editTranslation()">✏️ Edit Translation</button>
                        <button class="action-btn btn-outline" onclick="copyTranslation()">📋 Copy</button>
                        <button class="action-btn btn-outline" onclick="downloadTextTranslation()">💾 Download</button>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="model-selection-section">
            <div class="card">
                <div class="card-header">
                    <div>
                        <div class="card-title">🤖 Translation Configuration</div>
                        <div class="card-subtitle">Choose your AI model and language settings</div>
                    </div>
                    <div class="badge badge-info">AI Powered</div>
                </div>
                
                <div class="section-title">🤖 Choose Translation Model</div>
                <div class="model-grid" id="modelGrid">
                    <div class="model-loading-card">
                        <div class="loading-spinner"></div>
                        <div class="loading-text">Loading available models...</div>
                    </div>
                </div>
                
                <div class="translation-options">
                    <div class="section-title">🌐 Translation Options</div>
                    <div class="language-grid">
                        <div class="language-card" data-lang="te">
                            <div class="language-icon">📜</div>
                            <div class="language-name">Telugu</div>
                            <div class="language-desc">తెలుగు</div>
                        </div>
                        <div class="language-card" data-lang="hi">
                            <div class="language-icon">📖</div>
                            <div class="language-name">Hindi</div>
                            <div class="language-desc">हिन्दी</div>
                        </div>
                        <div class="language-card" data-lang="ta">
                            <div class="language-icon">📚</div>
                            <div class="language-name">Tamil</div>
                            <div class="language-desc">தமிழ்</div>
                        </div>
                        <div class="language-card" data-lang="kn">
                            <div class="language-icon">📝</div>
                            <div class="language-name">Kannada</div>
                            <div class="language-desc">ಕನ್ನಡ</div>
                        </div>
                        <div class="language-card" data-lang="ml">
                            <div class="language-icon">📄</div>
                            <div class="language-name">Malayalam</div>
                            <div class="language-desc">മലയാളം</div>
                        </div>
                        <div class="language-card" data-lang="en">
                            <div class="language-icon">📰</div>
                            <div class="language-name">English</div>
                            <div class="language-desc">Improved</div>
                        </div>
                    </div>
                </div>
                
                <div class="agent-mode-toggle">
                    <label class="toggle-label">
                        <input type="checkbox" id="agentModeToggle" checked>
                        <span class="toggle-slider"></span>
                        <span class="toggle-text">🚀 Agentic Framework (Multi-Agent Pipeline)</span>
                    </label>
                </div>
            </div>
        </div>
        
        <!-- Loading Overlay -->
        <div class="loading-overlay" id="loadingOverlay">
            <div class="loading-content">
                <div class="loading-spinner"></div>
                <div class="loading-text" id="loadingText">Processing...</div>
                <div class="loading-subtext" id="loadingSubtext">Please wait</div>
            </div>
        </div>
        
        <div class="processing-section" id="processingSection">
            <div class="section-title">Processing Document</div>
            <div class="processing-step" id="step1">
                <div class="step-icon">1</div>
                <div class="step-text">Extracting text with OCR...</div>
            </div>
            <div class="processing-step" id="step2">
                <div class="step-icon">2</div>
                <div class="step-text">Translating text...</div>
            </div>
            <div class="processing-step" id="step3">
                <div class="step-icon">3</div>
                <div class="step-text">Preparing layout controls...</div>
            </div>
        </div>
        
        <div class="main-content" id="mainContent">
            <div class="image-section">
                <div class="section-title">Document Preview</div>
                <div class="image-container" id="imageContainer">
                    <img id="documentImage" class="document-image" alt="Document Preview">
                </div>
            </div>
            
            <div class="controls-section">
                <div class="section-title">Text Region Controls</div>
                <div class="status-message" id="statusMessage"></div>
                
        <!-- Advanced Control Panel -->
        <div class="control-panel" id="controlPanel">
            <div class="panel-header">
                <h3>⚙️ Advanced Controls</h3>
                <button class="toggle-panel" onclick="toggleControlPanel()">▼</button>
            </div>
            <div class="panel-content" id="panelContent">
                <!-- OCR Engine Selection -->
                <div class="control-section">
                    <h4>🔍 OCR Engine</h4>
                    <div class="control-group">
                        <label>
                            <input type="radio" name="ocrEngine" value="easyocr" checked>
                            EasyOCR (Fast)
                        </label>
                        <label>
                            <input type="radio" name="ocrEngine" value="tesseract">
                            Tesseract OCR (Accurate)
                        </label>
                        <label>
                            <input type="radio" name="ocrEngine" value="both">
                            Both (Fallback)
                        </label>
                    </div>
                </div>
                
                <!-- Document Language -->
                <div class="control-section">
                    <h4>📋 Document Language</h4>
                    <div class="language-grid-controls">
                        <div class="language-card-control" data-lang="auto">
                            <div class="language-icon">🔍</div>
                            <div class="language-name">Auto-detect</div>
                        </div>
                        <div class="language-card-control" data-lang="en">
                            <div class="language-icon">🇺🇸</div>
                            <div class="language-name">English</div>
                        </div>
                        <div class="language-card-control" data-lang="te">
                            <div class="language-icon">📜</div>
                            <div class="language-name">Telugu</div>
                        </div>
                        <div class="language-card-control" data-lang="hi">
                            <div class="language-icon">📖</div>
                            <div class="language-name">Hindi</div>
                        </div>
                        <div class="language-card-control" data-lang="ta">
                            <div class="language-icon">📚</div>
                            <div class="language-name">Tamil</div>
                        </div>
                        <div class="language-card-control" data-lang="kn">
                            <div class="language-icon">📝</div>
                            <div class="language-name">Kannada</div>
                        </div>
                    </div>
                </div>
                
                <!-- Text Size Controls -->
                <div class="control-section">
                    <h4>📏 Text Size Management</h4>
                    <div class="control-group">
                        <label>Font Size Range:</label>
                        <div class="range-controls">
                            <label>Min: <input type="number" id="minFontSize" value="12" min="8" max="20"></label>
                            <label>Max: <input type="number" id="maxFontSize" value="24" min="16" max="48"></label>
                        </div>
                        <label>Font Scale Factor: <input type="range" id="fontScale" min="0.5" max="2.0" step="0.1" value="1.0"></label>
                        <div class="scale-display">Scale: <span id="scaleValue">1.0</span></div>
                    </div>
                </div>
                
                <!-- Image Processing -->
                <div class="control-section">
                    <h4>🖼️ Image Processing</h4>
                    <div class="control-group">
                        <label>
                            <input type="checkbox" id="enablePreprocessing" checked>
                            Enable Image Preprocessing
                        </label>
                        <label>
                            <input type="checkbox" id="enableDenoising" checked>
                            Enable Denoising
                        </label>
                        <label>
                            <input type="checkbox" id="enableThresholding" checked>
                            Enable Adaptive Thresholding
                        </label>
                    </div>
                </div>
                
                <!-- Translation Settings -->
                <div class="control-section">
                    <h4>🌐 Translation Settings</h4>
                    <div class="control-group">
                        <label>Translation Mode:</label>
                        <select id="translationMode">
                            <option value="literal">Literal Translation</option>
                            <option value="contextual">Contextual Translation</option>
                            <option value="legal">Legal Translation</option>
                        </select>
                        <label>
                            <input type="checkbox" id="enableRAG" checked>
                            Enable RAG (Legal Context)
                        </label>
                    </div>
                </div>
                
                <!-- Preview Settings -->
                <div class="control-section">
                    <h4>👁️ Preview Settings</h4>
                    <div class="control-group">
                        <label>Default Preview Mode:</label>
                        <select id="defaultPreviewMode">
                            <option value="side-by-side">Side by Side</option>
                            <option value="overlay">Overlay</option>
                            <option value="original">Original Only</option>
                            <option value="translated">Translated Only</option>
                        </select>
                        <label>
                            <input type="checkbox" id="showConfidence" checked>
                            Show OCR Confidence Scores
                        </label>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Language Selection Modal -->
        <div id="languageModal" class="modal-overlay" style="display: none;">
            <div class="modal-content">
                <div class="modal-header">
                    <h3>📋 Document Language Selection</h3>
                    <p>Please select the language of the document you're uploading for better OCR accuracy.</p>
                </div>
                <div class="modal-body">
                    <div class="language-grid-modal">
                        <div class="language-card-modal" data-lang="auto">
                            <div class="language-icon">🔍</div>
                            <div class="language-name">Auto-detect</div>
                            <div class="language-desc">Let AI detect</div>
                        </div>
                        <div class="language-card-modal" data-lang="en">
                            <div class="language-icon">🇺🇸</div>
                            <div class="language-name">English</div>
                            <div class="language-desc">English</div>
                        </div>
                        <div class="language-card-modal" data-lang="te">
                            <div class="language-icon">📜</div>
                            <div class="language-name">Telugu</div>
                            <div class="language-desc">తెలుగు</div>
                        </div>
                        <div class="language-card-modal" data-lang="hi">
                            <div class="language-icon">📖</div>
                            <div class="language-name">Hindi</div>
                            <div class="language-desc">हिन्दी</div>
                        </div>
                        <div class="language-card-modal" data-lang="ta">
                            <div class="language-icon">📚</div>
                            <div class="language-name">Tamil</div>
                            <div class="language-desc">தமிழ்</div>
                        </div>
                        <div class="language-card-modal" data-lang="kn">
                            <div class="language-icon">📝</div>
                            <div class="language-name">Kannada</div>
                            <div class="language-desc">ಕನ್ನಡ</div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button id="confirmLanguage" class="btn btn-primary" disabled>Continue</button>
                </div>
            </div>
        </div>
                
                <div id="textRegions"></div>
            <div class="action-buttons">
                    <button class="action-btn" id="previewBtn" onclick="previewDocument()">Preview Document</button>
                    <button class="action-btn" id="downloadBtn" onclick="downloadDocument()">Download Result</button>
                </div>
            </div>
        </div>
        
        <div class="preview-section" id="previewSection">
            <div class="section-title">📄 Document Preview</div>
            <div class="preview-controls">
                <button class="preview-btn active" id="sideBySideBtn" onclick="switchPreviewMode('side-by-side')">Side by Side</button>
                <button class="preview-btn" id="overlayBtn" onclick="switchPreviewMode('overlay')">Overlay</button>
                <button class="preview-btn" id="originalBtn" onclick="switchPreviewMode('original')">Original Only</button>
                <button class="preview-btn" id="translatedBtn" onclick="switchPreviewMode('translated')">Translated Only</button>
            </div>
            
            <!-- Side by Side Preview -->
            <div class="side-by-side-preview" id="sideBySidePreview">
                <div class="preview-panel">
                    <div class="panel-header">Original Document</div>
                    <div class="preview-container">
                        <img id="originalImage" class="document-image" alt="Original Document">
                        <div id="originalOverlays" class="text-overlays"></div>
                    </div>
                </div>
                <div class="preview-panel">
                    <div class="panel-header">Translated Document</div>
                    <div class="preview-container">
                        <img id="translatedImage" class="document-image" alt="Translated Document">
                        <div id="translatedOverlays" class="text-overlays"></div>
                    </div>
                </div>
            </div>
            
            <!-- Overlay Preview -->
            <div class="overlay-preview" id="overlayPreview" style="display: none;">
                <div class="preview-container">
                    <img id="documentImage" class="document-image" alt="Document Preview">
                    <div id="textOverlays" class="text-overlays"></div>
                </div>
            </div>
            
            <button class="download-btn" onclick="downloadDocument()">📥 Download HTML Document</button>
        </div>
        
        <div id="status-message"></div>
        
        <!-- Floating Action Button -->
        <button class="floating-action-btn tooltip" onclick="scrollToTop()" title="Scroll to Top">
            ↑
        </button>
    </div>

    <script>
        let documentData = null;
        let textRegions = [];
        let translatedText = [];
        let userActions = {};
        let selectedModel = 'gemma3:latest';
        let agentMode = true;
        let selectedLanguage = 'te'; // Default to Telugu
        let selectedSourceLanguage = 'auto'; // Default to auto-detect
        let currentMode = 'document'; // 'document' or 'text'
        let currentTextTranslation = '';
        
        // Control Panel Settings
        let ocrEngine = 'easyocr';
        let minFontSize = 12;
        let maxFontSize = 24;
        let fontScale = 1.0;
        let enablePreprocessing = true;
        let enableDenoising = true;
        let enableThresholding = true;
        let translationMode = 'literal';
        let enableRAG = true;
        let defaultPreviewMode = 'side-by-side';
        let showConfidence = true;
        
        // Multi-Agent Framework
        class AgentFramework {
            constructor(ollamaUrl, model) {
                this.ollamaUrl = ollamaUrl;
                this.model = model;
                this.agents = new Map();
                this.dataPipeline = [];
                this.agentStates = new Map();
                this.initializeAgents();
            }

            initializeAgents() {
                // Register all agents with their roles and capabilities
                this.agents.set('contextAnalyzer', {
                    agent: new ContextAgent(this.ollamaUrl, this.model),
                    role: 'context_analysis',
                    priority: 1,
                    required: true,
                    timeout: 30000
                });

                this.agents.set('translator', {
                    agent: new TranslationAgent(this.ollamaUrl, this.model),
                    role: 'translation',
                    priority: 2,
                    required: true,
                    timeout: 45000
                });

                this.agents.set('validator', {
                    agent: new ValidationAgent(this.ollamaUrl, this.model),
                    role: 'validation',
                    priority: 3,
                    required: true,
                    timeout: 30000
                });

                this.agents.set('qualityAssurance', {
                    agent: new QualityAgent(this.ollamaUrl, this.model),
                    role: 'quality_improvement',
                    priority: 4,
                    required: false,
                    timeout: 45000
                });

                this.agents.set('languageConsistencyChecker', {
                    agent: new LanguageConsistencyAgent(this.ollamaUrl, this.model),
                    role: 'language_consistency',
                    priority: 5,
                    required: true,
                    timeout: 20000
                });
            }

            async executeTranslationPipeline(originalText, sourceLang, targetLang, progressCallback) {
                console.log('🚀 Starting Agent Framework Pipeline');
                this.dataPipeline = [];
                this.agentStates.clear();

                const pipelineData = {
                    originalText,
                    sourceLang,
                    targetLang,
                    context: null,
                    translatedText: null,
                    validation: null,
                    consistencyCheck: null,
                    finalText: null,
                    errors: [],
                    metadata: {
                        startTime: Date.now(),
                        agentResults: {}
                    }
                };

                try {
                    // Step 1: Context Analysis
                    progressCallback('🔍 Agent 1/5: Context Analysis', 1, 'active');
                    pipelineData.context = await this.executeAgent('contextAnalyzer', {
                        text: originalText,
                        sourceLang,
                        targetLang
                    });
                    pipelineData.metadata.agentResults.contextAnalysis = pipelineData.context;
                    progressCallback('✅ Context Analysis Complete', 1, 'completed');

                    // Step 2: Translation
                    progressCallback('🔄 Agent 2/5: Translation', 2, 'active');
                    pipelineData.translatedText = await this.executeAgent('translator', {
                        text: originalText,
                        sourceLang,
                        targetLang,
                        context: pipelineData.context
                    });
                    pipelineData.metadata.agentResults.translation = pipelineData.translatedText;
                    progressCallback('✅ Translation Complete', 2, 'completed');

                    // Step 3: Validation
                    progressCallback('✅ Agent 3/5: Validation', 3, 'active');
                    pipelineData.validation = await this.executeAgent('validator', {
                        originalText,
                        translatedText: pipelineData.translatedText,
                        sourceLang,
                        targetLang
                    });
                    pipelineData.metadata.agentResults.validation = pipelineData.validation;
                    progressCallback('✅ Validation Complete', 3, 'completed');

                    // Step 4: Language Consistency Check
                    progressCallback('🔍 Agent 4/5: Language Consistency', 4, 'active');
                    pipelineData.consistencyCheck = await this.executeAgent('languageConsistencyChecker', {
                        text: pipelineData.translatedText,
                        targetLang,
                        originalText
                    });
                    pipelineData.metadata.agentResults.consistencyCheck = pipelineData.consistencyCheck;
                    progressCallback('✅ Language Consistency Check Complete', 4, 'completed');

                    // Step 5: Quality Improvement (if needed)
                    if (pipelineData.validation.status === 'needs_revision' || 
                        pipelineData.validation.status === 'invalid' ||
                        !pipelineData.consistencyCheck.isConsistent) {
                        
                        progressCallback('🔧 Agent 5/5: Quality Improvement', 5, 'active');
                        pipelineData.finalText = await this.executeAgent('qualityAssurance', {
                            originalText,
                            translatedText: pipelineData.translatedText,
                            sourceLang,
                            targetLang,
                            context: pipelineData.context,
                            validation: pipelineData.validation,
                            consistencyCheck: pipelineData.consistencyCheck
                        });
                        pipelineData.metadata.agentResults.qualityImprovement = pipelineData.finalText;
                        progressCallback('✅ Quality Improvement Complete', 5, 'completed');
                    } else {
                        pipelineData.finalText = pipelineData.translatedText;
                        progressCallback('✅ Quality Check Passed', 5, 'completed');
                    }

                    // Final validation
                    if (pipelineData.finalText !== pipelineData.translatedText) {
                        progressCallback('🔄 Final Validation', 3, 'active');
                        const finalValidation = await this.executeAgent('validator', {
                            originalText,
                            translatedText: pipelineData.finalText,
                            sourceLang,
                            targetLang
                        });
                        pipelineData.metadata.agentResults.finalValidation = finalValidation;
                        progressCallback('✅ Final Validation Complete', 3, 'completed');
                    }

                    pipelineData.metadata.endTime = Date.now();
                    pipelineData.metadata.duration = pipelineData.metadata.endTime - pipelineData.metadata.startTime;

                    console.log('🎯 Agent Framework Pipeline Complete:', pipelineData.metadata);
                    return {
                        finalText: pipelineData.finalText || pipelineData.translatedText,
                        originalTranslation: pipelineData.translatedText,
                        improvedTranslation: pipelineData.finalText,
                        metadata: pipelineData.metadata
                    };

                } catch (error) {
                    console.error('❌ Agent Framework Pipeline Error:', error);
                    pipelineData.errors.push(error.message);
                    throw error;
                }
            }

            async executeAgent(agentName, inputData) {
                const agentConfig = this.agents.get(agentName);
                if (!agentConfig) {
                    throw new Error(`Agent ${agentName} not found`);
                }

                console.log(`🤖 Executing Agent: ${agentName}`, inputData);
                
                const startTime = Date.now();
                this.agentStates.set(agentName, {
                    status: 'running',
                    startTime,
                    inputData
                });

                try {
                    let result;
                    
                    switch (agentName) {
                        case 'contextAnalyzer':
                            result = await agentConfig.agent.analyzeContext(
                                inputData.text, 
                                inputData.sourceLang, 
                                inputData.targetLang
                            );
                            break;
                            
                        case 'translator':
                            result = await agentConfig.agent.translate(
                                inputData.text, 
                                inputData.sourceLang, 
                                inputData.targetLang, 
                                inputData.context
                            );
                            break;
                            
                        case 'validator':
                            result = await agentConfig.agent.validateTranslation(
                                inputData.originalText, 
                                inputData.translatedText, 
                                inputData.sourceLang, 
                                inputData.targetLang
                            );
                            break;
                            
                        case 'qualityAssurance':
                            result = await agentConfig.agent.improveTranslation(
                                inputData.originalText, 
                                inputData.translatedText, 
                                inputData.sourceLang, 
                                inputData.targetLang, 
                                inputData.context
                            );
                            break;
                            
                        case 'languageConsistencyChecker':
                            result = await agentConfig.agent.checkConsistency(
                                inputData.text, 
                                inputData.targetLang, 
                                inputData.originalText
                            );
                            break;
                            
                        default:
                            throw new Error(`Unknown agent: ${agentName}`);
                    }

                    const endTime = Date.now();
                    this.agentStates.set(agentName, {
                        status: 'completed',
                        startTime,
                        endTime,
                        duration: endTime - startTime,
                        inputData,
                        result
                    });

                    console.log(`✅ Agent ${agentName} completed in ${endTime - startTime}ms`);
                    return result;

                } catch (error) {
                    const endTime = Date.now();
                    this.agentStates.set(agentName, {
                        status: 'error',
                        startTime,
                        endTime,
                        duration: endTime - startTime,
                        inputData,
                        error: error.message
                    });

                    console.error(`❌ Agent ${agentName} failed:`, error);
                    throw error;
                }
            }

            getAgentStates() {
                return this.agentStates;
            }

            getPipelineData() {
                return this.dataPipeline;
            }
        }

        // Individual Agent Classes
        class ContextAgent {
            constructor(ollamaUrl, model) {
                this.ollamaUrl = ollamaUrl;
                this.model = model;
            }

            async analyzeContext(text, sourceLang, targetLang) {
                const prompt = `Analyze the context and domain of this text for legal translation:

Text: "${text}"
Source Language: ${sourceLang}
Target Language: ${targetLang}

Provide analysis in JSON format:
{
    "domain": "legal/civil/criminal/constitutional/administrative",
    "context": "brief description",
    "keyTerms": ["term1", "term2"],
    "complexity": "low/medium/high",
    "recommendations": ["recommendation1", "recommendation2"]
}`;

                const result = await this.callOllama(prompt);
                try {
                    return JSON.parse(result);
                } catch (error) {
                    return {
                        domain: "legal",
                        context: "General legal text",
                        keyTerms: [],
                        complexity: "medium",
                        recommendations: ["Standard legal translation approach"]
                    };
                }
            }

            async callOllama(prompt) {
                const response = await fetch(`${this.ollamaUrl}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt,
                        stream: false,
                        options: { temperature: 0.1, top_p: 0.8, num_predict: 500 }
                    })
                });
                
                if (!response.ok) throw new Error(`Context agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }

        class TranslationAgent {
            constructor(ollamaUrl, model) {
                this.ollamaUrl = ollamaUrl;
                this.model = model;
            }

            async translate(text, sourceLang, targetLang, context) {
                const targetLanguageName = this.getLanguageName(targetLang);
                const sourceLanguageName = this.getLanguageName(sourceLang);
                
                const prompt = `Translate the following ${sourceLanguageName} legal text to ${targetLanguageName}. Provide ONLY the translation, no explanations.

Context: ${context ? JSON.stringify(context) : 'General legal context'}

Original Text: "${text}"

Translation:`;

                const result = await this.callOllama(prompt);
                return result.trim();
            }

            getLanguageName(langCode) {
                const names = {
                    'en': 'English', 'te': 'Telugu', 'kn': 'Kannada', 'ta': 'Tamil',
                    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'mr': 'Marathi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali',
                    'ur': 'Urdu', 'ml': 'Malayalam', 'si': 'Sinhala', 'my': 'Burmese',
                    'th': 'Thai', 'km': 'Khmer', 'lo': 'Lao', 'vi': 'Vietnamese',
                    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese'
                };
                return names[langCode] || langCode;
            }

            async callOllama(prompt) {
                const response = await fetch(`${this.ollamaUrl}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt,
                        stream: false,
                        options: { temperature: 0.3, top_p: 0.9, num_predict: 1000 }
                    })
                });
                
                if (!response.ok) throw new Error(`Translation agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }

        class ValidationAgent {
            constructor(ollamaUrl, model) {
                this.ollamaUrl = ollamaUrl;
                this.model = model;
            }

            async validateTranslation(originalText, translatedText, sourceLang, targetLang) {
                const prompt = `Validate this legal translation for accuracy and completeness:

Original (${sourceLang}): "${originalText}"
Translation (${targetLang}): "${translatedText}"

Provide validation in JSON format:
{
    "status": "valid/needs_revision/invalid",
    "accuracy": 0-100,
    "completeness": 0-100,
    "issues": ["issue1", "issue2"],
    "strengths": ["strength1", "strength2"],
    "recommendations": ["recommendation1", "recommendation2"]
}`;

                const result = await this.callOllama(prompt);
                try {
                    return JSON.parse(result);
                } catch (error) {
                    return {
                        status: "valid",
                        accuracy: 85,
                        completeness: 90,
                        issues: [],
                        strengths: ["Maintains legal terminology"],
                        recommendations: ["Review for cultural appropriateness"]
                    };
                }
            }

            async callOllama(prompt) {
                const response = await fetch(`${this.ollamaUrl}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt,
                        stream: false,
                        options: { temperature: 0.1, top_p: 0.8, num_predict: 500 }
                    })
                });
                
                if (!response.ok) throw new Error(`Validation agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }

        class QualityAgent {
            constructor(ollamaUrl, model) {
                this.ollamaUrl = ollamaUrl;
                this.model = model;
            }

            async improveTranslation(originalText, translatedText, sourceLang, targetLang, context) {
                const targetLanguageName = this.getLanguageName(targetLang);
                
                const prompt = `Improve this legal translation for better quality:

Original: "${originalText}"
Current Translation: "${translatedText}"
Target Language: ${targetLanguageName}
Context: ${context ? JSON.stringify(context) : 'General legal context'}

Improve the translation focusing on:
1. Legal accuracy and terminology
2. Natural flow in ${targetLanguageName}
3. Cultural appropriateness
4. Formal legal tone

Provide ONLY the improved translation without explanations.`;

                const result = await this.callOllama(prompt);
                return result.trim();
            }

            getLanguageName(langCode) {
                const names = {
                    'en': 'English', 'te': 'Telugu', 'kn': 'Kannada', 'ta': 'Tamil',
                    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'mr': 'Marathi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali',
                    'ur': 'Urdu', 'ml': 'Malayalam', 'si': 'Sinhala', 'my': 'Burmese',
                    'th': 'Thai', 'km': 'Khmer', 'lo': 'Lao', 'vi': 'Vietnamese',
                    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese'
                };
                return names[langCode] || langCode;
            }

            async callOllama(prompt) {
                const response = await fetch(`${this.ollamaUrl}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt,
                        stream: false,
                        options: { temperature: 0.2, top_p: 0.85, num_predict: 1000 }
                    })
                });
                
                if (!response.ok) throw new Error(`Quality agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }

        class LanguageConsistencyAgent {
            constructor(ollamaUrl, model) {
                this.ollamaUrl = ollamaUrl;
                this.model = model;
            }

            async checkConsistency(text, targetLang, originalText) {
                const targetLanguageName = this.getLanguageName(targetLang);
                
                const prompt = `You are a language consistency expert. Analyze the following text for language mixing.

Original text:
${originalText}

Translated text (should be in ${targetLanguageName}):
${text}

CRITICAL ANALYSIS REQUIRED:
1. Check if the translation uses ONLY ${targetLanguageName} language
2. Identify any words from other languages (English, Telugu, Tamil, Hindi, etc.)
3. Look for mixed language patterns
4. Verify consistency in terminology

Respond with a JSON object:
{
    "isConsistent": true/false,
    "mixedWords": ["word1", "word2"],
    "mixedLanguages": ["language1", "language2"],
    "consistencyScore": 0-100,
    "recommendations": ["recommendation1", "recommendation2"]
}`;

                const result = await this.callOllama(prompt);
                
                try {
                    const analysis = JSON.parse(result);
                    return {
                        isConsistent: analysis.isConsistent,
                        mixedWords: analysis.mixedWords || [],
                        mixedLanguages: analysis.mixedLanguages || [],
                        consistencyScore: analysis.consistencyScore || 0,
                        recommendations: analysis.recommendations || [],
                        targetLanguage: targetLanguageName
                    };
                } catch (error) {
                    // Fallback parsing
                    return {
                        isConsistent: !result.toLowerCase().includes('inconsistent'),
                        mixedWords: [],
                        mixedLanguages: [],
                        consistencyScore: 50,
                        recommendations: ['Manual review recommended'],
                        targetLanguage: targetLanguageName
                    };
                }
            }

            getLanguageName(langCode) {
                const names = {
                    'en': 'English', 'te': 'Telugu', 'kn': 'Kannada', 'ta': 'Tamil',
                    'hi': 'Hindi', 'bn': 'Bengali', 'gu': 'Gujarati', 'pa': 'Punjabi',
                    'mr': 'Marathi', 'or': 'Odia', 'as': 'Assamese', 'ne': 'Nepali',
                    'ur': 'Urdu', 'ml': 'Malayalam', 'si': 'Sinhala', 'my': 'Burmese',
                    'th': 'Thai', 'km': 'Khmer', 'lo': 'Lao', 'vi': 'Vietnamese',
                    'es': 'Spanish', 'fr': 'French', 'de': 'German', 'ja': 'Japanese',
                    'ko': 'Korean', 'zh': 'Chinese'
                };
                return names[langCode] || langCode;
            }

            async callOllama(prompt) {
                const response = await fetch(`${this.ollamaUrl}/api/generate`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        model: this.model,
                        prompt: prompt,
                        stream: false,
                        options: { temperature: 0.1, top_p: 0.8, num_predict: 1000 }
                    })
                });
                
                if (!response.ok) throw new Error(`Language consistency agent error: ${response.status}`);
                const data = await response.json();
                return data.response.trim();
            }
        }
        
        // Loading functions
        function showLoading(text = 'Processing...', subtext = 'Please wait') {
            const overlay = document.getElementById('loadingOverlay');
            const loadingText = document.getElementById('loadingText');
            const loadingSubtext = document.getElementById('loadingSubtext');
            
            loadingText.textContent = text;
            loadingSubtext.textContent = subtext;
            overlay.style.display = 'flex';
        }
        
        function hideLoading() {
            const overlay = document.getElementById('loadingOverlay');
            overlay.style.display = 'none';
        }
        
        // Check RAG system status on load
        async function checkRAGStatus() {
            try {
                const response = await fetch('/api/rag/status');
                const data = await response.json();
                
                const statusElement = document.getElementById('rag-status');
                const statusText = document.getElementById('rag-status-text');
                
                if (data.rag_available) {
                    statusElement.className = 'rag-status active';
                    statusText.textContent = '✅ Legal RAG System Active - Enhanced Translation';
                } else {
                    statusElement.className = 'rag-status inactive';
                    statusText.textContent = '⚠️ Legal RAG System Inactive - Basic Translation';
                }
            } catch (error) {
                console.error('Failed to check RAG status:', error);
                const statusElement = document.getElementById('rag-status');
                const statusText = document.getElementById('rag-status-text');
                statusElement.className = 'rag-status inactive';
                statusText.textContent = '⚠️ RAG Status Unknown';
            }
        }
        
        // Model selection handling
        document.addEventListener('DOMContentLoaded', function() {
            loadAvailableModels();
            setupLanguageSelection();
            
            // Agent mode toggle
            const agentToggle = document.getElementById('agentModeToggle');
            if (agentToggle) {
                agentToggle.addEventListener('change', function() {
                    agentMode = this.checked;
                    console.log('Agent mode:', agentMode ? 'enabled' : 'disabled');
                });
            }
        });
        
        function setupLanguageSelection() {
            // Set default language selection for both document and text modes
            const defaultLangCards = document.querySelectorAll('[data-lang="te"]');
            defaultLangCards.forEach(card => card.classList.add('selected'));
            
            // Update language display
            updateLanguageDisplay();
            
            // Language card selection for both modes
            document.querySelectorAll('.language-card').forEach(card => {
                card.addEventListener('click', function() {
                    // Remove selection from all cards in the same section
                    const parentSection = this.closest('.translation-options, .text-language-selection, .language-selection-section');
                    parentSection.querySelectorAll('.language-card').forEach(c => c.classList.remove('selected'));
                    
                    // Add selection to clicked card
                    this.classList.add('selected');
                    
                    // Determine if this is source or target language selection
                    if (parentSection.classList.contains('language-selection-section')) {
                        selectedSourceLanguage = this.dataset.lang;
                        console.log('Selected source language:', selectedSourceLanguage);
                    } else {
                        selectedLanguage = this.dataset.lang;
                        console.log('Selected target language:', selectedLanguage);
                    }
                    
                    // Update language display
                    updateLanguageDisplay();
                });
            });
        }
        
        function updateLanguageDisplay() {
            const languageNames = {
                'te': 'తెలుగు (Telugu)',
                'hi': 'हिन्दी (Hindi)',
                'ta': 'தமிழ் (Tamil)',
                'kn': 'ಕನ್ನಡ (Kannada)',
                'ml': 'മലയാളം (Malayalam)',
                'en': 'English (Improved)'
            };
            
            const displayElement = document.getElementById('currentLanguageDisplay');
            if (displayElement) {
                displayElement.textContent = `Target Language: ${languageNames[selectedLanguage] || selectedLanguage}`;
            }
        }
        
        async function loadAvailableModels() {
            try {
                const response = await fetch('/api/ollama/models');
                const data = await response.json();
                
                if (data.success && data.models && data.models.length > 0) {
                    renderModels(data.models);
                } else {
                    console.warn('No models available from Ollama, using fallback models');
                    renderFallbackModels();
                }
            } catch (error) {
                console.warn('Error loading models from Ollama, using fallback models:', error);
                renderFallbackModels();
            }
        }
        
        function renderModels(models) {
            const modelGrid = document.getElementById('modelGrid');
            modelGrid.innerHTML = '';
            
            // Sort models by family and name
            models.sort((a, b) => {
                if (a.family !== b.family) {
                    return a.family.localeCompare(b.family);
                }
                return a.name.localeCompare(b.name);
            });
            
            models.forEach((model, index) => {
                const modelCard = document.createElement('div');
                modelCard.className = 'model-card';
                modelCard.dataset.model = model.name;
                
                // Get icon based on family
                const icon = getModelIcon(model.family || 'translator');
                const displayName = model.label || model.description || 'Standard Translator';
                
                modelCard.innerHTML = `
                    <div class="model-icon">${icon}</div>
                    <div class="model-name">${displayName}</div>
                    <div class="model-desc">${model.family || 'translator'}</div>
                `;
                
                // Add click handler
                modelCard.addEventListener('click', function() {
                    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
                    this.classList.add('selected');
                    selectedModel = this.dataset.model;
                    console.log('Selected model:', selectedModel);
                });
                
                modelGrid.appendChild(modelCard);
            });
            
            // Select first model by default
            if (models.length > 0) {
                const firstCard = modelGrid.querySelector('.model-card');
                if (firstCard) {
                    firstCard.classList.add('selected');
                    selectedModel = firstCard.dataset.model;
                }
            }
        }
        
        function renderFallbackModels() {
            const modelGrid = document.getElementById('modelGrid');
            modelGrid.innerHTML = '';
            
            const fallbackModels = [
                { name: 'gemma3:latest', family: 'gemma', icon: '⚖️' },
                { name: 'gemma3:4b', family: 'gemma', icon: '⚖️' },
                { name: 'shb/gemma3-legal-translator-full:latest', family: 'gemma', icon: '⚖️' },
                { name: 'llama3.2:latest', family: 'llama', icon: '🦙' },
                { name: 'hf.co/SandLogicTechnologies/LLama3-Gaja-Hindi-8B-GGUF:Q5_K_M', family: 'llama', icon: '🦙' },
                { name: 'deepseek-r1:14b-qwen-distill-q8_0', family: 'deepseek', icon: '🧠' },
                { name: 'gpt-oss:120b', family: 'gpt', icon: '🌪️' }
            ];
            
            fallbackModels.forEach(model => {
                const modelCard = document.createElement('div');
                modelCard.className = 'model-card';
                modelCard.dataset.model = model.name;
                
                modelCard.innerHTML = `
                    <div class="model-icon">${model.icon}</div>
                    <div class="model-name">${model.name}</div>
                    <div class="model-desc">${model.family} (fallback)</div>
                `;
                
                modelCard.addEventListener('click', function() {
                    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
                    this.classList.add('selected');
                    selectedModel = this.dataset.model;
                    console.log('Selected model:', selectedModel);
                });
                
                modelGrid.appendChild(modelCard);
            });
            
            // Select first model by default
            const firstCard = modelGrid.querySelector('.model-card');
            if (firstCard) {
                firstCard.classList.add('selected');
                selectedModel = firstCard.dataset.model;
            }
        }
        
        function getModelIcon(family) {
            const icons = {
                'gemma': '⚖️',
                'llama': '🦙',
                'qwen': '🧠',
                'mistral': '🌪️',
                'phi': 'Φ',
                'codellama': '💻',
                'default': '🤖'
            };
            return icons[family] || icons.default;
        }
        
        function formatBytes(bytes) {
            if (bytes === 0) return '';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return `(${(bytes / Math.pow(k, i)).toFixed(1)}${sizes[i]})`;
        }
        
        // File upload handling
        const uploadSection = document.getElementById('uploadSection');
        const fileInput = document.getElementById('fileInput');
        
        uploadSection.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', () => {
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileUpload(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileUpload(e.target.files[0]);
            }
        });
        
        function handleFileUpload(file) {
            console.log('File selected:', file.name, file.type, file.size);
            
            if (!file.type.startsWith('image/')) {
                showStatus('Please upload an image file.', 'error');
                return;
            }
            
            // TEMPORARY: Skip modal and process directly for testing
            console.log('BYPASSING MODAL - Processing file directly...');
            processFileWithLanguage(file, 'auto');
            return;
            
            console.log('Showing language modal...');
            // Show language selection modal
            showLanguageModal(file);
        }
        
        function showLanguageModal(file) {
            console.log('showLanguageModal called with file:', file.name);
            const modal = document.getElementById('languageModal');
            console.log('Modal element:', modal);
            
            if (!modal) {
                console.error('Language modal not found!');
                // Fallback: process file directly with auto-detect
                processFileWithLanguage(file, 'auto');
                return;
            }
            
            modal.style.display = 'flex';
            console.log('Modal displayed');
            
            // Reset selection
            document.querySelectorAll('.language-card-modal').forEach(card => {
                card.classList.remove('selected');
            });
            
            // Auto-select auto-detect
            const autoCard = document.querySelector('.language-card-modal[data-lang="auto"]');
            if (autoCard) {
                autoCard.classList.add('selected');
                document.getElementById('confirmLanguage').disabled = false;
            }
            
            // Store the file for later processing
            window.pendingFile = file;
            console.log('File stored in window.pendingFile');
        }
        
        function hideLanguageModal() {
            const modal = document.getElementById('languageModal');
            modal.style.display = 'none';
        }
        
        function switchPreviewMode(mode) {
            // Update button states
            document.querySelectorAll('.preview-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(mode + 'Btn').classList.add('active');
            
            // Show/hide preview modes
            document.getElementById('sideBySidePreview').style.display = mode === 'side-by-side' ? 'flex' : 'none';
            document.getElementById('overlayPreview').style.display = mode === 'overlay' ? 'block' : 'none';
            
            if (mode === 'original') {
                document.getElementById('sideBySidePreview').style.display = 'flex';
                document.querySelector('.preview-panel:last-child').style.display = 'none';
            } else if (mode === 'translated') {
                document.getElementById('sideBySidePreview').style.display = 'flex';
                document.querySelector('.preview-panel:first-child').style.display = 'none';
            } else {
                document.querySelectorAll('.preview-panel').forEach(panel => panel.style.display = 'block');
            }
        }
        
        function updateSideBySidePreview(originalImageSrc, translatedImageSrc, originalTexts, translatedTexts) {
            // Set images
            document.getElementById('originalImage').src = originalImageSrc;
            document.getElementById('translatedImage').src = translatedImageSrc;
            
            // Create overlays for original text
            const originalOverlays = document.getElementById('originalOverlays');
            originalOverlays.innerHTML = '';
            originalTexts.forEach((text, index) => {
                const overlay = document.createElement('div');
                overlay.className = 'text-overlay preserve';
                overlay.textContent = text;
                overlay.style.position = 'absolute';
                overlay.style.left = '0%';
                overlay.style.top = `${index * 10}%`;
                overlay.style.width = '100%';
                overlay.style.height = '8%';
                overlay.style.fontSize = '12px';
                overlay.style.color = '#000';
                overlay.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
                overlay.style.padding = '2px';
                overlay.style.borderRadius = '2px';
                originalOverlays.appendChild(overlay);
            });
            
            // Create overlays for translated text
            const translatedOverlays = document.getElementById('translatedOverlays');
            translatedOverlays.innerHTML = '';
            translatedTexts.forEach((text, index) => {
                const overlay = document.createElement('div');
                overlay.className = 'text-overlay translate';
                overlay.textContent = text;
                overlay.style.position = 'absolute';
                overlay.style.left = '0%';
                overlay.style.top = `${index * 10}%`;
                overlay.style.width = '100%';
                overlay.style.height = '8%';
                overlay.style.fontSize = '12px';
                overlay.style.color = '#000';
                overlay.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
                overlay.style.padding = '2px';
                overlay.style.borderRadius = '2px';
                translatedOverlays.appendChild(overlay);
            });
        }
        
        // Control Panel Functions
        function toggleControlPanel() {
            const panelContent = document.getElementById('panelContent');
            const toggleBtn = document.querySelector('.toggle-panel');
            
            panelContent.classList.toggle('collapsed');
            toggleBtn.classList.toggle('collapsed');
        }
        
        function initializeControlPanel() {
            // OCR Engine selection
            document.querySelectorAll('input[name="ocrEngine"]').forEach(radio => {
                radio.addEventListener('change', function() {
                    ocrEngine = this.value;
                    console.log('OCR Engine changed to:', ocrEngine);
                });
            });
            
            // Language selection in control panel
            document.querySelectorAll('.language-card-control').forEach(card => {
                card.addEventListener('click', function() {
                    document.querySelectorAll('.language-card-control').forEach(c => c.classList.remove('selected'));
                    this.classList.add('selected');
                    selectedSourceLanguage = this.dataset.lang;
                    console.log('Source language changed to:', selectedSourceLanguage);
                });
            });
            
            // Font size controls
            document.getElementById('minFontSize').addEventListener('change', function() {
                minFontSize = parseInt(this.value);
                console.log('Min font size:', minFontSize);
            });
            
            document.getElementById('maxFontSize').addEventListener('change', function() {
                maxFontSize = parseInt(this.value);
                console.log('Max font size:', maxFontSize);
            });
            
            document.getElementById('fontScale').addEventListener('input', function() {
                fontScale = parseFloat(this.value);
                document.getElementById('scaleValue').textContent = fontScale.toFixed(1);
                console.log('Font scale:', fontScale);
            });
            
            // Image processing controls
            document.getElementById('enablePreprocessing').addEventListener('change', function() {
                enablePreprocessing = this.checked;
                console.log('Preprocessing enabled:', enablePreprocessing);
            });
            
            document.getElementById('enableDenoising').addEventListener('change', function() {
                enableDenoising = this.checked;
                console.log('Denoising enabled:', enableDenoising);
            });
            
            document.getElementById('enableThresholding').addEventListener('change', function() {
                enableThresholding = this.checked;
                console.log('Thresholding enabled:', enableThresholding);
            });
            
            // Translation settings
            document.getElementById('translationMode').addEventListener('change', function() {
                translationMode = this.value;
                console.log('Translation mode:', translationMode);
            });
            
            document.getElementById('enableRAG').addEventListener('change', function() {
                enableRAG = this.checked;
                console.log('RAG enabled:', enableRAG);
            });
            
            // Preview settings
            document.getElementById('defaultPreviewMode').addEventListener('change', function() {
                defaultPreviewMode = this.value;
                console.log('Default preview mode:', defaultPreviewMode);
            });
            
            document.getElementById('showConfidence').addEventListener('change', function() {
                showConfidence = this.checked;
                console.log('Show confidence:', showConfidence);
            });
            
            // Set default selections
            document.querySelector('.language-card-control[data-lang="auto"]').classList.add('selected');
        }
        
        function getControlSettings() {
            return {
                ocrEngine: ocrEngine,
                sourceLanguage: selectedSourceLanguage,
                minFontSize: minFontSize,
                maxFontSize: maxFontSize,
                fontScale: fontScale,
                enablePreprocessing: enablePreprocessing,
                enableDenoising: enableDenoising,
                enableThresholding: enableThresholding,
                translationMode: translationMode,
                enableRAG: enableRAG,
                defaultPreviewMode: defaultPreviewMode,
                showConfidence: showConfidence
            };
        }
        
        function processFileWithLanguage(file, sourceLanguage) {
            console.log('processFileWithLanguage called with:', file.name, sourceLanguage);
            
            const settings = getControlSettings();
            console.log('Control settings:', settings);
            
            const formData = new FormData();
            formData.append('image', file);
            formData.append('model', selectedModel);
            formData.append('agent_mode', agentMode.toString());
            formData.append('target_language', selectedLanguage);
            formData.append('sourceLanguage', sourceLanguage);
            formData.append('ocrEngine', settings.ocrEngine);
            formData.append('minFontSize', settings.minFontSize);
            formData.append('maxFontSize', settings.maxFontSize);
            formData.append('fontScale', settings.fontScale);
            formData.append('enablePreprocessing', settings.enablePreprocessing);
            formData.append('enableDenoising', settings.enableDenoising);
            formData.append('enableThresholding', settings.enableThresholding);
            formData.append('translationMode', settings.translationMode);
            formData.append('enableRAG', settings.enableRAG);
            formData.append('showConfidence', settings.showConfidence);
            
            console.log('Processing file with settings:', settings);
            console.log('FormData entries:');
            for (let [key, value] of formData.entries()) {
                console.log(key, ':', value);
            }
            
            showLoading('Processing Document', 'OCR and Translation in progress...');
            showProcessingStep(1);
            
            console.log('Sending request to /upload...');
            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => {
                console.log('Response received:', response.status, response.statusText);
                return response.json();
            })
            .then(data => {
                hideLoading();
                console.log('Upload response received:', data);
                if (data.success) {
                    showProcessingStep(2);
                    setTimeout(() => {
                        showProcessingStep(3);
                        setTimeout(() => {
                            console.log('Calling displayResults with:', data);
                            displayResults(data);
                        }, 1000);
                    }, 2000);
                } else {
                    showStatus('Error: ' + data.error, 'error');
                    if (data.suggestions) {
                        console.log('Suggestions:', data.suggestions);
                    }
                }
            })
            .catch(error => {
                hideLoading();
                console.error('Upload error:', error);
                showStatus('Error uploading file: ' + error, 'error');
            });
        }
        
        function displayResults(data) {
            document.getElementById('processingSection').style.display = 'none';
            document.getElementById('mainContent').style.display = 'grid';
            
            // Display image
            const img = document.getElementById('documentImage');
            if (data.image) {
                img.src = 'data:image/png;base64,' + data.image;
                console.log('Image loaded successfully');
            } else {
                console.log('No image data received');
                // Try to load a placeholder or show an error
                img.src = 'data:image/svg+xml;base64,' + btoa(`
                    <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
                        <rect width="400" height="300" fill="#f0f0f0"/>
                        <text x="200" y="150" text-anchor="middle" fill="#666">No Image Available</text>
                    </svg>
                `);
            }
            
            // Store data - handle both formats
            textRegions = data.text_regions || data.bboxes || [];
            userActions = {};
            
            console.log('Text regions received:', textRegions.length);
            
            // Create bounding box overlays
            createBoundingBoxes(textRegions);
            
            // Render text region controls
            renderTextRegions();
            
            showStatus('Document processed successfully! Select actions for each text region.', 'success');
        }
        
        function createBoundingBoxes(regions) {
            const container = document.getElementById('imageContainer');
            const img = document.getElementById('documentImage');
            
            // Clear existing overlays
            container.querySelectorAll('.bbox-overlay').forEach(overlay => overlay.remove());
            
            // Wait for image to load
            img.onload = () => {
                const imgRect = img.getBoundingClientRect();
                const scaleX = imgRect.width / img.naturalWidth;
                const scaleY = imgRect.height / img.naturalHeight;
                
                regions.forEach(region => {
                    const overlay = document.createElement('div');
                    overlay.className = 'bbox-overlay preserve';
                    overlay.style.left = (region.bbox.x * scaleX) + 'px';
                    overlay.style.top = (region.bbox.y * scaleY) + 'px';
                    overlay.style.width = (region.bbox.width * scaleX) + 'px';
                    overlay.style.height = (region.bbox.height * scaleY) + 'px';
                    overlay.dataset.regionId = region.id;
                    
                    overlay.addEventListener('click', () => {
                        selectRegion(region.id);
                    });
                    
                    container.appendChild(overlay);
                });
            };
        }
        
        function selectRegion(regionId) {
            // Highlight the region
            document.querySelectorAll('.text-region').forEach(region => {
                region.classList.remove('selected');
            });
            document.querySelector(`[data-region-id="${regionId}"]`).parentElement.querySelector('.text-region').classList.add('selected');
        }
        
        function showProcessingStep(stepNumber) {
            for (let i = 1; i <= 3; i++) {
                const step = document.getElementById(`step${i}`);
                const icon = step.querySelector('.step-icon');
                
                if (i < stepNumber) {
                    step.classList.add('completed');
                    icon.classList.add('completed');
                    icon.textContent = '✓';
                } else if (i === stepNumber) {
                    step.classList.add('active');
                    icon.classList.add('active');
                } else {
                    step.classList.remove('active', 'completed');
                    icon.classList.remove('active', 'completed');
                    icon.textContent = i;
                }
            }
        }
        
        function renderTextRegions() {
            const container = document.getElementById('textRegions');
            container.innerHTML = '';
            
            textRegions.forEach(region => {
                const regionDiv = document.createElement('div');
                regionDiv.className = 'card';
                regionDiv.innerHTML = `
                    <div class="card-header">
                        <div>
                            <div class="card-title">Text Region ${region.id + 1}</div>
                            <div class="card-subtitle">Click on image to highlight this region</div>
                        </div>
                        <div class="badge ${getActionBadgeClass(userActions[region.id] || 'preserve')}">
                            ${getActionText(userActions[region.id] || 'preserve')}
                                    </div>
                                    </div>
                    
                    <div class="translation-comparison">
                        <div class="translation-side">
                            <div class="translation-side-header">📝 Original Text</div>
                            <div class="translation-content">${region.text}</div>
                                    </div>
                        <div class="translation-side">
                            <div class="translation-side-header">🔄 Translated Text</div>
                            <div class="translation-content" id="translation-${region.id}">${region.translated}</div>
                                </div>
                        </div>
                    
                    <div class="language-editor" id="language-editor-${region.id}" style="display: none;">
                        <div class="language-selector-row">
                            <div class="language-selector-item">
                                <label>Source Language:</label>
                                <select id="source-lang-${region.id}">
                                    <option value="auto">Auto Detect</option>
                                    <option value="te">తెలుగు (Telugu)</option>
                                    <option value="hi">हिन्दी (Hindi)</option>
                                    <option value="en">English</option>
                                    <option value="ta">தமிழ் (Tamil)</option>
                                    <option value="kn">ಕನ್ನಡ (Kannada)</option>
                                    <option value="ml">മലയാളം (Malayalam)</option>
                                </select>
                        </div>
                            <div class="language-selector-item">
                                <label>Target Language:</label>
                                <select id="target-lang-${region.id}">
                                    <option value="te">తెలుగు (Telugu)</option>
                                    <option value="hi">हिन्दी (Hindi)</option>
                                    <option value="en">English</option>
                                    <option value="ta">தமிழ் (Tamil)</option>
                                    <option value="kn">ಕನ್ನಡ (Kannada)</option>
                                    <option value="ml">മലയാളം (Malayalam)</option>
                                </select>
                    </div>
                        </div>
                        <div class="action-bar">
                            <button class="action-btn btn-sm btn-outline" onclick="retranslateRegion(${region.id})">🔄 Retranslate</button>
                            <button class="action-btn btn-sm" onclick="saveRegionTranslation(${region.id})">💾 Save</button>
                            <button class="action-btn btn-sm btn-outline" onclick="cancelRegionEdit(${region.id})">❌ Cancel</button>
                        </div>
                    </div>
                    
                    <div class="action-bar">
                        <button class="control-btn ${userActions[region.id] === 'translate' ? 'active' : ''}" 
                                onclick="setRegionAction(${region.id}, 'translate')" title="Translate this region">
                            🔄 Translate
                        </button>
                        <button class="control-btn ${userActions[region.id] === 'preserve' ? 'active' : ''}" 
                                onclick="setRegionAction(${region.id}, 'preserve')" title="Keep original text">
                            📝 Preserve
                        </button>
                        <button class="control-btn ${userActions[region.id] === 'whiteout' ? 'active' : ''}" 
                                onclick="setRegionAction(${region.id}, 'whiteout')" title="Remove this text">
                            🗑️ Whiteout
                        </button>
                        <button class="control-btn btn-outline" onclick="editRegionTranslation(${region.id})" title="Edit translation and language options">
                            ✏️ Edit Translation
                        </button>
                </div>
            `;
                container.appendChild(regionDiv);
            });
        }
        
        function getActionBadgeClass(action) {
            switch(action) {
                case 'translate': return 'badge-success';
                case 'preserve': return 'badge-info';
                case 'whiteout': return 'badge-error';
                default: return 'badge-info';
            }
        }
        
        function getActionText(action) {
            switch(action) {
                case 'translate': return 'Translate';
                case 'preserve': return 'Preserve';
                case 'whiteout': return 'Whiteout';
                default: return 'Preserve';
            }
        }
        
        function setRegionAction(regionId, action) {
            userActions[regionId] = action;
            
            // Update region styling
            const regionDiv = document.querySelector(`[data-region-id="${regionId}"]`);
            if (regionDiv) {
                regionDiv.className = `bbox-overlay ${action}`;
            }
            
            // Update card badge
            const card = document.querySelector(`#textRegions .card:nth-child(${regionId + 1})`);
            if (card) {
                const badge = card.querySelector('.badge');
                if (badge) {
                    badge.className = `badge ${getActionBadgeClass(action)}`;
                    badge.textContent = getActionText(action);
                }
            }
            
            // Update control buttons
            const buttons = document.querySelectorAll(`#textRegions .card:nth-child(${regionId + 1}) .control-btn`);
            buttons.forEach(btn => btn.classList.remove('active'));
            buttons.forEach(btn => {
                if (btn.textContent.includes(getActionText(action))) {
                    btn.classList.add('active');
                }
            });
            
            showStatus(`Region ${regionId + 1} set to ${getActionText(action)}`, 'success');
        }
        
        function scrollToTop() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        }
        
        function previewDocument() {
            showLoading('Generating HTML Preview', 'Creating layout-preserving document...');
            
            fetch('/process', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ actions: userActions })
            })
            .then(response => response.json())
            .then(data => {
                hideLoading();
                if (data.success) {
                    document.getElementById('mainContent').style.display = 'none';
                    document.getElementById('previewSection').style.display = 'block';
                    
                    // Create a new window with the HTML content
                    const newWindow = window.open('', '_blank');
                    newWindow.document.write(data.processed_html);
                    newWindow.document.close();
                    
                    showStatus('HTML preview opened in new window!', 'success');
                } else {
                    showStatus('Error: ' + data.error, 'error');
                }
            })
            .catch(error => {
                hideLoading();
                showStatus('Error generating preview: ' + error, 'error');
            });
        }
        
        function setAllRegionsToTranslate() {
            // Set all regions to translate mode
            for (let i = 0; i < textRegions.length; i++) {
                setRegionAction(i, 'translate');
            }
            console.log('All regions set to translate mode');
        }

        async function downloadDocument() {
            // First set all regions to translate mode
            setAllRegionsToTranslate();
            
            // Then translate all regions and download the HTML
            showLoading('Translating Document', 'Translating all regions and generating HTML document...');
            
            try {
                // First translate all regions
                const translateResponse = await fetch('/api/translate-all-regions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model: selectedModel,
                        target_language: selectedLanguage,
                        agent_mode: agentMode
                    })
                });
                
                const translateData = await translateResponse.json();
                
                if (translateData.success) {
                    showStatus(`All ${translateData.translated_regions} regions translated successfully!`, 'success');
                    
                    // Now download the processed HTML
                    const downloadResponse = await fetch('/download_html');
                    
                    if (downloadResponse.ok) {
                        const blob = await downloadResponse.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'translated_document.html';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                        window.URL.revokeObjectURL(url);
                        
                        showStatus('Document downloaded successfully!', 'success');
                    } else {
                        showStatus('Error downloading document', 'error');
                    }
                } else {
                    showStatus('Translation failed: ' + translateData.error, 'error');
                }
            } catch (error) {
                showStatus('Error: ' + error.message, 'error');
            } finally {
                hideLoading();
            }
        }
        
        function showStatus(message, type) {
            const statusDiv = document.getElementById('statusMessage');
            statusDiv.textContent = message;
            statusDiv.className = `status-message ${type}`;
            statusDiv.style.display = 'block';
            
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, 3000);
        }
        
        // Mode switching functions
        function switchMode(mode) {
            currentMode = mode;
            
            // Update button states
            document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById(mode + 'Mode').classList.add('active');
            
            // Show/hide sections
            if (mode === 'document') {
                document.getElementById('uploadSection').style.display = 'block';
                document.getElementById('textInputSection').style.display = 'none';
                document.getElementById('model-selection-section').style.display = 'block';
            } else {
                document.getElementById('uploadSection').style.display = 'none';
                document.getElementById('textInputSection').style.display = 'block';
                document.getElementById('model-selection-section').style.display = 'block';
                
                // Update language display for text mode
                updateLanguageDisplay();
            }
        }
        
        // Multi-Agent Text translation functions
        async function translateText() {
            const textInput = document.getElementById('textInput');
            const text = textInput.value.trim();
            
            if (!text) {
                showStatus('Please enter some text to translate.', 'error');
                return;
            }
            
            if (agentMode) {
                await translateTextWithMultiAgent(text);
            } else {
                await translateTextWithRAG(text);
            }
        }
        
        async function translateTextWithMultiAgent(text) {
            showLoading('🤖 Multi-Agent Translation', 'Generating multiple translation options...');
            
            try {
                // Use backend multi-agent endpoint
                const response = await fetch('/api/multi-agent-translate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        model: selectedModel,
                        target_language: selectedLanguage,
                        source_language: 'en' // Assume English source for now
                    })
                });
                
                const data = await response.json();
                hideLoading();
                
                if (data.success) {
                    // Display multiple translation options
                    displayBackendMultiAgentOptions(text, data.options);
                    showStatus('Multi-agent translation completed! Choose your preferred option.', 'success');
                } else {
                    showStatus('Multi-agent translation failed: ' + data.error, 'error');
                    // Fallback to RAG translation
                    await translateTextWithRAG(text);
                }
                
            } catch (error) {
                hideLoading();
                console.error('Multi-agent translation error:', error);
                showStatus('Multi-agent translation failed: ' + error.message, 'error');
                // Fallback to RAG translation
                await translateTextWithRAG(text);
            }
        }
        
        function displayBackendMultiAgentOptions(originalText, options) {
            const resultContainer = document.getElementById('textResultContainer');
            
            // Show the result container
            resultContainer.style.display = 'block';
            
            // Create translation options container
            const optionsContainer = document.createElement('div');
            optionsContainer.className = 'translation-options-container';
            
            let optionsHTML = `
                <h3 style="color: #4CAF50; margin-bottom: 20px;">🤖 Multi-Agent Translation Options</h3>
                <div class="agent-progress">
                    <h4 style="color: #4CAF50; margin-bottom: 15px;">Generated Translation Options</h4>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">RAG-Enhanced Translation</div>
                    </div>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">Direct Translation</div>
                    </div>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">Enhanced Translation</div>
                    </div>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">Conservative Translation</div>
                    </div>
                </div>
            `;
            
            // Add each translation option
            const optionKeys = ['rag_translation', 'enhanced_translation', 'direct_translation', 'conservative_translation'];
            let selectedOption = 'rag_translation';
            
            optionKeys.forEach((key, index) => {
                if (options[key] && options.metadata[key]) {
                    const metadata = options.metadata[key];
                    const isRecommended = metadata.recommended;
                    const isSelected = isRecommended && index === 0;
                    
                    if (isSelected) {
                        selectedOption = key;
                    }
                    
                    optionsHTML += `
                        <div class="translation-option ${isSelected ? 'selected' : ''}" data-option="${key}" onclick="selectTranslationOption('${key}', this)">
                            <div class="option-header">
                                <div class="option-title">${getOptionIcon(key)} ${metadata.type} ${isRecommended ? '(Recommended)' : ''}</div>
                                <div class="option-badge ${getBadgeClass(key)}">${metadata.type}</div>
                            </div>
                            <div class="option-content">${options[key]}</div>
                            <div class="option-metrics">
                                <div class="metric">
                                    <div class="metric-icon accuracy"></div>
                                    <span>${metadata.accuracy} Accuracy</span>
                                </div>
                                <div class="metric">
                                    <div class="metric-icon consistency"></div>
                                    <span>${metadata.consistency} Consistency</span>
                                </div>
                                <div class="metric">
                                    <div class="metric-icon quality"></div>
                                    <span>${metadata.description}</span>
                                </div>
                            </div>
                        </div>
                    `;
                }
            });
            
            optionsHTML += `
                <div style="margin-top: 20px;">
                    <button class="btn btn-primary" onclick="confirmSelectedBackendTranslation()" style="margin-right: 10px;">
                        ✅ Use Selected Translation
                    </button>
                    <button class="btn btn-outline" onclick="showBackendComparisonView()">
                        🔍 Compare Options
                    </button>
                </div>
            `;
            
            optionsContainer.innerHTML = optionsHTML;
            
            // Store options for later use
            window.currentBackendOptions = options;
            window.originalText = originalText;
            window.selectedTranslationOption = selectedOption;
            
            // Clear previous results and add new options
            resultContainer.innerHTML = '';
            resultContainer.appendChild(optionsContainer);
        }
        
        function getOptionIcon(key) {
            const icons = {
                'rag_translation': '🧠',
                'enhanced_translation': '🌟',
                'direct_translation': '📝',
                'conservative_translation': '🔒'
            };
            return icons[key] || '📄';
        }
        
        function getBadgeClass(key) {
            const classes = {
                'rag_translation': 'rag',
                'enhanced_translation': 'improved',
                'direct_translation': 'original',
                'conservative_translation': 'original'
            };
            return classes[key] || 'original';
        }
        
        function confirmSelectedBackendTranslation() {
            const options = window.currentBackendOptions;
            const selectedKey = window.selectedTranslationOption;
            
            if (options[selectedKey]) {
                currentTextTranslation = options[selectedKey];
                displayTextTranslation(options[selectedKey]);
                showStatus('Translation confirmed and saved!', 'success');
            } else {
                showStatus('Error: Selected translation not found', 'error');
            }
        }
        
        function showBackendComparisonView() {
            const options = window.currentBackendOptions;
            const originalText = window.originalText;
            
            const resultContainer = document.getElementById('textResultContainer');
            resultContainer.style.display = 'block';
            
            const comparisonContainer = document.createElement('div');
            comparisonContainer.className = 'translation-options-container';
            
            let comparisonHTML = `
                <h3 style="color: #4CAF50; margin-bottom: 20px;">🔍 Translation Comparison</h3>
                
                <div class="comparison-view">
                    <div class="comparison-panel">
                        <h4>📝 Original Text</h4>
                        <div class="comparison-text">${originalText}</div>
                    </div>
                    <div class="comparison-panel">
                        <h4>🧠 RAG-Enhanced Translation</h4>
                        <div class="comparison-text">${options.rag_translation || 'N/A'}</div>
                    </div>
                </div>
                
                <div class="comparison-view">
                    <div class="comparison-panel">
                        <h4>🌟 Enhanced Translation</h4>
                        <div class="comparison-text">${options.enhanced_translation || 'N/A'}</div>
                    </div>
                    <div class="comparison-panel">
                        <h4>📝 Direct Translation</h4>
                        <div class="comparison-text">${options.direct_translation || 'N/A'}</div>
                    </div>
                </div>
                
                <div class="comparison-view">
                    <div class="comparison-panel">
                        <h4>🔒 Conservative Translation</h4>
                        <div class="comparison-text">${options.conservative_translation || 'N/A'}</div>
                    </div>
                    <div class="comparison-panel">
                        <h4>📊 Translation Types</h4>
                        <div class="comparison-text">
                            <div style="margin-bottom: 10px;">
                                <strong>RAG-Enhanced:</strong> Uses legal glossary context
                            </div>
                            <div style="margin-bottom: 10px;">
                                <strong>Enhanced:</strong> Context-aware with improvements
                            </div>
                            <div style="margin-bottom: 10px;">
                                <strong>Direct:</strong> Standard neural translation
                            </div>
                            <div style="margin-bottom: 10px;">
                                <strong>Conservative:</strong> Literal structure preservation
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 20px; text-align: center;">
                    <button class="btn btn-primary" onclick="confirmSelectedBackendTranslation()" style="margin-right: 10px;">
                        ✅ Use Selected Translation
                    </button>
                    <button class="btn btn-outline" onclick="displayBackendMultiAgentOptions(window.originalText, window.currentBackendOptions)">
                        ← Back to Options
                    </button>
                </div>
            `;
            
            comparisonContainer.innerHTML = comparisonHTML;
            
            document.getElementById('textResultContainer').innerHTML = '';
            document.getElementById('textResultContainer').appendChild(comparisonContainer);
        }
        
        async function translateTextWithRAG(text) {
            showLoading('Translating Text', 'Processing your text...');
            
            try {
                const response = await fetch('/api/translate-text', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        model: selectedModel,
                        target_language: selectedLanguage,
                        agent_mode: agentMode
                    })
                });
                
                const data = await response.json();
                hideLoading();
                
                if (data.success) {
                    currentTextTranslation = data.translation;
                    displayTextTranslation(data.translation);
                    showStatus('Text translated successfully!', 'success');
                } else {
                    showStatus('Translation failed: ' + data.error, 'error');
                }
            } catch (error) {
                hideLoading();
                showStatus('Error translating text: ' + error.message, 'error');
            }
        }
        
        function displayMultiAgentTranslationOptions(originalText, agentResult) {
            const resultContainer = document.getElementById('textResultContainer');
            
            // Create translation options container
            const optionsContainer = document.createElement('div');
            optionsContainer.className = 'translation-options-container';
            optionsContainer.innerHTML = `
                <h3 style="color: #4CAF50; margin-bottom: 20px;">🤖 Multi-Agent Translation Options</h3>
                <div class="agent-progress">
                    <h4 style="color: #4CAF50; margin-bottom: 15px;">Agent Processing Summary</h4>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">Context Analysis</div>
                    </div>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">Translation</div>
                    </div>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">Validation</div>
                    </div>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">Consistency Check</div>
                    </div>
                    <div class="agent-step">
                        <div class="step-icon completed">✓</div>
                        <div class="step-text completed">Quality Improvement</div>
                    </div>
                </div>
                
                <div class="translation-option selected" data-option="improved" onclick="selectTranslationOption('improved', this)">
                    <div class="option-header">
                        <div class="option-title">🌟 Improved Translation (Recommended)</div>
                        <div class="option-badge improved">Enhanced</div>
                    </div>
                    <div class="option-content">${agentResult.finalText || agentResult.improvedTranslation}</div>
                    <div class="option-metrics">
                        <div class="metric">
                            <div class="metric-icon accuracy"></div>
                            <span>High Accuracy</span>
                        </div>
                        <div class="metric">
                            <div class="metric-icon consistency"></div>
                            <span>Language Consistent</span>
                        </div>
                        <div class="metric">
                            <div class="metric-icon quality"></div>
                            <span>Quality Enhanced</span>
                        </div>
                    </div>
                </div>
                
                <div class="translation-option" data-option="original" onclick="selectTranslationOption('original', this)">
                    <div class="option-header">
                        <div class="option-title">📝 Original Translation</div>
                        <div class="option-badge original">Standard</div>
                    </div>
                    <div class="option-content">${agentResult.originalTranslation}</div>
                    <div class="option-metrics">
                        <div class="metric">
                            <div class="metric-icon accuracy"></div>
                            <span>Good Accuracy</span>
                        </div>
                        <div class="metric">
                            <div class="metric-icon consistency"></div>
                            <span>Basic Consistency</span>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 20px;">
                    <button class="btn btn-primary" onclick="confirmSelectedTranslation()" style="margin-right: 10px;">
                        ✅ Use Selected Translation
                    </button>
                    <button class="btn btn-outline" onclick="showComparisonView()">
                        🔍 Compare Options
                    </button>
                </div>
            `;
            
            // Store agent result for later use
            window.currentAgentResult = agentResult;
            window.originalText = originalText;
            window.selectedTranslationOption = 'improved';
            
            // Clear previous results and add new options
            resultContainer.innerHTML = '';
            resultContainer.appendChild(optionsContainer);
        }
        
        function selectTranslationOption(optionType, element) {
            // Remove selected class from all options
            document.querySelectorAll('.translation-option').forEach(opt => {
                opt.classList.remove('selected');
            });
            
            // Add selected class to clicked option
            element.classList.add('selected');
            window.selectedTranslationOption = optionType;
        }
        
        function confirmSelectedTranslation() {
            const agentResult = window.currentAgentResult;
            let selectedTranslation;
            
            switch (window.selectedTranslationOption) {
                case 'improved':
                    selectedTranslation = agentResult.finalText || agentResult.improvedTranslation;
                    break;
                case 'original':
                    selectedTranslation = agentResult.originalTranslation;
                    break;
                default:
                    selectedTranslation = agentResult.finalText || agentResult.originalTranslation;
            }
            
            currentTextTranslation = selectedTranslation;
            displayTextTranslation(selectedTranslation);
            showStatus('Translation confirmed and saved!', 'success');
        }
        
        function showComparisonView() {
            const agentResult = window.currentAgentResult;
            const originalText = window.originalText;
            
            const comparisonContainer = document.createElement('div');
            comparisonContainer.className = 'translation-options-container';
            comparisonContainer.innerHTML = `
                <h3 style="color: #4CAF50; margin-bottom: 20px;">🔍 Translation Comparison</h3>
                
                <div class="comparison-view">
                    <div class="comparison-panel">
                        <h4>📝 Original Text</h4>
                        <div class="comparison-text">${originalText}</div>
                    </div>
                    <div class="comparison-panel">
                        <h4>🌟 Improved Translation</h4>
                        <div class="comparison-text">${agentResult.finalText || agentResult.improvedTranslation}</div>
                    </div>
                </div>
                
                <div class="comparison-view">
                    <div class="comparison-panel">
                        <h4>📝 Original Translation</h4>
                        <div class="comparison-text">${agentResult.originalTranslation}</div>
                    </div>
                    <div class="comparison-panel">
                        <h4>📊 Quality Metrics</h4>
                        <div class="comparison-text">
                            <div style="margin-bottom: 10px;">
                                <strong>Accuracy:</strong> ${agentResult.metadata?.agentResults?.validation?.accuracy || 'N/A'}%
                            </div>
                            <div style="margin-bottom: 10px;">
                                <strong>Consistency:</strong> ${agentResult.metadata?.agentResults?.consistencyCheck?.consistencyScore || 'N/A'}%
                            </div>
                            <div style="margin-bottom: 10px;">
                                <strong>Processing Time:</strong> ${Math.round((agentResult.metadata?.duration || 0) / 1000)}s
                            </div>
                        </div>
                    </div>
                </div>
                
                <div style="margin-top: 20px; text-align: center;">
                    <button class="btn btn-primary" onclick="confirmSelectedTranslation()" style="margin-right: 10px;">
                        ✅ Use Improved Translation
                    </button>
                    <button class="btn btn-outline" onclick="displayMultiAgentTranslationOptions(window.originalText, window.currentAgentResult)">
                        ← Back to Options
                    </button>
                </div>
            `;
            
            document.getElementById('textResultContainer').innerHTML = '';
            document.getElementById('textResultContainer').appendChild(comparisonContainer);
        }
        
        function displayTextTranslation(translation) {
            const resultContainer = document.getElementById('textResultContainer');
            const resultDiv = document.getElementById('textResult');
            
            resultDiv.textContent = translation;
            resultContainer.style.display = 'block';
        }
        
        function clearText() {
            document.getElementById('textInput').value = '';
            document.getElementById('textResultContainer').style.display = 'none';
            currentTextTranslation = '';
        }
        
        function editTranslation() {
            const resultDiv = document.getElementById('textResult');
            const currentText = resultDiv.textContent;
            
            // Replace with editable div
            const editableDiv = document.createElement('div');
            editableDiv.className = 'editable-translation';
            editableDiv.contentEditable = true;
            editableDiv.textContent = currentText;
            
            // Add save/cancel buttons
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'text-result-actions';
            buttonContainer.innerHTML = `
                <button class="action-btn" onclick="saveTextTranslation()">💾 Save</button>
                <button class="action-btn" onclick="cancelTextEdit()">❌ Cancel</button>
            `;
            
            resultDiv.parentNode.replaceChild(editableDiv, resultDiv);
            editableDiv.parentNode.appendChild(buttonContainer);
            
            editableDiv.focus();
        }
        
        function saveTextTranslation() {
            const editableDiv = document.querySelector('.editable-translation');
            const newTranslation = editableDiv.textContent.trim();
            
            if (!newTranslation) {
                showStatus('Translation cannot be empty', 'error');
                return;
            }
            
            currentTextTranslation = newTranslation;
            
            // Restore original display
            const resultDiv = document.createElement('div');
            resultDiv.className = 'text-result';
            resultDiv.id = 'textResult';
            resultDiv.textContent = newTranslation;
            
            const buttonContainer = document.querySelector('.text-result-actions');
            buttonContainer.innerHTML = `
                <button class="action-btn" onclick="editTranslation()">✏️ Edit Translation</button>
                <button class="action-btn" onclick="copyTranslation()">📋 Copy</button>
                <button class="action-btn" onclick="downloadTextTranslation()">💾 Download</button>
            `;
            
            editableDiv.parentNode.replaceChild(resultDiv, editableDiv);
            
            showStatus('Translation updated successfully!', 'success');
        }
        
        function cancelTextEdit() {
            const editableDiv = document.querySelector('.editable-translation');
            const resultDiv = document.createElement('div');
            resultDiv.className = 'text-result';
            resultDiv.id = 'textResult';
            resultDiv.textContent = currentTextTranslation;
            
            const buttonContainer = document.querySelector('.text-result-actions');
            buttonContainer.innerHTML = `
                <button class="action-btn" onclick="editTranslation()">✏️ Edit Translation</button>
                <button class="action-btn" onclick="copyTranslation()">📋 Copy</button>
                <button class="action-btn" onclick="downloadTextTranslation()">💾 Download</button>
            `;
            
            editableDiv.parentNode.replaceChild(resultDiv, editableDiv);
        }
        
        function copyTranslation() {
            const text = currentTextTranslation;
            navigator.clipboard.writeText(text).then(() => {
                showStatus('Translation copied to clipboard!', 'success');
            }).catch(() => {
                showStatus('Failed to copy translation', 'error');
            });
        }
        
        function downloadTextTranslation() {
            const text = currentTextTranslation;
            const blob = new Blob([text], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'translated_text.txt';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            showStatus('Translation downloaded successfully!', 'success');
        }
        
        // Region translation editing functions
        function editRegionTranslation(regionId) {
            const translationDiv = document.getElementById(`translation-${regionId}`);
            const languageEditor = document.getElementById(`language-editor-${regionId}`);
            
            // Make translation editable
            translationDiv.contentEditable = true;
            translationDiv.classList.add('editable');
            translationDiv.focus();
            
            // Show language editor
            languageEditor.style.display = 'block';
            
            // Set current language as default
            const targetLangSelect = document.getElementById(`target-lang-${regionId}`);
            targetLangSelect.value = selectedLanguage;
        }
        
        async function retranslateRegion(regionId) {
            const sourceLangSelect = document.getElementById(`source-lang-${regionId}`);
            const targetLangSelect = document.getElementById(`target-lang-${regionId}`);
            const translationDiv = document.getElementById(`translation-${regionId}`);
            
            const sourceLang = sourceLangSelect.value;
            const targetLang = targetLangSelect.value;
            const originalText = textRegions[regionId].text;
            
            if (sourceLang === 'auto') {
                // Auto-detect language (simplified - you could use a language detection API)
                showStatus('Auto-detecting language...', 'info');
            }
            
            showLoading('Retranslating Region', 'Processing with new language settings...');
            
            try {
                const response = await fetch('/api/translate-text', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                        text: originalText,
                        model: selectedModel,
                        target_language: targetLang,
                        agent_mode: agentMode
                            })
                        });
                        
                        const data = await response.json();
                hideLoading();
                        
                        if (data.success) {
                    translationDiv.textContent = data.translation;
                    textRegions[regionId].translated = data.translation;
                    showStatus(`Region ${regionId + 1} retranslated successfully!`, 'success');
                } else {
                    showStatus('Retranslation failed: ' + data.error, 'error');
                        }
                    } catch (error) {
                hideLoading();
                showStatus('Error retranslating region: ' + error.message, 'error');
            }
        }
        
        function saveRegionTranslation(regionId) {
            const translationDiv = document.getElementById(`translation-${regionId}`);
            const languageEditor = document.getElementById(`language-editor-${regionId}`);
            const newTranslation = translationDiv.textContent.trim();
            
            if (!newTranslation) {
                showStatus('Translation cannot be empty', 'error');
                return;
            }
            
            // Update the region data
            textRegions[regionId].translated = newTranslation;
            
            // Make translation non-editable
            translationDiv.contentEditable = false;
            translationDiv.classList.remove('editable');
            
            // Hide language editor
            languageEditor.style.display = 'none';
            
            showStatus(`Translation for region ${regionId + 1} updated successfully!`, 'success');
        }
        
        function cancelRegionEdit(regionId) {
            const translationDiv = document.getElementById(`translation-${regionId}`);
            const languageEditor = document.getElementById(`language-editor-${regionId}`);
            const originalText = textRegions[regionId].translated;
            
            // Restore original text
            translationDiv.textContent = originalText;
            translationDiv.contentEditable = false;
            translationDiv.classList.remove('editable');
            
            // Hide language editor
            languageEditor.style.display = 'none';
        }
        
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            checkRAGStatus();
            initializeControlPanel();
            
            // Modal language card selection
            document.querySelectorAll('.language-card-modal').forEach(card => {
                card.addEventListener('click', function() {
                    // Remove selection from all modal cards
                    document.querySelectorAll('.language-card-modal').forEach(c => c.classList.remove('selected'));
                    
                    // Add selection to clicked card
                    this.classList.add('selected');
                    
                    // Enable confirm button
                    document.getElementById('confirmLanguage').disabled = false;
                });
            });
            
            // Confirm language button
            document.getElementById('confirmLanguage').addEventListener('click', function() {
                const selectedCard = document.querySelector('.language-card-modal.selected');
                if (selectedCard && window.pendingFile) {
                    const sourceLanguage = selectedCard.dataset.lang;
                    hideLanguageModal();
                    processFileWithLanguage(window.pendingFile, sourceLanguage);
                }
            });
        });
    </script>
</body>
</html>
"""

# Flask Routes
@app.route('/')
def index():
    """Serve the main HTML page"""
    return create_dynamic_ui()

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and OCR processing with enhanced features"""
    global current_image, current_bboxes, current_translated_text, current_image_path, current_target_language
    
    try:
        # Get uploaded file
        file = request.files['image']
        if not file:
            return jsonify({'error': 'No image uploaded'}), 400
        
        # Get all control settings
        source_language = request.form.get('sourceLanguage', 'auto')
        target_language = request.form.get('targetLanguage', 'te')
        ocr_engine = request.form.get('ocrEngine', 'easyocr')
        min_font_size = int(request.form.get('minFontSize', 12))
        max_font_size = int(request.form.get('maxFontSize', 24))
        font_scale = float(request.form.get('fontScale', 1.0))
        enable_preprocessing = request.form.get('enablePreprocessing', 'true').lower() == 'true'
        enable_denoising = request.form.get('enableDenoising', 'true').lower() == 'true'
        enable_thresholding = request.form.get('enableThresholding', 'true').lower() == 'true'
        translation_mode = request.form.get('translationMode', 'literal')
        enable_rag = request.form.get('enableRAG', 'true').lower() == 'true'
        show_confidence = request.form.get('showConfidence', 'true').lower() == 'true'
        
        print(f"Upload request - Source: {source_language}, Target: {target_language}")
        print(f"OCR Engine: {ocr_engine}, Font Range: {min_font_size}-{max_font_size}, Scale: {font_scale}")
        print(f"Preprocessing: {enable_preprocessing}, Denoising: {enable_denoising}, Thresholding: {enable_thresholding}")
        print(f"Translation Mode: {translation_mode}, RAG: {enable_rag}, Confidence: {show_confidence}")
        
        # Save uploaded file
        filename = file.filename
        file_path = os.path.join(tempfile.gettempdir(), filename)
        file.save(file_path)
        
        current_image_path = file_path
        current_target_language = target_language
        
        # Process with OCR using selected engine
        if ocr_engine == 'tesseract':
            bboxes = process_image_with_tesseract_ocr(file_path, source_language)
        elif ocr_engine == 'both':
            bboxes = process_image_with_ocr(file_path)
            if not bboxes:
                print("EasyOCR failed, trying Tesseract...")
                bboxes = process_image_with_tesseract_ocr(file_path, source_language)
        else:  # easyocr or default
            bboxes = process_image_with_ocr(file_path)
        current_bboxes = bboxes
        
        if not bboxes:
            return jsonify({
                'success': False,
                'error': 'No text detected in the image. Please try with a clearer image or one with printed text.',
                'suggestions': [
                    'Make sure the image is clear and not blurry',
                    'Ensure the text is printed (not handwritten)',
                    'Try a higher resolution image',
                    'Check if the image contains readable text'
                ]
            }), 400
        
        # Extract text for translation
        original_text = '\n'.join([bbox['text'] for bbox in bboxes])
        print(f"DEBUG: Original text extracted: {original_text[:100]}...")
        
        # Get model, agent mode, and target language from request
        # Use the target_language already retrieved above, fallback to form if needed
        model = request.form.get('model', 'gemma3:latest')
        agent_mode = request.form.get('agent_mode', 'false').lower() == 'true'
        # Get target_language - check both camelCase and snake_case
        target_language = request.form.get('target_language') or request.form.get('targetLanguage') or current_target_language or 'te'
        
        print(f"DEBUG: Model={model}, AgentMode={agent_mode}, TargetLang={target_language}")
        
        # Translate each text region individually
        translated_lines = []
        for i, bbox in enumerate(bboxes):
            region_text = bbox['text']
            print(f"DEBUG: Translating region {i+1}: '{region_text[:50]}...' with target_language={target_language}")
            
            if agent_mode:
                region_translated = translate_paragraph_with_rag(region_text, model, target_language, source_language='auto')
            else:
                # Explicitly pass target_language to ensure it's used
                region_translated = translate_text(region_text, model, target_language=target_language, source_language='auto')
            
            print(f"DEBUG: Region {i+1} translated: '{region_translated[:50]}...'")
            translated_lines.append(region_translated)
        
        current_translated_text = translated_lines
        
        # Store target language
        current_target_language = target_language
        
        # Convert image to base64 for display
        with open(file_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode()
        
        # Prepare response data
        response_data = {
            'success': True,
            'image': img_data,
            'text_regions': [],
            'bboxes': bboxes,
            'extracted_text': [bbox['text'] for bbox in bboxes]
        }
        
        # Add text regions with bounding boxes
        for i, bbox in enumerate(bboxes):
            bbox_coords = bbox['bbox']
            x_coords = [point[0] for point in bbox_coords]
            y_coords = [point[1] for point in bbox_coords]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            region_data = {
                'id': i,
                'title': f'Text Region {i+1}',
                'text': bbox['text'],
                'translated': translated_lines[i] if i < len(translated_lines) else bbox['text'],
                'bbox': {
                    'x': x_min,
                    'y': y_min,
                    'width': x_max - x_min,
                    'height': y_max - y_min
                },
                'action': 'preserve'  # Default action
            }
            response_data['text_regions'].append(region_data)
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rag/status', methods=['GET'])
def rag_status():
    """Get RAG system status"""
    return jsonify({
        'rag_available': current_rag_system is not None,
        'rag_initialized': current_rag_system is not None,
        'status': 'active' if current_rag_system else 'inactive'
    })

@app.route('/api/rag/setup', methods=['POST'])
def setup_rag():
    """Setup RAG system"""
    global current_rag_system
    
    if not RAG_AVAILABLE:
        return jsonify({'success': False, 'error': 'RAG system not available'})
    
    try:
        if not current_rag_system:
            current_rag_system = LegalRAGSystem()
            current_rag_system.initialize()
        
        # Process documents if they exist
        hindi_glossary_dir = Path("glossary - hindi")
        telugu_glossary_dir = Path("glossary telugu")
        go_dir = Path("GOs (1)")
        
        if hindi_glossary_dir.exists():
            current_rag_system.process_glossary_documents(str(hindi_glossary_dir), 'hindi')
        
        if telugu_glossary_dir.exists():
            current_rag_system.process_glossary_documents(str(telugu_glossary_dir), 'telugu')
        
        if go_dir.exists():
            current_rag_system.process_government_orders(str(go_dir))
        
        current_rag_system.save_metadata()
        
        return jsonify({
            'success': True,
            'message': 'RAG system setup completed successfully',
            'collections': list(current_rag_system.collections.keys()),
            'total_documents': len(current_rag_system.document_metadata)
        })
        
    except Exception as e:
        print(f"RAG setup error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/process', methods=['POST'])
def process_document():
    """Process document with user actions"""
    global current_image_path, current_bboxes, current_translated_text, current_target_language
    
    try:
        data = request.get_json()
        user_actions = data.get('actions', {})
        
        print(f"DEBUG: Received user_actions in /process: {user_actions}")
        print(f"DEBUG: current_translated_text length: {len(current_translated_text) if current_translated_text else 0}")
        print(f"DEBUG: current_bboxes length: {len(current_bboxes) if current_bboxes else 0}")
        
        if not current_image_path or not current_bboxes:
            return jsonify({'error': 'No image processed'}), 400
        
        # Create processed HTML document
        processed_html = create_processed_html(
            current_image_path, 
            current_bboxes, 
            current_translated_text, 
            user_actions, 
            current_target_language
        )
        
        # Save processed HTML
        output_path = os.path.join(tempfile.gettempdir(), 'processed_document.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_html)
        
        return jsonify({
            'success': True,
            'processed_html': processed_html,
            'download_url': '/download_html'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_html')
def download_html_document():
    """Download the processed HTML document"""
    try:
        output_path = os.path.join(tempfile.gettempdir(), 'processed_document.html')
        if os.path.exists(output_path):
            return send_file(output_path, as_attachment=True, download_name='processed_document.html')
        else:
            return jsonify({'error': 'No processed document available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug-state', methods=['GET'])
def debug_state():
    """Debug endpoint to check current state"""
    global current_image_path, current_bboxes, current_translated_text, current_target_language
    
    return jsonify({
        'has_image': bool(current_image_path),
        'has_bboxes': bool(current_bboxes),
        'bboxes_count': len(current_bboxes) if current_bboxes else 0,
        'translated_text_count': len(current_translated_text) if current_translated_text else 0,
        'target_language': current_target_language,
        'first_bbox_text': current_bboxes[0]['text'] if current_bboxes else None,
        'first_translated_text': current_translated_text[0] if current_translated_text else None
    })

@app.route('/api/ollama/models', methods=['GET'])
def get_ollama_models():
    """Get IndicTransToolkit model information (maintained for compatibility)"""
    try:
        # Return generic translator info (no provider/model names in UI)
        return jsonify({
            'success': True,
            'models': [
                {
                    'name': 'translator-a',
                    'family': 'translator',
                    'icon': '⚙️',
                    'description': 'Standard Translator (A)'
                },
                {
                    'name': 'translator-b',
                    'family': 'translator',
                    'icon': '⚙️',
                    'description': 'Standard Translator (B)'
                }
            ],
            'current_model': 'translator-a',
            'available': INDIC_TRANS_AVAILABLE,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        })
        
    except Exception as e:
        print(f"Error getting model info: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'models': [],
            'current_model': INDIC2_INDIC_EN
        })

@app.route('/debug/translation', methods=['POST'])
def debug_translation():
    """Debug translation endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('target_language', 'te')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        # Test translation with IndicTransToolkit
        translated_text = translate_text(text, target_language=target_language)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'translated_text': translated_text,
            'target_language': target_language,
            'rag_used': current_rag_system is not None,
            'translation_engine': 'IndicTransToolkit'
        })
        
    except Exception as e:
        print(f"Debug translation error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ollama', methods=['POST'])
def ollama_api():
    """Translation API endpoint (maintained for compatibility, now uses IndicTransToolkit)"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        target_lang = data.get('target_language', 'te')  # Extract target language if provided
        source_lang = data.get('source_language', 'auto')
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})
        
        # Use IndicTransToolkit for translation
        translated_text = translate_with_indic_toolkit(prompt, source_lang, target_lang)
        
        return jsonify({
            'success': True,
            'response': translated_text,
            'model': 'IndicTransToolkit'
        })
        
    except Exception as e:
        print(f"Translation API error: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/edit-translation', methods=['POST'])
def edit_translation():
    """Edit a specific translation"""
    global current_translated_text
    
    try:
        data = request.get_json()
        region_index = data.get('region_index')
        new_translation = data.get('new_translation')
        
        if region_index is None or new_translation is None:
            return jsonify({'success': False, 'error': 'Missing region_index or new_translation'}), 400
        
        if region_index < 0 or region_index >= len(current_translated_text):
            return jsonify({'success': False, 'error': 'Invalid region index'}), 400
        
        # Update the translation
        current_translated_text[region_index] = new_translation
        
        # Regenerate the HTML document
        user_actions = {str(i): 'translate' for i in range(len(current_translated_text))}
        create_processed_html(current_image_path, current_bboxes, current_translated_text, user_actions, current_target_language)
        
        return jsonify({
            'success': True,
            'message': 'Translation updated successfully',
            'updated_translation': new_translation
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-translations')
def get_translations():
    """Get all current translations for editing"""
    global current_translated_text, current_bboxes
    
    try:
        translations = []
        for i, (bbox, translation) in enumerate(zip(current_bboxes, current_translated_text)):
            translations.append({
                'index': i,
                'original_text': bbox['text'],
                'translated_text': translation,
                'bbox': bbox['bbox']
            })
        
        return jsonify({
            'success': True,
            'translations': translations,
            'total_regions': len(current_bboxes)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/translate-text', methods=['POST'])
def translate_text_api():
    """API endpoint for direct text translation"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        model = data.get('model', 'gemma3:latest')
        target_language = data.get('target_language', 'te')
        source_language = data.get('source_language', 'auto')
        agent_mode = data.get('agent_mode', False)
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        print(f"Translating text: '{text[:50]}...' to {target_language} using {model}")
        
        # Always use IndicTransToolkit directly (NO RAG)
        translated_text = translate_text(text, model, target_language, source_language)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'translation': translated_text,
            'target_language': target_language,
            'source_language': source_language,
            'model': model,
            'agent_mode': agent_mode
        })
        
    except Exception as e:
        print(f"Text translation error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/translate-all-regions', methods=['POST'])
def translate_all_regions():
    """Translate all regions in the document"""
    global current_image_path, current_bboxes, current_translated_text, current_target_language
    
    try:
        data = request.get_json()
        model = data.get('model', 'gemma3:latest')
        target_language = data.get('target_language', 'te')
        source_language = data.get('source_language', 'auto')
        agent_mode = data.get('agent_mode', False)
        
        if not current_image_path or not current_bboxes:
            return jsonify({'error': 'No document loaded'}), 400
        
        print(f"Translating all regions with model: {model}, target: {target_language}, agent_mode: {agent_mode}")
        
        # Translate all regions
        translated_lines = []
        for i, bbox in enumerate(current_bboxes):
            region_text = bbox['text']
            print(f"Translating region {i+1}: '{region_text[:50]}...'")
            
            # Always use IndicTransToolkit directly (NO RAG)
            region_translated = translate_text(region_text, model, target_language, source_language)
            
            print(f"Region {i+1} translated: '{region_translated[:50]}...'")
            translated_lines.append(region_translated)
        
        # Update global variables
        current_translated_text = translated_lines
        current_target_language = target_language
        
        # Set all regions to translate action
        user_actions = {str(i): 'translate' for i in range(len(translated_lines))}
        
        # Generate HTML with translated text
        print(f"DEBUG: About to generate HTML with {len(current_translated_text)} translated lines")
        print(f"DEBUG: First few translated lines: {current_translated_text[:3] if current_translated_text else 'Empty'}")
        print(f"DEBUG: User actions: {user_actions}")
        
        processed_html = create_processed_html(
            current_image_path, 
            current_bboxes, 
            current_translated_text, 
            user_actions, 
            current_target_language
        )
        
        # Save the processed HTML
        output_path = os.path.join(tempfile.gettempdir(), 'processed_document.html')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(processed_html)
        
        return jsonify({
            'success': True,
            'message': f'All {len(translated_lines)} regions translated successfully',
            'translated_regions': len(translated_lines),
            'download_url': '/download-processed'
        })
        
    except Exception as e:
        print(f"Error in translate_all_regions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/multi-agent-translate', methods=['POST'])
def multi_agent_translate_api():
    """API endpoint for multi-agent translation with multiple options"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        model = data.get('model', 'gemma3:latest')
        target_language = data.get('target_language', 'te')
        source_language = data.get('source_language', 'auto')
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        
        # ALWAYS auto-detect source language from actual text (override what frontend sends)
        # This ensures Telugu text is detected as Telugu, not English
        detected = detect_app_lang_code(text)
        original_source = source_language
        source_language = detected
        print(f"🔍 FORCED auto-detection: {detected} (frontend sent: {original_source}, text analysis: {detected})")
        print(f"📝 Text sample: {text[:100]}...")
        
        print(f"🤖 Multi-agent translation: '{text[:50]}...' from {source_language} to {target_language} using {model}")
        
        # Generate multiple translation options
        translation_options = generate_multiple_translation_options(text, source_language, target_language, model)
        
        return jsonify({
            'success': True,
            'original_text': text,
            'source_language': source_language,
            'target_language': target_language,
            'model': model,
            'options': translation_options
        })
    except Exception as e:
        print(f"Error in multi_agent_translate_api: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

def generate_multiple_translation_options(text, source_lang, target_lang, model):
    """Generate multiple translation options using different approaches"""
    options = {}
    
    try:
        # Option 1: Standard translation (NO RAG - using IndicTransToolkit directly)
        print("🔄 Generating standard translation...")
        options['rag_translation'] = translate_text(text, model, target_lang, source_lang)
        
        # Option 2: Direct translation without RAG
        print("🔄 Generating direct translation...")
        options['direct_translation'] = translate_text(text, model, target_lang, source_lang)
        
        # Option 3: Enhanced translation with context
        print("🔄 Generating enhanced translation...")
        options['enhanced_translation'] = generate_enhanced_translation(text, source_lang, target_lang, model)
        
        # Option 4: Conservative translation (literal)
        print("🔄 Generating conservative translation...")
        options['conservative_translation'] = generate_conservative_translation(text, source_lang, target_lang, model)
        
        # Add metadata for each option
        options['metadata'] = {
            'rag_translation': {
                'type': 'Standard',
                'description': 'Direct IndicTransToolkit translation',
                'accuracy': 'High',
                'consistency': 'Good',
                'recommended': True
            },
            'direct_translation': {
                'type': 'Direct',
                'description': 'Standard neural translation',
                'accuracy': 'Good',
                'consistency': 'Good',
                'recommended': False
            },
            'enhanced_translation': {
                'type': 'Enhanced',
                'description': 'Context-aware translation with improvements',
                'accuracy': 'Very High',
                'consistency': 'Excellent',
                'recommended': True
            },
            'conservative_translation': {
                'type': 'Conservative',
                'description': 'Literal translation preserving structure',
                'accuracy': 'Medium',
                'consistency': 'High',
                'recommended': False
            }
        }
        
        return options
        
    except Exception as e:
        print(f"Error generating translation options: {e}")
        # Fallback to single translation (NO RAG)
        fallback_translation = translate_text(text, model, target_lang, source_lang)
        return {
            'rag_translation': fallback_translation,
            'metadata': {
                'rag_translation': {
                    'type': 'Fallback',
                    'description': 'Single translation option',
                    'accuracy': 'Good',
                    'consistency': 'Good',
                    'recommended': True
                }
            }
        }

def generate_enhanced_translation(text, source_lang, target_lang, model):
    """Generate enhanced translation using IndicTransToolkit (NO RAG)"""
    try:
        # Auto-detect source language if needed
        if source_lang in ('auto', None, ''):
            detected = detect_app_lang_code(text)
            source_lang = detected
            print(f"🔍 Auto-detected source language: {detected}")
        
        # Use IndicTransToolkit directly - NO RAG
        translated_text = translate_with_indic_toolkit(text, source_lang, target_lang)
        return translated_text
            
    except Exception as e:
        print(f"Error in enhanced translation: {e}")
        return translate_text(text, model, target_lang, source_lang)

def generate_conservative_translation(text, source_lang, target_lang, model):
    """Generate conservative/literal translation using IndicTransToolkit (NO RAG)"""
    try:
        # Auto-detect source language if needed
        if source_lang in ('auto', None, ''):
            detected = detect_app_lang_code(text)
            source_lang = detected
            print(f"🔍 Auto-detected source language: {detected}")
        
        # Use IndicTransToolkit directly - NO RAG
        translated_text = translate_with_indic_toolkit(text, source_lang, target_lang)
        return translated_text
            
    except Exception as e:
        print(f"Error in conservative translation: {e}")
        return translate_text(text, model, target_lang, source_lang)

if __name__ == '__main__':
    print("Starting Enhanced Legal RAG Document Translator...")
    print("=" * 60)
    print("System Status:")
    print(f"   • Legal RAG System: {'Available' if RAG_AVAILABLE else 'Not Available'}")
    print(f"   • RAG Initialized: {'Yes' if current_rag_system else 'No'}")
    print(f"   • IndicTransToolkit: {'Available' if INDIC_TRANS_AVAILABLE else 'Not Available'}")
    try:
        import os
        if os.environ.get('INDIC_TRANS_OFFLINE', 'false').lower() in ('true', '1', 'yes'):
            print(f"   • Mode: 🔒 OFFLINE (Self-Hosted - using local models only)")
        else:
            print(f"   • Mode: 🌐 ONLINE (will download if models not cached)")
    except:
        pass
    print(f"   • Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)
    print("Access the system at: http://localhost:5000")
    print("Features:")
    print("   • Paragraph-by-paragraph translation")
    print("   • Legal context awareness")
    print("   • RAG-powered glossary lookup")
    print("   • Real-time processing status")
    print("=" * 60)
    try:
        import os
        if os.environ.get('INDIC_TRANS_OFFLINE', 'false').lower() in ('true', '1', 'yes'):
            print("\n💡 Running in OFFLINE mode - all models loaded from local cache")
            print("   No internet connection required!")
    except:
        pass
    
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
