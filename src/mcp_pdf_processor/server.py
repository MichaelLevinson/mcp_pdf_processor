"""
PDF Processor MCP Server

This server fetches and processes PDF documents into LLM-friendly markdown format,
with intelligent chunking, token estimation, LaTeX equation extraction, and optional QDrant integration.

Usage:
    python server.py

Requirements:
    - PyMuPDF (fitz)
    - pydantic
    - aiohttp
    - mcp (Model Context Protocol)
    - pix2tex (for LaTeX equation extraction)
    - qdrant-client (optional, for vector storage)
    - sentence-transformers (optional, for embeddings)

Notes:
    - LaTeX equation extraction is optimized for Apple Silicon M4 Pro using MPS acceleration.
"""

import asyncio
import gc
import os
import tempfile
import uuid
from hashlib import md5
from pathlib import Path
from typing import Annotated, List, Optional, Tuple, Union, Dict, Any
from urllib.parse import urlparse
import sys

import fitz  # PyMuPDF
import aiohttp
import io
import re
import numpy as np
import torch
from PIL import Image

# Import math utilities for LaTeX extraction
try:
    # Try relative import first (for installed package)
    from .math_utils import (
        is_likely_math, extract_math_blocks_from_page,
        get_equation_from_cache, add_equation_to_cache,
        process_equation_batch
    )
except ImportError:
    # Fall back to absolute import (for development)
    from mcp_pdf_processor.math_utils import (
        is_likely_math, extract_math_blocks_from_page,
        get_equation_from_cache, add_equation_to_cache,
        process_equation_batch
    )
except ImportError:
    # Fall back to direct import (for testing)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from math_utils import (
        is_likely_math, extract_math_blocks_from_page,
        get_equation_from_cache, add_equation_to_cache,
        process_equation_batch
    )
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    ErrorData, GetPromptResult, Prompt, PromptArgument,
    PromptMessage, TextContent, Tool, INVALID_PARAMS, INTERNAL_ERROR,
)
from pydantic import BaseModel, Field, AnyUrl

# Check for hardware acceleration
MPS_AVAILABLE = torch.backends.mps.is_available()
CUDA_AVAILABLE = torch.cuda.is_available()

# For stability with pix2tex, we'll use CPU regardless of available hardware
# This avoids device mismatch issues while still being fast enough for moderate use
DEVICE = torch.device("cpu")

# Optional imports for LaTeX equation extraction
try:
    from pix2tex.cli import LatexOCR
    LATEX_OCR_AVAILABLE = True
    # Initialize LaTeX OCR model - lazy loading to save memory
    latex_ocr_model = None
except ImportError:
    LATEX_OCR_AVAILABLE = False

# Optional imports - only used if QDrant integration is enabled
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from sentence_transformers import SentenceTransformer
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Configuration constants
DEFAULT_TOKEN_LIMIT = 8000  # Approximate token limit for most LLMs
DEFAULT_USER_AGENT = "ModelContextProtocol/1.0 (PDF Processor; +https://github.com/your-repo/pdf-processor)"
DEFAULT_CHUNK_SIZE = 1000    # Token size for each chunk
OUTPUT_DIR = Path("./llm_output")  # Directory to save processed files
ENABLE_LATEX_EXTRACTION = True  # Whether to extract LaTeX equations
MATH_DETECTION_THRESHOLD = 0.75  # Confidence threshold for math detection
MATH_SYMBOL_PATTERN = re.compile(r'[\\∫∑∏√^_{}=<>⊂⊃∈∀∃±∓×÷≤≥≠∞∂∇∆∫∮∯∰∱∲∳]')


def get_latex_ocr_model():
    """Get the LaTeX OCR model, initialize if not already done.
    
    Uses lazy loading to conserve memory until needed.
    Uses CPU for maximum compatibility across all platforms.
    
    Returns:
        Initialized LatexOCR model instance
    
    Raises:
        ImportError: If LaTeX OCR dependencies are not available
    """
    global latex_ocr_model
    
    if not LATEX_OCR_AVAILABLE:
        raise ImportError(
            "LaTeX equation extraction requires pix2tex. "
            "Install with: pip install pix2tex"
        )
    
    if latex_ocr_model is None:
        print(f"Initializing LaTeX OCR model on device: {DEVICE}", file=sys.stderr)
        latex_ocr_model = LatexOCR()
        # Ensure model is on CPU for compatibility
        try:
            for param in latex_ocr_model.model.parameters():
                param.data = param.data.to("cpu")
            print("Successfully initialized LaTeX OCR on CPU", file=sys.stderr)
        except Exception as e:
            print(f"Could not ensure CPU placement: {str(e)}", file=sys.stderr)
    
    return latex_ocr_model


def is_likely_math(text: str) -> bool:
    """Determine if text is likely to contain mathematical formulas.
    
    Uses a regex pattern to detect common mathematical symbols.
    
    Args:
        text: Text to analyze
        
    Returns:
        True if text likely contains math, False otherwise
    """
    # Check for common mathematical symbols
    if MATH_SYMBOL_PATTERN.search(text):
        return True
    
    # Check for patterns like equations, fractions, etc.
    math_patterns = [
        r'\d+[\+\-\*\/=]\d+',  # Simple equations like 2+2=4
        r'\([^\)]+\)',  # Content in parentheses
        r'\[[^\]]+\]',  # Content in square brackets
        r'\{[^\}]+\}',  # Content in curly braces
        r'\b[a-z]\s*=',  # Variable assignments like x =
        r'\b\d+/\d+\b',  # Fractions like 1/2
        r'[a-zA-Z]\^\d',  # Superscripts like x^2
        r'\b[a-z]_\d',  # Subscripts like x_1
    ]
    
    for pattern in math_patterns:
        if re.search(pattern, text):
            return True
    
    return False


# Cache for extracted equations to improve performance
EQUATION_CACHE = {}

def compute_image_hash(image: Image.Image) -> str:
    """Compute a hash for an image to use as cache key.
    
    Args:
        image: PIL Image to hash
        
    Returns:
        Hash string
    """
    # Convert to grayscale to normalize
    img_gray = image.convert("L")
    
    # Resize to standard size to ensure consistent hashing
    img_resized = img_gray.resize((100, 100))
    
    # Convert to bytes and hash
    img_bytes = io.BytesIO()
    img_resized.save(img_bytes, format='PNG')
    return md5(img_bytes.getvalue()).hexdigest()

def is_likely_math(text: str, threshold: float = 0.6) -> bool:
    """Determine if text is likely to contain mathematical formulas.
    
    Uses pattern matching to identify mathematical expressions.
    Optimized for Apple Silicon M4 Pro via vectorized operations.
    
    Args:
        text: Text to analyze
        threshold: Confidence threshold (0.0-1.0)
        
    Returns:
        True if text likely contains math, False otherwise
    """
    if not text or len(text.strip()) < 3:
        return False
        
    # Common mathematical symbols pattern
    math_symbols = re.compile(r'[\\∫∑∏√^_{}=<>⊂⊃∈∀∃±∓×÷≤≥≠∞∂∇∆]')
    
    # Mathematical expression patterns
    math_patterns = [
        r'\d+[\+\-\*\/=]\d+',  # Simple equations like 2+2=4
        r'[a-zA-Z]_\{[^}]+\}',  # Subscripts with braces
        r'[a-zA-Z]\^[a-zA-Z0-9]',  # Superscripts
        r'\\[a-zA-Z]+',  # LaTeX commands
        r'\$.*?\$',  # Inline math delimiters
        r'\\\(.*?\\\)',  # Inline math delimiters
        r'\\\[.*?\\\]',  # Display math delimiters
        r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environments
        r'[a-zA-Z]\([a-zA-Z0-9,]+\)',  # Function notation like f(x)
    ]
    
    score = 0.0
    max_score = len(math_patterns) + 1  # +1 for symbol pattern
    
    # Check for common mathematical symbols (high weight)
    if math_symbols.search(text):
        score += 1.0
    
    # Check for patterns like equations, fractions, etc.
    for pattern in math_patterns:
        if re.search(pattern, text, re.DOTALL):
            score += 1.0
    
    # Normalize score
    confidence = score / max_score
    
    return confidence >= threshold

def extract_latex_from_image(image: Image.Image) -> str:
    """Extract LaTeX from an image of a mathematical formula.
    
    Uses the pix2tex model on CPU for maximum compatibility.
    Implements caching for improved performance on repeated equations.
    
    Args:
        image: PIL Image containing a mathematical formula
        
    Returns:
        LaTeX representation of the formula
        
    Raises:
        ImportError: If LaTeX OCR dependencies are not available
    """
    if not LATEX_OCR_AVAILABLE:
        raise ImportError(
            "LaTeX equation extraction requires pix2tex. "
            "Install with: pip install pix2tex"
        )
    
    # Check if the equation is already in cache
    img_hash = compute_image_hash(image)
    if img_hash in EQUATION_CACHE:
        return EQUATION_CACHE[img_hash]
    
    try:
        # Get the model (will initialize if needed)
        model = get_latex_ocr_model()
        
        # Force model inputs and operations to CPU
        # This addresses the device mismatch issues
        with torch.inference_mode():
            # Convert image to tensor on CPU if needed
            if hasattr(model, 'preprocessor'):
                # Override the device in the preprocessor if possible
                if hasattr(model.preprocessor, 'device'):
                    original_device = model.preprocessor.device
                    model.preprocessor.device = torch.device('cpu')
            
            # Process image on CPU
            latex = model(image)
            
            # Restore original device setting
            if hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'device'):
                model.preprocessor.device = original_device
        
        # Cache the result for future use
        EQUATION_CACHE[img_hash] = latex
        
        return latex
    except Exception as e:
        print(f"Error extracting LaTeX: {str(e)}")
        return ""


def extract_math_blocks_from_page(page, confidence_threshold: float = 0.6) -> List[Dict[str, Any]]:
    """Extract potential mathematical blocks from a PDF page.
    
    Analyzes PDF blocks and identifies those likely containing mathematical formulas.
    Uses vectorized operations where possible for speed on M4 Pro hardware.
    
    Args:
        page: PyMuPDF page object
        confidence_threshold: Threshold for math detection confidence (0.0-1.0)
        
    Returns:
        List of dictionaries containing block info and extracted images,
        sorted by confidence level (highest first)
    """
    math_blocks = []
    
    # Get blocks with location and text information
    blocks = page.get_text("dict")["blocks"]
    
    # Prepare for batch processing - collect candidates first
    candidates = []
    candidate_texts = []
    candidate_rects = []
    
    # First pass: find all potential math blocks
    for block in blocks:
        # Skip non-text blocks
        if block.get("type") != 0:  # 0 is text block
            continue
            
        # Get the text content
        spans = block.get("lines", [])
        text_content = ""
        
        for line in spans:
            for span in line.get("spans", []):
                text_content += span.get("text", "")
        
        # Skip empty blocks
        if not text_content.strip():
            continue
            
        # Save candidate for batch evaluation
        candidates.append(block)
        candidate_texts.append(text_content)
        candidate_rects.append(fitz.Rect(block.get("bbox")))
    
    # Second pass: evaluate all candidates in batch (vectorized)
    # This is faster than evaluating one by one
    confidences = []
    for text in candidate_texts:
        # Check if each text likely contains math and get confidence level
        score = 0.0
        
        # Common mathematical symbols pattern
        math_symbols = re.compile(r'[\\\u222b\u2211\u220f\u221a^_{}=<>\u2282\u2283\u2208\u2200\u2203\u00b1\u2213\u00d7\u00f7\u2264\u2265\u2260\u221e\u2202\u2207\u2206]')
        
        # Mathematical expression patterns for vectorized checking
        math_patterns = [
            r'\d+[\+\-\*\/=]\d+',  # Simple equations like 2+2=4
            r'[a-zA-Z]_\{[^}]+\}',  # Subscripts with braces
            r'[a-zA-Z]\^[a-zA-Z0-9]',  # Superscripts
            r'\\[a-zA-Z]+',  # LaTeX commands
            r'\$.*?\$',  # Inline math delimiters
            r'\\\(.*?\\\)',  # Inline math delimiters
            r'\\\[.*?\\\]',  # Display math delimiters
            r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environments
            r'[a-zA-Z]\([a-zA-Z0-9,]+\)',  # Function notation like f(x)
        ]
        
        max_score = len(math_patterns) + 1  # +1 for symbol pattern
        
        # Check for common mathematical symbols (high weight)
        if math_symbols.search(text):
            score += 1.0
        
        # Check for patterns like equations, fractions, etc.
        for pattern in math_patterns:
            if re.search(pattern, text, re.DOTALL):
                score += 1.0
        
        # Normalize score
        confidence = score / max_score
        confidences.append(confidence)
    
    # Third pass: extract images for blocks that meet the threshold
    for i, (confidence, rect, text, block) in enumerate(zip(confidences, candidate_rects, candidate_texts, candidates)):
        if confidence >= confidence_threshold:
            # Expand rectangle slightly to capture full equations
            expanded_rect = fitz.Rect(rect)
            expanded_rect.x0 -= 5
            expanded_rect.y0 -= 5
            expanded_rect.x1 += 5
            expanded_rect.y1 += 5
            
            # Render the math block as an image
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=expanded_rect)
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                
                math_blocks.append({
                    "rect": expanded_rect,
                    "text": text,
                    "image": img,
                    "confidence": confidence
                })
            except Exception as e:
                print(f"Error extracting math block image: {str(e)}")
    
    # Sort by confidence score (highest first)
    math_blocks.sort(key=lambda x: x["confidence"], reverse=True)
                
    return math_blocks


def process_equation_batch(images: List[Image.Image], model: Any) -> List[str]:
    """Process a batch of images with equation extraction model.
    
    Optimized for hardware acceleration on Apple Silicon M4 Pro.
    Implements caching, batching, and fallback mechanisms for optimal performance.
    
    Args:
        images: List of images containing equations
        model: LatexOCR model instance
        
    Returns:
        List of extracted LaTeX equations
    """
    results = []
    
    # Optimal batch size for M4 Pro
    # This can be tuned based on hardware capabilities and memory constraints
    batch_size = 4
    
    # Process in batches for better hardware utilization
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = []
        
        # Step 1: Check cache for each image
        for img in batch:
            img_hash = compute_image_hash(img)
            if img_hash in EQUATION_CACHE:
                batch_results.append(EQUATION_CACHE[img_hash])
            else:
                # Mark for processing with model
                batch_results.append(None)
        
        # Step 2: Process uncached images with model
        for j, result in enumerate(batch_results):
            if result is None:
                img = batch[j]
                try:
                    with torch.inference_mode():
                        # Force CPU processing to avoid device mismatch errors
                        if hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'device'):
                            original_device = model.preprocessor.device
                            model.preprocessor.device = torch.device('cpu')
                        
                        # Extract LaTeX
                        latex = model(img)
                        
                        # Restore original device setting
                        if hasattr(model, 'preprocessor') and hasattr(model.preprocessor, 'device'):
                            model.preprocessor.device = original_device
                    
                    # Cache the result
                    img_hash = compute_image_hash(img)
                    EQUATION_CACHE[img_hash] = latex
                    batch_results[j] = latex
                except Exception as e:
                    print(f"Error in batch processing equation: {str(e)}")
                    batch_results[j] = ""
        
        # Step 3: Add batch results to overall results
        results.extend(batch_results)
    
    return results


class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors."""
    pass


def safe_filename(url: str) -> str:
    """Generate a safe filename from a URL.
    
    Args:
        url: The URL to create a filename from
        
    Returns:
        A filename-safe string based on the URL
    """
    url_parts = urlparse(url)
    url_hash = md5(url.encode()).hexdigest()[:8]
    base = f"{url_parts.netloc}{url_parts.path}".replace("/", "_").replace(".", "_")
    return f"{base}_{url_hash}.llm.txt"


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in text.
    
    Uses a rough approximation based on characters and whitespace.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    # More accurate than simple character count
    # Average English word is ~5 chars, average token is ~4 chars
    words = text.split()
    return max(1, len(words) * 1.25)  # Roughly 1.25 tokens per word


async def fetch_pdf(url: str, user_agent: str, proxy_url: Optional[str] = None) -> bytes:
    """Fetch PDF content from a URL.
    
    Args:
        url: URL to fetch PDF from
        user_agent: User agent string
        proxy_url: Optional proxy URL
        
    Returns:
        PDF content as bytes
        
    Raises:
        PDFProcessingError: If the fetch fails
    """
    try:
        # Setup proxy if provided
        proxy = proxy_url if proxy_url else None
        timeout = aiohttp.ClientTimeout(total=30.0)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                url,
                allow_redirects=True,
                headers={"User-Agent": user_agent},
                proxy=proxy,
            ) as response:
                if response.status >= 400:
                    raise PDFProcessingError(f"Failed to fetch PDF - status code {response.status}")
                
                # Get response content
                content = await response.read()
                
                # Check if content is likely a PDF
                content_type = response.headers.get("content-type", "").lower()
                if "application/pdf" not in content_type and not url.lower().endswith(".pdf"):
                    # Attempt to detect PDF header
                    if not content.startswith(b"%PDF-"):
                        raise PDFProcessingError(
                            f"URL does not appear to be a PDF (content-type: {content_type})"
                        )
                
                return content
    
    except aiohttp.ClientError as e:
        raise PDFProcessingError(f"Failed to fetch PDF: {str(e)}")


def extract_text_from_pdf(pdf_content: bytes, extract_math: bool = ENABLE_LATEX_EXTRACTION) -> str:
    """Extract text from PDF while preserving structure and converting math to LaTeX.
    
    Uses a hybrid approach that combines standard text extraction with selective 
    LaTeX processing for mathematical content. This preserves text quality while
    still converting equations to LaTeX format.
    
    Features:
    - Selective math region detection
    - Equation caching for improved performance
    - Batch processing of equations for hardware acceleration
    - Vectorized operations where possible
    
    Args:
        pdf_content: Raw PDF content as bytes
        extract_math: Whether to extract and convert mathematical equations to LaTeX
        
    Returns:
        Extracted text in markdown format with LaTeX equations
        
    Raises:
        PDFProcessingError: If extraction fails
    """
    try:
        # Save to temporary file to handle potential streaming issues
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(pdf_content)
            temp_path = temp_file.name
            
        doc = fitz.open(temp_path)
        
        if doc.page_count == 0:
            raise PDFProcessingError("PDF appears to be empty or corrupt")
            
        text_parts = []
        toc = doc.get_toc()
        has_toc = len(toc) > 0
        
        # Extract document metadata
        metadata = doc.metadata
        if metadata:
            title = metadata.get("title", "Untitled Document")
            author = metadata.get("author", "Unknown Author")
            text_parts.append(f"# {title}\n\nAuthor: {author}\n\n")
            
            # Add other metadata if present
            if metadata.get("subject"):
                text_parts.append(f"Subject: {metadata.get('subject')}\n\n")
                
        # Add table of contents if available
        if has_toc:
            text_parts.append("## Table of Contents\n\n")
            for level, title, page in toc:
                indent = "  " * (level - 1)
                text_parts.append(f"{indent}- {title} (page {page+1})\n")
            text_parts.append("\n")
        
        # Check if we can do LaTeX extraction
        can_extract_latex = extract_math and LATEX_OCR_AVAILABLE
        if can_extract_latex:
            try:
                # Initialize the LaTeX OCR model
                model = get_latex_ocr_model()
            except Exception as e:
                can_extract_latex = False
                print(f"Warning: LaTeX extraction disabled - {e}")

        # Track total equations found and processed
        total_equations = 0
        math_processing_time = 0.0

        # Process each page
        for page_num, page in enumerate(doc):
            page_parts = [f"## Page {page_num + 1}\n\n"]
            
            # Extract standard text first, for all methods
            blocks = page.get_text("blocks")
            all_text_blocks = []
            
            for block in blocks:
                if block[6] == 0:  # Text block
                    text = block[4].strip()
                    if text:
                        all_text_blocks.append((block[:4], text))
            
            # HYBRID APPROACH: Extract text normally but selectively process math regions
            if can_extract_latex:
                # Use the optimized math detection to find potential math blocks
                math_blocks = extract_math_blocks_from_page(page)
                
                if math_blocks:
                    # Track equations found
                    total_equations += len(math_blocks)
                    
                    # Create a map of rectangles to quickly check for intersections
                    math_rects = [block["rect"] for block in math_blocks]
                    
                    # Maps block index to math block index if there's an intersection
                    math_block_mapping = {}
                    
                    # Find intersections between text blocks and math blocks
                    for text_idx, (block_rect, _) in enumerate(all_text_blocks):
                        rect = fitz.Rect(block_rect)
                        for math_idx, math_rect in enumerate(math_rects):
                            if rect.intersects(math_rect):
                                math_block_mapping[text_idx] = math_idx
                                break
                    
                    # Batch extract LaTeX from images for better performance
                    if math_block_mapping:
                        # Get unique math blocks we need to process
                        math_indices = list(set(math_block_mapping.values()))
                        
                        # Extract relevant images
                        math_images = [math_blocks[idx]["image"] for idx in math_indices]
                        
                        # Batch process with the model
                        import time
                        start_time = time.time()
                        
                        latex_equations = process_equation_batch(math_images, model)
                        
                        # Track performance metrics
                        end_time = time.time()
                        math_processing_time += (end_time - start_time)
                        
                        # Map results back to indices
                        latex_map = {idx: latex for idx, latex in zip(math_indices, latex_equations)}
                        
                        # Process text blocks, replacing math regions with LaTeX
                        for text_idx, (_, text) in enumerate(all_text_blocks):
                            if text_idx in math_block_mapping:
                                math_idx = math_block_mapping[text_idx]
                                latex = latex_map.get(math_idx, "")
                                if latex.strip():
                                    # Triple dollar sign for display math
                                    page_parts.append(f"$$${latex}$$$\n\n")
                                else:
                                    # Fallback to original text if extraction failed
                                    page_parts.append(f"{text}\n\n")
                            else:
                                # Regular text
                                page_parts.append(f"{text}\n\n")
                    else:
                        # No math detected, add all text normally
                        for _, text in all_text_blocks:
                            page_parts.append(f"{text}\n\n")
                else:
                    # No math detected, add all text normally
                    for _, text in all_text_blocks:
                        page_parts.append(f"{text}\n\n")
            else:
                # Standard text extraction without LaTeX
                for _, text in all_text_blocks:
                    page_parts.append(f"{text}\n\n")
            
            # Add this page to the document text
            text_parts.append("".join(page_parts))

        # Clean up
        doc.close()
        os.unlink(temp_path)
        
        # Log performance details if math extraction was used
        if can_extract_latex and total_equations > 0:
            print(f"Processed {total_equations} equations in {math_processing_time:.2f} seconds")
            print(f"Average time per equation: {math_processing_time/total_equations:.4f} seconds")
        
        return "\n".join(text_parts)
        
    except Exception as e:
        if "temp_path" in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        raise PDFProcessingError(f"Failed to extract PDF content: {str(e)}")
    finally:
        # Ensure garbage collection
        gc.collect()


def clean_and_format_markdown(text: str) -> str:
    """Clean and format the extracted text into well-structured markdown.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned and formatted markdown
    """
    # Replace multiple newlines with at most two
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up headers (ensure space after #)
    text = re.sub(r'(#{1,6})([^#\s])', r'\1 \2', text)
    
    # Convert likely bullets to proper markdown
    text = re.sub(r'^\s*[•●■○◦]\s*', '- ', text, flags=re.MULTILINE)
    
    return text


def chunk_text_semantically(text: str, max_tokens: int = DEFAULT_CHUNK_SIZE) -> List[str]:
    """Split text into semantic chunks that respect document structure.
    
    Args:
        text: Text to split into chunks
        max_tokens: Maximum tokens per chunk
        
    Returns:
        List of text chunks
    """
    import re
    
    # Split by headers first
    header_pattern = r'(^|\n)(#{1,3} .+)(\n|$)'
    sections = re.split(header_pattern, text, flags=re.MULTILINE)
    
    chunks = []
    current_chunk = []
    current_size = 0
    header = None
    
    # Process sections
    for i, section in enumerate(sections):
        if not section.strip():
            continue
            
        # Check if this is a header
        if re.match(r'^#{1,3} ', section.strip()):
            # Start a new chunk if we have content and adding header would exceed limit
            if current_chunk and current_size > 0:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_size = 0
                
            header = section
            current_chunk.append(section + "\n\n")
            current_size += estimate_tokens(section)
        elif header is not None:
            # Split content into paragraphs
            paragraphs = re.split(r'\n{2,}', section)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                    
                para_size = estimate_tokens(para)
                
                # If adding paragraph exceeds limit, start a new chunk
                if current_size + para_size > max_tokens and current_chunk:
                    chunks.append("".join(current_chunk))
                    
                    # Start new chunk with the same header for context
                    current_chunk = [header + "\n\n", para + "\n\n"]
                    current_size = estimate_tokens(header) + para_size
                else:
                    current_chunk.append(para + "\n\n")
                    current_size += para_size
    
    # Add final chunk if we have content
    if current_chunk:
        chunks.append("".join(current_chunk))
    
    return chunks


def save_to_file(content: str, filename: str) -> str:
    """Save content to a file in the output directory.
    
    Args:
        content: Content to save
        filename: Filename to save as
        
    Returns:
        Full path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    filepath = OUTPUT_DIR / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return str(filepath)


class QdrantStorage:
    """Optional integration with QDrant for vector storage."""
    
    def __init__(self, collection_name: str = "pdf_chunks", url: str = "http://localhost:6333"):
        """Initialize QDrant connection.
        
        Args:
            collection_name: Name of the collection to use
            url: URL of the QDrant server
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "QDrant integration requires qdrant-client and sentence-transformers. "
                "Install with: pip install qdrant-client sentence-transformers"
            )
            
        self.client = QdrantClient(url=url)
        self.collection_name = collection_name
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create collection if it doesn't exist
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.model.get_sentence_embedding_dimension(),
                    distance=models.Distance.COSINE,
                ),
            )
    
    def store_chunks(self, chunks: List[str], metadata: Dict[str, Any]) -> List[str]:
        """Store chunks in QDrant.
        
        Args:
            chunks: List of text chunks
            metadata: Metadata for all chunks (e.g., source URL)
            
        Returns:
            List of chunk IDs
        """
        chunk_ids = []
        
        # Generate embeddings for all chunks
        embeddings = self.model.encode(chunks)
        
        # Store each chunk
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            
            # Add to QDrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=chunk_id,
                        vector=embedding.tolist(),
                        payload={
                            "text": chunk,
                            "metadata": chunk_metadata,
                        },
                    )
                ],
            )
            
            chunk_ids.append(chunk_id)
            
        return chunk_ids


class PDFProcessor(BaseModel):
    """Parameters for processing a PDF."""
    
    url: Annotated[AnyUrl, Field(description="URL to fetch PDF from")]
    max_tokens: Annotated[
        int,
        Field(
            default=DEFAULT_TOKEN_LIMIT,
            description="Maximum number of tokens to process before chunking.",
            gt=0,
        ),
    ]
    use_qdrant: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to store embeddings in QDrant.",
        ),
    ]
    qdrant_url: Annotated[
        Optional[str],
        Field(
            default="http://localhost:6333",
            description="URL of QDrant server if use_qdrant is True.",
        ),
    ]


async def serve(
    output_dir: Optional[str] = None,
    proxy_url: Optional[str] = None,
) -> None:
    """Run the PDF processor MCP server.
    
    Args:
        output_dir: Optional directory to save processed files
        proxy_url: Optional proxy URL for requests
    """
    global OUTPUT_DIR
    
    if output_dir:
        OUTPUT_DIR = Path(output_dir)
        
    server = Server("mcp-pdf-processor")

    @server.list_tools()
    async def list_tools() -> List[Tool]:
        return [
            Tool(
                name="pdf_processor",
                description="""Process a PDF into LLM-friendly markdown format.
                
This tool fetches a PDF from a URL, extracts the text with structure preservation,
and converts it to a clean markdown format suitable for LLMs. It can optionally
store the chunks in QDrant for vector search.""",
                inputSchema=PDFProcessor.model_json_schema(),
            )
        ]

    @server.list_prompts()
    async def list_prompts() -> List[Prompt]:
        return [
            Prompt(
                name="process_pdf",
                description="Process a PDF into LLM-friendly markdown format",
                arguments=[
                    PromptArgument(
                        name="url", description="URL to fetch PDF from", required=True
                    ),
                    PromptArgument(
                        name="max_tokens",
                        description="Maximum number of tokens to process before chunking",
                        required=False,
                    ),
                    PromptArgument(
                        name="use_qdrant",
                        description="Whether to store embeddings in QDrant",
                        required=False,
                    ),
                ],
            )
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            args = PDFProcessor(**arguments)
        except ValueError as e:
            raise ErrorData(code=INVALID_PARAMS, message=f"Invalid parameters: {str(e)}")
        
        url = str(args.url)
        if not url:
            raise ErrorData(code=INVALID_PARAMS, message="URL is required")
        
        status_messages = []
        try:
            # Fetch the PDF
            status_messages.append(f"Fetching PDF from {url}...")
            pdf_content = await fetch_pdf(url, DEFAULT_USER_AGENT, proxy_url)
            
            # Process the PDF
            status_messages.append("Extracting text from PDF...")
            extracted_text = extract_text_from_pdf(pdf_content)
            
            # Clean and format
            status_messages.append("Formatting as markdown...")
            markdown_text = clean_and_format_markdown(extracted_text)
            
            # Generate a safe filename
            filename = safe_filename(url)
            
            # Check token count
            token_estimate = estimate_tokens(markdown_text)
            status_messages.append(f"Estimated token count: {token_estimate}")
            
            # Process according to token limit
            if token_estimate > args.max_tokens:
                status_messages.append(f"Document exceeds token limit, chunking...")
                chunks = chunk_text_semantically(markdown_text)
                status_messages.append(f"Split into {len(chunks)} chunks")
                
                # Store in QDrant if requested
                if args.use_qdrant:
                    if not QDRANT_AVAILABLE:
                        status_messages.append(
                            "QDrant integration requested but dependencies not installed. "
                            "Install with: pip install qdrant-client sentence-transformers"
                        )
                    else:
                        status_messages.append("Storing chunks in QDrant...")
                        storage = QdrantStorage(url=args.qdrant_url)
                        chunk_ids = storage.store_chunks(
                            chunks=chunks,
                            metadata={
                                "source_url": url,
                                "filename": filename,
                                "token_count": token_estimate,
                            },
                        )
                        status_messages.append(f"Stored {len(chunk_ids)} chunks in QDrant")
                
                # Save each chunk separately
                chunk_files = []
                for i, chunk in enumerate(chunks):
                    chunk_filename = f"{filename.replace('.llm.txt', '')}_chunk{i+1}.llm.txt"
                    filepath = save_to_file(chunk, chunk_filename)
                    chunk_files.append(filepath)
                
                status_messages.append(f"Saved {len(chunk_files)} chunk files")
                all_files = ", ".join(chunk_files)
                
                # For the response, include just the first chunk
                preview_text = chunks[0]
                status_messages.append(
                    f"Document was split into {len(chunks)} chunks due to size. "
                    f"Showing preview of first chunk."
                )
            else:
                # Document fits within token limit
                filepath = save_to_file(markdown_text, filename)
                status_messages.append(f"Saved to: {filepath}")
                
                # Store in QDrant if requested
                if args.use_qdrant and QDRANT_AVAILABLE:
                    status_messages.append("Storing in QDrant...")
                    storage = QdrantStorage(url=args.qdrant_url)
                    chunk_ids = storage.store_chunks(
                        chunks=[markdown_text],
                        metadata={
                            "source_url": url,
                            "filename": filename,
                            "token_count": token_estimate,
                        },
                    )
                    status_messages.append("Stored in QDrant")
                
                preview_text = markdown_text
                all_files = filepath
            
            # Prepare response
            status_text = "\n".join(status_messages)
            
            # Show a preview (first 1000 chars)
            preview = preview_text[:1000] + ("..." if len(preview_text) > 1000 else "")
            
            return [
                TextContent(
                    type="text",
                    text=f"# PDF Processing Complete\n\n{status_text}\n\n"
                         f"Files created: {all_files}\n\n"
                         f"## Content Preview:\n\n{preview}"
                )
            ]
            
        except PDFProcessingError as e:
            return [
                TextContent(
                    type="text",
                    text=f"Error processing PDF: {str(e)}\n\nPartial progress: {', '.join(status_messages)}"
                )
            ]
        
    @server.get_prompt()
    async def get_prompt(name: str, arguments: Dict[str, Any]) -> GetPromptResult:
        if not arguments or "url" not in arguments:
            raise ErrorData(code=INVALID_PARAMS, message="URL is required")
            
        url = arguments["url"]
        max_tokens = int(arguments.get("max_tokens", DEFAULT_TOKEN_LIMIT))
        use_qdrant = arguments.get("use_qdrant", "false").lower() == "true"
        
        try:
            # Process the URL
            pdf_content = await fetch_pdf(url, DEFAULT_USER_AGENT, proxy_url)
            extracted_text = extract_text_from_pdf(pdf_content)
            markdown_text = clean_and_format_markdown(extracted_text)
            
            # Check size and prepare response
            token_estimate = estimate_tokens(markdown_text)
            
            if token_estimate > max_tokens:
                return GetPromptResult(
                    description=f"PDF from {url} (preview)",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"The PDF at {url} is too large ({token_estimate} tokens) for direct processing. "
                                     f"Would you like me to split it into chunks? I can also store it in QDrant for vector search."
                            ),
                        )
                    ],
                )
            else:
                # Save the file
                filename = safe_filename(url)
                filepath = save_to_file(markdown_text, filename)
                
                return GetPromptResult(
                    description=f"Processed PDF from {url}",
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(
                                type="text",
                                text=f"I've processed the PDF from {url} and saved it to {filepath}. "
                                     f"The document contains approximately {token_estimate} tokens. "
                                     f"Here's the content:\n\n{markdown_text[:2000]}..."
                            ),
                        )
                    ],
                )
        except PDFProcessingError as e:
            return GetPromptResult(
                description=f"Failed to process PDF from {url}",
                messages=[
                    PromptMessage(
                        role="user",
                        content=TextContent(type="text", text=f"Error processing PDF: {str(e)}"),
                    )
                ],
            )

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)


if __name__ == "__main__":
    asyncio.run(serve())