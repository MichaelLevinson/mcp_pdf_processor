"""
Mathematical Equation Utilities for PDF Processing
Handles detection, extraction, and caching of mathematical equations
Optimized for Apple Silicon M4 Pro
"""

import re
import io
import hashlib
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from pathlib import Path
import fitz  # PyMuPDF

# Cache for extracted equations
EQUATION_CACHE: Dict[str, str] = {}

# Common mathematical symbols and patterns for detection
MATH_SYMBOL_PATTERN = re.compile(r'[\\∫∑∏√^_{}=<>⊂⊃∈∀∃±∓×÷≤≥≠∞∂∇∆∫∮∯∰∱∲∳]')

# Mathematical expression patterns
MATH_PATTERNS = [
    r'\d+[\+\-\*\/=]\d+',  # Simple equations like 2+2=4
    r'[a-zA-Z]_\{[^}]+\}',  # Subscripts with braces
    r'[a-zA-Z]\^[a-zA-Z0-9]',  # Superscripts
    r'\\[a-zA-Z]+',  # LaTeX commands
    r'\$.*?\$',  # Inline math delimiters
    r'\\\(.*?\\\)',  # Inline math delimiters
    r'\\\[.*?\\\]',  # Display math delimiters
    r'\\begin\{.*?\}.*?\\end\{.*?\}',  # LaTeX environments
    r'[a-zA-Z]\([a-zA-Z0-9,]+\)',  # Function notation like f(x)
    r'\{.*?\}',  # Braces
    r'\(.*?\)',  # Parentheses with possibly complex content
    r'\[.*?\]',  # Brackets with complex content
    r'[a-zA-Z]\s*=\s*[^=]',  # Variable assignments
    r'\\frac\{.*?\}\{.*?\}',  # Fractions
]


def compute_hash(image: Image.Image) -> str:
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
    return hashlib.md5(img_bytes.getvalue()).hexdigest()


def is_likely_math(text: str, confidence_threshold: float = 0.6) -> Tuple[bool, float]:
    """Determine if text is likely to contain mathematical formulas.
    
    Uses a weighted scoring system based on pattern matching.
    
    Args:
        text: Text to analyze
        confidence_threshold: Threshold for positive identification
        
    Returns:
        Tuple of (is_math, confidence)
    """
    if not text or len(text.strip()) < 3:
        return False, 0.0
        
    score = 0.0
    max_score = len(MATH_PATTERNS) + 1  # +1 for symbol pattern
    
    # Check for common mathematical symbols (high weight)
    if MATH_SYMBOL_PATTERN.search(text):
        score += 1.0
    
    # Check for patterns like equations, fractions, etc.
    for pattern in MATH_PATTERNS:
        if re.search(pattern, text, re.DOTALL):
            score += 1.0
    
    # Normalize score
    confidence = score / max_score
    
    return confidence >= confidence_threshold, confidence


def get_equation_from_cache(image: Image.Image) -> Optional[str]:
    """Get cached LaTeX equation for an image if available.
    
    Args:
        image: PIL Image containing math
        
    Returns:
        Cached LaTeX or None if not in cache
    """
    img_hash = compute_hash(image)
    return EQUATION_CACHE.get(img_hash)


def add_equation_to_cache(image: Image.Image, latex: str) -> None:
    """Add LaTeX equation to cache.
    
    Args:
        image: PIL Image containing math
        latex: Extracted LaTeX equation
    """
    if not latex:
        return
        
    img_hash = compute_hash(image)
    EQUATION_CACHE[img_hash] = latex


def extract_math_blocks_from_page(
    page, 
    confidence_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """Extract potential mathematical blocks from a PDF page.
    
    Analyzes PDF blocks and identifies those likely containing mathematical formulas.
    Uses vectorized operations where possible for speed on M4 Pro hardware.
    
    Args:
        page: PyMuPDF page object
        confidence_threshold: Threshold for math detection
        
    Returns:
        List of dictionaries containing block info and extracted images
    """
    math_blocks = []
    
    # Get blocks with location and text information
    blocks = page.get_text("dict")["blocks"]
    
    # Prepare for vectorized operations
    block_texts = []
    block_rects = []
    block_indices = []
    
    # First pass: collect all text blocks for batch processing
    for i, block in enumerate(blocks):
        # Skip non-text blocks
        if block.get("type") != 0:  # 0 is text block
            continue
            
        # Get the text content
        spans = block.get("lines", [])
        text_content = ""
        
        for line in spans:
            for span in line.get("spans", []):
                text_content += span.get("text", "")
        
        if text_content.strip():
            block_texts.append(text_content)
            block_rects.append(block.get("bbox"))
            block_indices.append(i)
    
    # Batch evaluate which blocks are likely math
    is_math_results = []
    confidence_scores = []
    
    # Vectorized operation (as much as possible)
    for text in block_texts:
        is_math, confidence = is_likely_math(text, confidence_threshold)
        is_math_results.append(is_math)
        confidence_scores.append(confidence)
    
    # Second pass: process math blocks
    for i, is_math in enumerate(is_math_results):
        if not is_math:
            continue
            
        block_index = block_indices[i]
        text = block_texts[i]
        rect = fitz.Rect(block_rects[i])
        confidence = confidence_scores[i]
        
        # Expand rectangle slightly to capture full equations
        rect.x0 -= 5
        rect.y0 -= 5
        rect.x1 += 5
        rect.y1 += 5
        
        # Render the math block as an image
        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Add confidence score for better sorting/filtering
            math_blocks.append({
                "rect": rect,
                "text": text,
                "image": img,
                "confidence": confidence
            })
        except Exception as e:
            print(f"Error extracting math block image: {str(e)}")
    
    # Sort blocks by confidence (highest first)
    math_blocks.sort(key=lambda b: b["confidence"], reverse=True)
                
    return math_blocks


def process_equation_batch(
    images: List[Image.Image], 
    model: Any
) -> List[str]:
    """Process a batch of images with equation extraction model.
    
    Optimized for hardware acceleration on Apple Silicon.
    
    Args:
        images: List of images containing equations
        model: LatexOCR model instance
        
    Returns:
        List of extracted LaTeX equations
    """
    results = []
    
    # Process in batches of optimal size for the hardware
    batch_size = 4  # Can be tuned based on hardware capabilities
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        batch_results = []
        
        # Process each image in the batch
        for img in batch:
            # Check cache first
            cached_latex = get_equation_from_cache(img)
            if cached_latex is not None:
                batch_results.append(cached_latex)
                continue
                
            # Process with model if not in cache
            try:
                with torch.inference_mode():
                    latex = model(img)
                    
                # Add to cache
                add_equation_to_cache(img, latex)
                batch_results.append(latex)
            except Exception as e:
                print(f"Error in batch processing equation: {str(e)}")
                batch_results.append("")
        
        results.extend(batch_results)
    
    return results
