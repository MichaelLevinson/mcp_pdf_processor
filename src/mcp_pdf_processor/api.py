"""
MCP PDF Processor API Server
Provides endpoints for Claude Desktop and Claude Code integration

This API server allows for:
1. PDF processing via direct upload
2. PDF processing via URL
3. Asynchronous processing with status tracking
4. Direct integration with Claude's retrieval interface

Optimized for Apple Silicon M4 Pro with fallbacks for other hardware.
"""

import os
import asyncio
import tempfile
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request, Response
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl, validator

# Import local modules for PDF processing
from .server import (
    fetch_pdf,
    extract_text_from_pdf,
    get_latex_ocr_model,
    ENABLE_LATEX_EXTRACTION
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MCP PDF Processor API",
    description="API for extracting text and LaTeX equations from PDF documents",
    version="1.0.0",
)

# Enable CORS for integration with web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "api_output"))
OUTPUT_DIR.mkdir(exist_ok=True)

# Track processing jobs
active_jobs: Dict[str, Dict[str, Any]] = {}

# Data models
class PDFUrlRequest(BaseModel):
    url: HttpUrl
    extract_latex: bool = True
    user_agent: Optional[str] = None
    
    @validator('url')
    def validate_pdf_url(cls, v):
        """Ensure URL appears to be a PDF."""
        if not str(v).lower().endswith('.pdf') and 'pdf' not in str(v).lower():
            # Allow URLs that might be PDF even if not obvious from extension
            if not any(domain in str(v).lower() for domain in ['arxiv.org', 'papers', 'article']):
                raise ValueError("URL does not appear to be a PDF")
        return v

class ProcessingStatus(BaseModel):
    job_id: str
    status: str
    progress: Optional[float] = None
    started_at: str
    completed_at: Optional[str] = None
    output_url: Optional[str] = None
    error: Optional[str] = None
    
class ProcessingResponse(BaseModel):
    job_id: str
    message: str
    status_url: str

class ClaudeChunk(BaseModel):
    """Model for Claude retrieval data format."""
    document_id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Chunk content to be retrieved")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata for the chunk")
    position: int = Field(..., description="Position of this chunk in the document")
    embedding: Optional[List[float]] = None

# API routes
@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "name": "MCP PDF Processor API",
        "version": "1.0.0",
        "description": "API for extracting text and LaTeX equations from PDF documents",
        "endpoints": {
            "POST /process/url": "Process PDF from URL",
            "POST /process/upload": "Process uploaded PDF",
            "GET /status/{job_id}": "Check job status",
            "GET /output/{job_id}": "Get processed output",
            "POST /claude/process": "Process PDF and return Claude-compatible chunks"
        }
    }

@app.post("/process/url", response_model=ProcessingResponse)
async def process_pdf_url(request: PDFUrlRequest, background_tasks: BackgroundTasks):
    """Process a PDF from a URL and return job ID for status tracking."""
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    active_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "started_at": datetime.now().isoformat(),
        "url": str(request.url)
    }
    
    # Launch background task for processing
    background_tasks.add_task(
        process_pdf_background,
        job_id=job_id,
        url=str(request.url),
        extract_latex=request.extract_latex,
        user_agent=request.user_agent or DEFAULT_USER_AGENT
    )
    
    return ProcessingResponse(
        job_id=job_id,
        message="PDF processing started",
        status_url=f"/status/{job_id}"
    )

@app.post("/process/upload", response_model=ProcessingResponse)
async def process_uploaded_pdf(
    extract_latex: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    file: UploadFile = File(...),
):
    """Process an uploaded PDF file and return job ID for status tracking."""
    job_id = str(uuid.uuid4())
    
    # Verify file is a PDF
    if not file.filename.lower().endswith('.pdf'):
        content_type = file.content_type
        if content_type and 'pdf' not in content_type.lower():
            raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Save uploaded file temporarily
    temp_dir = Path(tempfile.mkdtemp())
    temp_file = temp_dir / f"{job_id}.pdf"
    
    try:
        # Read and save content
        content = await file.read()
        temp_file.write_bytes(content)
        
        # Initialize job status
        active_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0.0,
            "started_at": datetime.now().isoformat(),
            "filename": file.filename
        }
        
        # Launch background task for processing
        background_tasks.add_task(
            process_pdf_background,
            job_id=job_id,
            pdf_path=str(temp_file),
            extract_latex=extract_latex,
            original_filename=file.filename
        )
        
        return ProcessingResponse(
            job_id=job_id,
            message="PDF processing started",
            status_url=f"/status/{job_id}"
        )
    except Exception as e:
        # Clean up on error
        if temp_file.exists():
            temp_file.unlink()
        
        logger.error(f"Error processing uploaded PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_job_status(job_id: str):
    """Get the status of a processing job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    # Build status response
    return ProcessingStatus(
        job_id=job_id,
        status=job.get("status", "unknown"),
        progress=job.get("progress", 0.0),
        started_at=job.get("started_at", ""),
        completed_at=job.get("completed_at"),
        output_url=f"/output/{job_id}" if job.get("status") == "completed" else None,
        error=job.get("error")
    )

@app.get("/output/{job_id}")
async def get_processed_output(job_id: str):
    """Get the processed output for a completed job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed")
    
    output_file = Path(job.get("output_file", ""))
    
    if not output_file.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=str(output_file),
        filename=output_file.name,
        media_type="text/markdown"
    )

@app.post("/claude/process")
async def process_for_claude(request: PDFUrlRequest, background_tasks: BackgroundTasks):
    """
    Process a PDF for Claude and return the document ID that 
    can be used with the Claude retrieval interface.
    """
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    active_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0.0,
        "started_at": datetime.now().isoformat(),
        "url": str(request.url),
        "for_claude": True
    }
    
    # Launch background task for processing
    background_tasks.add_task(
        process_pdf_background,
        job_id=job_id,
        url=str(request.url),
        extract_latex=request.extract_latex,
        user_agent=request.user_agent or DEFAULT_USER_AGENT,
        for_claude=True
    )
    
    return {
        "job_id": job_id,
        "message": "PDF processing started for Claude integration",
        "status_url": f"/status/{job_id}",
        "claude_chunks_url": f"/claude/chunks/{job_id}"
    }

@app.get("/claude/chunks/{job_id}")
async def get_claude_chunks(job_id: str, request: Request):
    """Get processed PDF content in Claude-compatible chunks format."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed")
    
    chunks_file = Path(job.get("claude_chunks_file", ""))
    
    if not chunks_file.exists():
        raise HTTPException(status_code=404, detail="Claude chunks not found")
    
    # Construct the base URL for Claude to access chunks
    base_url = str(request.base_url).rstrip('/')
    
    # Return the chunks metadata with URLs for Claude
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)
    
    # Add URLs for Claude to access individual chunks
    for chunk in chunks_data["chunks"]:
        chunk["url"] = f"{base_url}/claude/chunk/{job_id}/{chunk['position']}"
    
    return chunks_data

@app.get("/claude/chunk/{job_id}/{position}")
async def get_claude_chunk(job_id: str, position: int):
    """Get a specific chunk of processed PDF for Claude."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed")
    
    chunks_file = Path(job.get("claude_chunks_file", ""))
    
    if not chunks_file.exists():
        raise HTTPException(status_code=404, detail="Claude chunks not found")
    
    # Load chunks and find the requested position
    with open(chunks_file, 'r') as f:
        chunks_data = json.load(f)
    
    for chunk in chunks_data["chunks"]:
        if chunk["position"] == position:
            return ClaudeChunk(**chunk)
    
    raise HTTPException(status_code=404, detail=f"Chunk at position {position} not found")

# --- Processing functions ---

async def process_pdf_background(
    job_id: str, 
    url: Optional[str] = None, 
    pdf_path: Optional[str] = None,
    extract_latex: bool = True,
    user_agent: Optional[str] = None,
    original_filename: Optional[str] = None,
    for_claude: bool = False
):
    """Process PDF in the background, updating job status."""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "processing"
        active_jobs[job_id]["progress"] = 10.0
        
        # Get PDF content
        if url:
            # Fetch PDF from URL
            try:
                pdf_content = await fetch_pdf(url, user_agent or DEFAULT_USER_AGENT)
                filename = url.split("/")[-1]
                if not filename.lower().endswith('.pdf'):
                    filename = f"document_{job_id}.pdf"
            except Exception as e:
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["error"] = f"Failed to fetch PDF: {str(e)}"
                return
        elif pdf_path:
            # Read PDF from file
            try:
                with open(pdf_path, 'rb') as f:
                    pdf_content = f.read()
                filename = original_filename or Path(pdf_path).name
            except Exception as e:
                active_jobs[job_id]["status"] = "failed"
                active_jobs[job_id]["error"] = f"Failed to read PDF: {str(e)}"
                return
        else:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = "No URL or file provided"
            return
        
        # Update progress
        active_jobs[job_id]["progress"] = 30.0
        
        # Extract text with LaTeX
        try:
            # Pre-load model for performance
            if extract_latex and ENABLE_LATEX_EXTRACTION:
                model = get_latex_ocr_model()
                logger.info(f"LaTeX OCR model loaded successfully")
            
            # Extract text from PDF
            extracted_text = extract_text_from_pdf(pdf_content, extract_latex=extract_latex)
        except Exception as e:
            active_jobs[job_id]["status"] = "failed"
            active_jobs[job_id]["error"] = f"Failed to extract text: {str(e)}"
            logger.error(f"Text extraction error: {str(e)}")
            return
        
        # Update progress
        active_jobs[job_id]["progress"] = 70.0
        
        # Save extracted text
        safe_filename = filename.replace(' ', '_').replace('/', '_')
        output_file = OUTPUT_DIR / f"{job_id}_{safe_filename}.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        
        # Update job info
        active_jobs[job_id]["output_file"] = str(output_file)
        
        # For Claude integration, create chunks
        if for_claude:
            chunks = create_claude_chunks(extracted_text, job_id, safe_filename)
            chunks_file = OUTPUT_DIR / f"{job_id}_claude_chunks.json"
            
            with open(chunks_file, 'w', encoding='utf-8') as f:
                json.dump({"document_id": job_id, "chunks": chunks}, f)
            
            active_jobs[job_id]["claude_chunks_file"] = str(chunks_file)
        
        # Complete job
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["progress"] = 100.0
        active_jobs[job_id]["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = f"Processing error: {str(e)}"
        logger.error(f"Job {job_id} failed with error: {str(e)}")

def create_claude_chunks(text: str, document_id: str, filename: str) -> List[Dict[str, Any]]:
    """
    Create Claude-compatible chunks from the extracted text.
    
    Args:
        text: Extracted text with LaTeX
        document_id: Unique identifier for the document
        filename: Original filename
        
    Returns:
        List of chunk objects in Claude format
    """
    # Split document into chunks by headings or fixed length
    chunks = []
    
    # First, try to split by headings (chapters, sections)
    sections = []
    current_section = []
    
    for line in text.split('\n'):
        if line.startswith('#'):
            # New section found
            if current_section:
                sections.append('\n'.join(current_section))
                current_section = []
        current_section.append(line)
    
    # Add the last section
    if current_section:
        sections.append('\n'.join(current_section))
    
    # If no sections were found, create fixed-size chunks
    if not sections:
        # Chunk size of ~1500 characters
        chunk_size = 1500
        for i in range(0, len(text), chunk_size):
            sections.append(text[i:i+chunk_size])
    
    # Create Claude chunks
    for i, section in enumerate(sections):
        chunks.append({
            "document_id": document_id,
            "content": section,
            "metadata": {
                "filename": filename,
                "section_number": i + 1,
                "total_sections": len(sections)
            },
            "position": i
        })
    
    return chunks

# API server startup and hardware detection
@app.on_event("startup")
async def startup_event():
    """Startup event handler to detect hardware and optimize settings."""
    # Check for MPS availability (Apple Silicon)
    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()
    
    if has_mps:
        logger.info("Apple Silicon detected: MPS available for hardware acceleration")
    elif has_cuda:
        logger.info("NVIDIA GPU detected: CUDA available for hardware acceleration")
    else:
        logger.info("Running on CPU: No hardware acceleration detected")

def start_server(host="0.0.0.0", port=8000):
    """Start the API server."""
    uvicorn.run("mcp_pdf_processor.api:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    # Start the API server directly when run as a script
    start_server()
