"""
MCP Server for Claude PDF Processing

This MCP server implements a set of tools that allow Claude to:
1. Fetch PDFs without directly reading them
2. Process PDFs with LaTeX extraction
3. Read the processed output for context

Optimized for Apple Silicon with fallbacks for other hardware.
"""

import os
import sys
import json
import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Add parent directory to path if needed for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import local modules for PDF processing
from mcp_pdf_processor.server import (
    fetch_pdf,
    extract_text_from_pdf,
    get_latex_ocr_model,
    ENABLE_LATEX_EXTRACTION
)

# Import MCP server components
from mcp.server import Server
from mcp.server.stdio import stdio_server
import sys
from mcp.types import (
    Request,
    Result,
    ServerResult,
    JSONRPCError,
    PaginatedRequest,
    PaginatedResult,
    ResourceReference,
    RequestParams,
    Resource,
    ResourceTemplate
)

# Configure output directory
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "llm_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Track fetched PDFs
FETCHED_PDFS: Dict[str, Dict[str, Any]] = {}

# Default user agent for PDF fetching
DEFAULT_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"

def compute_url_hash(url: str) -> str:
    """
    Compute a hash for a URL to use as file identifier.
    
    Args:
        url: URL to hash
        
    Returns:
        Hash string (MD5)
    """
    return hashlib.md5(url.encode()).hexdigest()[:12]

async def fetch_pdf_mcp(url: str, user_agent: Optional[str] = None) -> str:
    """
    MCP wrapper for fetching a PDF without reading it.
    
    Args:
        url: URL of the PDF to fetch
        user_agent: Optional user agent string
        
    Returns:
        Hash ID of the fetched PDF
    """
    try:
        # Fetch the PDF
        pdf_content = await fetch_pdf(url, user_agent or DEFAULT_USER_AGENT)
        
        # Compute hash for the URL (not content, to keep consistent)
        pdf_hash = compute_url_hash(url)
        
        # Save metadata about the fetched PDF
        FETCHED_PDFS[pdf_hash] = {
            "url": url,
            "size": len(pdf_content),
            "content": pdf_content,  # Store in memory for now
            "timestamp": asyncio.get_event_loop().time(),
        }
        
        print(f"Fetched PDF from {url}, size: {len(pdf_content)} bytes", file=sys.stderr)
        
        # Return the hash ID
        return pdf_hash
    except Exception as e:
        raise Exception(f"Failed to fetch PDF: {str(e)}")

async def process_pdf_mcp(pdf_hash: str, extract_latex: bool = True) -> str:
    """
    MCP wrapper for processing a fetched PDF.
    
    Args:
        pdf_hash: Hash ID of a previously fetched PDF
        extract_latex: Whether to extract LaTeX equations
        
    Returns:
        Path to the processed output file
    """
    try:
        # Check if PDF was fetched
        if pdf_hash not in FETCHED_PDFS:
            print(f"PDF not found with hash {pdf_hash}", file=sys.stderr)
            raise Exception(f"PDF with hash {pdf_hash} not found. Fetch it first.")
        
        # Get PDF content
        pdf_data = FETCHED_PDFS[pdf_hash]
        pdf_content = pdf_data["content"]
        url = pdf_data["url"]
        
        # Process the PDF
        print(f"Processing PDF {pdf_hash}, extract_latex={extract_latex}", file=sys.stderr)
        extracted_text = extract_text_from_pdf(pdf_content, extract_latex=extract_latex)
        
        # Save to output file
        safe_url = url.replace("://", "_").replace("/", "_").replace(".", "_")
        output_path = OUTPUT_DIR / f"{safe_url}_{pdf_hash}.llm.txt"
        output_path.write_text(extracted_text, encoding="utf-8")
        
        print(f"Saved processed PDF output to {output_path}", file=sys.stderr)
        
        # Update metadata
        pdf_data["processed"] = True
        pdf_data["output_path"] = str(output_path)
        
        # Optional: Release the PDF content to save memory
        # pdf_data.pop("content", None)
        
        return output_path.name
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")

async def read_processed_pdf(filename: str) -> str:
    """
    MCP wrapper for reading a processed PDF output.
    
    Args:
        filename: Name of the processed output file
        
    Returns:
        Processed text content
    """
    try:
        # Find the file
        file_path = OUTPUT_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            print(f"File not found: {file_path}", file=sys.stderr)
            raise Exception(f"Processed file {filename} not found.")
        
        print(f"Reading processed PDF: {filename}", file=sys.stderr)
        
        # Read the content
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise Exception(f"Failed to read processed PDF: {str(e)}")

class PDFProcessorMCPServer(Server):
    """MCP Server for PDF processing operations."""

    def __init__(self):
        super().__init__(name="PDF_PROCESSOR")
        self.initialized = False
    
    async def handle_request(self, request: Request) -> ServerResult:
        """Handle incoming MCP requests."""
        try:
            # Handle initialization lifecycle first
            if request.method == "initialize":
                print(f"Received initialize request with protocol version: {request.params.get('protocolVersion', 'unknown')}", file=sys.stderr)
                self.initialized = False
                return Result(value={
                    "capabilities": {
                        "tools": True,
                        "resources": True
                    }
                })
            
            # Handle initialized notification
            elif request.method == "notifications/initialized":
                self.initialized = True
                print("Received initialized notification, server is now ready", file=sys.stderr)
                return None
                
            # Only process other methods after initialization
            elif not self.initialized:
                print(f"Warning: Received {request.method} before initialization was complete", file=sys.stderr)
                return JSONRPCError(
                    code=-32600,
                    message="Server has not been initialized"
                )
                
            # Normal operation - after initialization
            elif request.method == "tools/call":
                # Get the tool name from the parameters
                tool_name = request.params.get("tool_name", "")
                arguments = request.params.get("arguments", {})
                
                print(f"Tool call: {tool_name} with args: {arguments}", file=sys.stderr)
                
                # Handle PDF fetching
                if tool_name == "fetch_pdf":
                    url = arguments.get("url")
                    user_agent = arguments.get("user_agent")
                    
                    if not url:
                        return JSONRPCError(code=-32602, message="URL is required")
                    
                    try:
                        pdf_hash = await fetch_pdf_mcp(url, user_agent)
                        
                        return Result(value={
                            "hash_id": pdf_hash,
                            "message": f"PDF fetched successfully. Use this hash_id with process_pdf."
                        })
                    except Exception as e:
                        return JSONRPCError(code=-32603, message=str(e))
                
                # Handle process_pdf command
                elif tool_name == "process_pdf":
                    pdf_hash = arguments.get("pdf_hash")
                    extract_latex = arguments.get("extract_latex", True)
                    
                    if not pdf_hash:
                        return JSONRPCError(code=-32602, message="pdf_hash is required")
                    
                    # Convert string to boolean if needed
                    if isinstance(extract_latex, str):
                        extract_latex = extract_latex.lower() in ('true', 'yes', '1')
                    
                    try:
                        output_file = await process_pdf_mcp(pdf_hash, extract_latex)
                        
                        return Result(value={
                            "output_file": output_file,
                            "message": f"PDF processed successfully. Use this output_file with read_processed_pdf."
                        })
                    except Exception as e:
                        return JSONRPCError(code=-32603, message=str(e))
                
                # Handle reading processed PDF content
                elif tool_name == "read_processed_pdf":
                    filename = arguments.get("filename")
                
                    if not filename:
                        return JSONRPCError(code=-32602, message="filename is required")
                    
                    try:
                        content = await read_processed_pdf(filename)
                        
                        return Result(value={
                            "content": content,
                        })
                    except Exception as e:
                        return JSONRPCError(code=-32603, message=str(e))
                
                # Unknown tool
                else:
                    return JSONRPCError(
                        code=-32601, 
                        message=f"Unknown tool: {tool_name}"
                    )
            
            # Unknown method
            else:
                return JSONRPCError(code=-32601, message=f"Unknown method: {request.method}")
                
        except Exception as e:
            return JSONRPCError(
                code=-32603,
                message=str(e),
            )
    
    async def list_resources(self, request: PaginatedRequest) -> PaginatedResult:
        """List available MCP resources (commands)."""
        resources = [
            Resource(
                uri="fetch_pdf",
                name="Fetch PDF",
                description="Download a PDF from a URL without reading it",
            ),
            Resource(
                uri="process_pdf",
                name="Process PDF",
                description="Process a previously fetched PDF with optional LaTeX extraction",
            ),
            Resource(
                uri="read_processed_pdf",
                name="Read Processed PDF",
                description="Read the processed content of a PDF",
            ),
        ]
        
        return PaginatedResult(resources=resources)

def main():
    """Start the MCP server for PDF processing."""
    print("Initializing server...", file=sys.stderr)
    # Initialize the LaTeX OCR model at startup if needed
    if ENABLE_LATEX_EXTRACTION:
        try:
            get_latex_ocr_model()
            print("LaTeX OCR model loaded", file=sys.stderr)
        except Exception as e:
            print(f"Warning: LaTeX extraction unavailable - {str(e)}", file=sys.stderr)
    
    # Start the MCP server
    stdio_server(PDFProcessorMCPServer())

if __name__ == "__main__":
    main()
