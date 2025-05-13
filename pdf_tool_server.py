#!/usr/bin/env python3
"""
PDF Processing MCP Server based on the official simple-tool example.
Implements tools for fetching, processing, and reading PDFs.
"""
import os
import sys
import asyncio
import hashlib
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any

import anyio
import click
import aiohttp
from mcp.server.lowlevel import Server
import mcp.types as types

# Globals for PDF storage
FETCHED_PDFS = {}  # Maps hash_id -> {pdf_data}
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "llm_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Configure LaTeX extraction (if available)
ENABLE_LATEX_EXTRACTION = True
try:
    from mcp_pdf_processor.server import extract_text_from_pdf, get_latex_ocr_model
    # Initialize LaTeX model early
    if ENABLE_LATEX_EXTRACTION:
        print("Initializing LaTeX OCR model...", file=sys.stderr)
        get_latex_ocr_model()
        print("LaTeX OCR model loaded", file=sys.stderr)
except ImportError:
    ENABLE_LATEX_EXTRACTION = False
    print("LaTeX extraction disabled - pix2tex not available", file=sys.stderr)
    
    # Fallback extraction function if pix2tex not available
    def extract_text_from_pdf(pdf_content: bytes, extract_math: bool = False) -> str:
        import fitz  # PyMuPDF
        try:
            # Open the PDF from memory
            doc = fitz.open(stream=pdf_content, filetype="pdf")
            text = ""
            
            # Extract text from each page
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += "\n\n"
                
            return text
        except Exception as e:
            print(f"Error extracting text: {str(e)}", file=sys.stderr)
            raise


async def fetch_pdf(url: str, user_agent: Optional[str] = None) -> str:
    """
    Fetch a PDF from a URL and cache it.
    
    Args:
        url: URL to fetch the PDF from
        user_agent: Optional user agent string
        
    Returns:
        Hash ID of the fetched PDF
    """
    import aiohttp
    
    # Compute a hash for the URL to use as ID
    pdf_hash = hashlib.md5(url.encode()).hexdigest()
    
    # Use custom headers if provided
    headers = {
        "User-Agent": user_agent or "MCP PDF Processor (MCP/Claude Integration)"
    }
    
    # Fetch the PDF
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                pdf_content = await response.read()
            
                # Cache the PDF content
                FETCHED_PDFS[pdf_hash] = {
                    "url": url,
                    "size": len(pdf_content),
                    "content": pdf_content,
                    "processed": False,
                    "timestamp": asyncio.get_event_loop().time(),
                }
            
                print(f"Fetched PDF from {url}, size: {len(pdf_content)} bytes", file=sys.stderr)
                return pdf_hash
            
    except Exception as e:
        print(f"Error fetching PDF: {str(e)}", file=sys.stderr)
        raise


async def process_pdf(pdf_hash: str, extract_latex: bool = True) -> str:
    """
    Process a previously fetched PDF.
    
    Args:
        pdf_hash: Hash ID from fetch_pdf
        extract_latex: Whether to extract LaTeX equations
        
    Returns:
        Path to the output file
    """
    try:
        # Check if PDF was fetched
        if pdf_hash not in FETCHED_PDFS:
            print(f"PDF not found with hash {pdf_hash}", file=sys.stderr)
            raise ValueError(f"PDF with hash {pdf_hash} not found. Fetch it first.")
        
        # Get PDF content
        pdf_data = FETCHED_PDFS[pdf_hash]
        pdf_content = pdf_data["content"]
        url = pdf_data["url"]
        
        # Process the PDF - run in thread pool since this is CPU-intensive
        print(f"Processing PDF {pdf_hash}, extract_math={extract_latex}", file=sys.stderr)
        extracted_text = extract_text_from_pdf(pdf_content, extract_math=extract_latex)
        
        # Save to output file
        safe_url = url.replace("://", "_").replace("/", "_").replace("?", "_")[:50]
        output_file = f"{safe_url}_{pdf_hash}.txt"
        output_path = OUTPUT_DIR / output_file
        output_path.write_text(extracted_text, encoding="utf-8")
        
        print(f"Saved processed PDF output to {output_path}", file=sys.stderr)
        
        # Update metadata
        pdf_data["processed"] = True
        pdf_data["output_file"] = output_file
        
        return output_file
        
    except Exception as e:
        print(f"Error processing PDF: {str(e)}", file=sys.stderr)
        raise


async def read_processed_pdf(filename: str) -> str:
    """
    Read the processed output of a PDF.
    
    Args:
        filename: Output filename from process_pdf
        
    Returns:
        Processed text content
    """
    try:
        # Check if file exists
        file_path = OUTPUT_DIR / filename
        if not file_path.exists():
            print(f"File not found: {file_path}", file=sys.stderr)
            raise ValueError(f"Processed file {filename} not found.")
        
        print(f"Reading processed PDF: {filename}", file=sys.stderr)
        
        # Read the content - use loop.run_in_executor for file I/O to avoid blocking
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, lambda: file_path.read_text(encoding="utf-8"))
        return content
        
    except Exception as e:
        print(f"Error reading processed PDF: {str(e)}", file=sys.stderr)
        raise


@click.command()
@click.option("--port", default=8000, help="Port to listen on for SSE")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="Transport type",
)
def main(port: int, transport: str) -> int:
    print("Starting PDF Processor MCP Server...", file=sys.stderr)
    # Use app name that will match Claude Desktop configuration
    app = Server("PDF_TOOLS")
    
    @app.list_tools()
    async def list_tools() -> List[types.Tool]:
        """List available PDF processing tools."""
        print("Listing tools...", file=sys.stderr)
        tools = [
            types.Tool(
                name="fetch_pdf",
                description="Fetch a PDF from a URL without reading it",
                inputSchema={
                    "type": "object",
                    "required": ["url"],
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL of PDF to fetch"
                        },
                        "user_agent": {
                            "type": "string",
                            "description": "Optional user agent string"
                        }
                    }
                }
            ),
            types.Tool(
                name="process_pdf",
                description="Process a previously fetched PDF",
                inputSchema={
                    "type": "object",
                    "required": ["pdf_hash"],
                    "properties": {
                        "pdf_hash": {
                            "type": "string",
                            "description": "PDF hash ID from fetch_pdf"
                        },
                        "extract_latex": {
                            "type": "boolean",
                            "description": "Whether to extract LaTeX equations",
                            "default": True
                        }
                    }
                }
            ),
            types.Tool(
                name="read_processed_pdf",
                description="Read the processed content of a PDF",
                inputSchema={
                    "type": "object",
                    "required": ["filename"],
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "Filename from process_pdf"
                        }
                    }
                }
            ),
            # Echo tool for testing connectivity
            types.Tool(
                name="echo",
                description="Echo back a message for testing connectivity",
                inputSchema={
                    "type": "object",
                    "required": ["message"],
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to echo back"
                        }
                    }
                }
            )
        ]
        print(f"Returning {len(tools)} tools", file=sys.stderr)
        return tools

    @app.call_tool()
    async def call_tool(
        name: str, arguments: Dict[str, Any]
    ) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        """Handle tool calls for PDF processing."""
        print(f"Tool call: {name} with arguments: {arguments}", file=sys.stderr)
        
        try:
            if name == "echo":
                # Simple echo tool for testing connectivity
                message = arguments.get("message", "No message provided")
                return [types.TextContent(
                    type="text",
                    text=f"Echo: {message}"
                )]
                
            elif name == "fetch_pdf":
                url = arguments.get("url")
                if not url:
                    raise ValueError("URL is required")
                    
                user_agent = arguments.get("user_agent")
                pdf_hash = await fetch_pdf(url, user_agent)
                
                return [types.TextContent(
                    type="text",
                    text=f"PDF fetched successfully. Hash ID: {pdf_hash}"
                )]
                
            elif name == "process_pdf":
                pdf_hash = arguments.get("pdf_hash")
                if not pdf_hash:
                    raise ValueError("PDF hash is required")
                    
                extract_latex = arguments.get("extract_latex", True)
                # Convert string to boolean if needed
                if isinstance(extract_latex, str):
                    extract_latex = extract_latex.lower() in ('true', 'yes', '1')
                    
                output_file = await process_pdf(pdf_hash, extract_latex)
                
                return [types.TextContent(
                    type="text",
                    text=f"PDF processed successfully. Output filename: {output_file}"
                )]
                
            elif name == "read_processed_pdf":
                filename = arguments.get("filename")
                if not filename:
                    raise ValueError("Filename is required")
                    
                content = await read_processed_pdf(filename)
                
                return [types.TextContent(
                    type="text",
                    text=content
                )]
                
            else:
                raise ValueError(f"Unknown tool: {name}")
                
        except Exception as e:
            print(f"Error in tool call: {str(e)}", file=sys.stderr)
            # Return error message as text
            return [types.TextContent(
                type="text",
                text=f"Error: {str(e)}"
            )]

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )
            return Response()

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/sse", endpoint=handle_sse, methods=["GET"]),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn
        uvicorn.run(starlette_app, host="0.0.0.0", port=port)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)

    return 0


if __name__ == "__main__":
    sys.exit(main())
