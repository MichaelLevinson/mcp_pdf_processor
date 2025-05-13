from typing import Optional
import argparse
import asyncio
from .server import serve

def main():
    """MCP PDF Processor Server - PDF processing functionality for MCP"""
    parser = argparse.ArgumentParser(
        description="Give a model the ability to process PDFs into LLM-friendly format"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory to save processed files"
    )
    parser.add_argument(
        "--proxy-url", 
        type=str, 
        help="Proxy URL to use for requests"
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default="http://localhost:6333",
        help="URL for QDrant server if using vector storage"
    )
    
    args = parser.parse_args()
    asyncio.run(serve(output_dir=args.output_dir, proxy_url=args.proxy_url))

if __name__ == "__main__":
    main()