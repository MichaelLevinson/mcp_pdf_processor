# PDF Processor MCP Server

A Model Context Protocol (MCP) server for processing PDF documents with advanced features including LaTeX equation extraction. This server enables Claude to fetch, process, and extract information from PDF documents, including LaTeX mathematical equations.

## Features

- PDF fetching from URLs
- Text extraction from PDFs
- LaTeX equation recognition and extraction
- Integration with Claude via MCP

## Installation

### Standard Installation

```bash
pip install -e .
```

### Installing for Claude Desktop/Claude Code

To use this MCP server with Claude Desktop or Claude Code:

1. Install the MCP CLI tools if not already installed:
   ```bash
   pip install "mcp[cli]"
   ```

2. Install the server using the MCP CLI tool:
   ```bash
   mcp install /path/to/pdf_tool_server.py --with-editable /path/to/mcp_pdf_processor
   ```
   
   For example, if you've cloned this repository to `~/mcp_pdf_processor`:
   ```bash
   mcp install ~/mcp_pdf_processor/pdf_tool_server.py --with-editable ~/mcp_pdf_processor
   ```

3. For development with the MCP Inspector:
   ```bash
   mcp dev /path/to/pdf_tool_server.py --with-editable /path/to/mcp_pdf_processor
   ```

4. In Claude Desktop, you can now use the PDF_TOOLS server in your conversations with these commands:
   ```
   /mcp PDF_TOOLS fetch_pdf url=https://example.com/document.pdf
   /mcp PDF_TOOLS process_pdf hash_id=<HASH_ID> extract_latex=true
   /mcp PDF_TOOLS read_processed_pdf filename=<FILENAME>
   ```

## Usage

### Running Standalone

```bash
python pdf_tool_server.py
```

### Environment Variables
- `OUTPUT_DIR`: Directory to store processed PDFs (default: `llm_output`)
- `PYTHONPATH`: Set to the directory containing the mcp_pdf_processor package

### Using with Claude

When the server is registered, you can ask Claude to:
- "Fetch and analyze the PDF at [URL]"
- "Extract LaTeX equations from the PDF at [URL]"
- "Summarize the content of the PDF at [URL]"

## Requirements

The server requires the following main dependencies:

- Python 3.9 or higher
- `pymupdf`: PDF processing and text extraction
- `mcp`: Model Context Protocol support
- `pydantic`: Data validation and serialization
- `aiohttp`: Asynchronous HTTP client/server
- `torch`: For LaTeX equation extraction (optional)
- `pix2tex`: For LaTeX equation recognition (optional)

See `pyproject.toml` for the complete list of dependencies and version requirements.

## Usage Examples

Here's a complete example workflow for using the PDF processor with Claude Desktop:

```
# 1. Fetch a PDF without reading it
/mcp PDF_TOOLS fetch_pdf url=https://arxiv.org/pdf/2505.05522

# This returns a hash_id, which you'll use in the next step

# 2. Process the PDF with LaTeX extraction
/mcp PDF_TOOLS process_pdf hash_id=<HASH_ID> extract_latex=true

# This returns a filename for the processed output

# 3. Read the processed content
/mcp PDF_TOOLS read_processed_pdf filename=<FILENAME>

# Now Claude can analyze the PDF content, including any LaTeX equations
```

## License

MIT
