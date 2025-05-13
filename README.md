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

1. Install the package:
   ```bash
   pip install -e .
   ```

2. Register the server with Claude using the MCP CLI tool:
   ```bash
   mcp server:register PDF_TOOLS --command "python -m pdf_tool_server" \
      --working-dir "/path/to/installation/directory"
   ```

3. Verify the server is registered:
   ```bash
   mcp server:list
   ```

4. In Claude Desktop or Claude Code, you can now use the PDF_TOOLS server in your conversations.

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

See `pyproject.toml` for detailed dependencies. Key requirements include:
- Python 3.9+
- PyMuPDF (PDF processing)
- pix2tex (LaTeX equation extraction)
- MCP 1.1.3+ (Model Context Protocol)

## License

MIT
