# Financials

A financial data analysis and visualization application.

## CLI Tool

The package includes a command-line interface (CLI) for processing PDF documents:

```bash
# Load a PDF file
finanalyze load path-to-file.pdf

# Process a PDF file
finanalyze process path-to-file.pdf

# Chunk a PDF file
finanalyze chunk path-to-file.pdf
```

Before using the CLI, make sure to set the `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="your-api-key"
```

## Project Structure

```
financials/
├── src/               # Source code directory
│   └── financials/    # Main package
│       ├── __init__.py
│       ├── app.py     # Streamlit application
│       ├── pdf_processor.py # PDF processing pipeline
│       └── cli/       # Command-line interface
│           ├── __init__.py
│           └── cli.py # CLI implementation
├── main.py           # Entry point script
├── pyproject.toml    # Project configuration
└── uv.lock           # Lock file for dependencies
```

## Technology Stack

- Python 3.11+
- Streamlit 1.45.0+ (UI framework)
- Pandas 2.2.3+ (Data manipulation)
- NumPy 2.2.5+ (Numerical computing)
- uv (Package manager)

## Development Environment

### Setup

1. Ensure you have Python 3.11 or higher installed
2. Install uv package manager if not already installed
3. Clone the repository
4. Install dependencies:

```bash
uv pip install -e .
```

### Running the Application

#### Directly
To start the Streamlit UI:

```bash
python main.py
```

#### Using Docker
This will launch the Streamlit server and open the application in your default web browser.

```bash
 6145  docker run financial:v0 -p 8501:8501
 6146  docker run -p 8501:8501 financial:v0
 ```

## Patterns and Best Practices

- Source code is organized in the `src` directory
- Package management is handled by `uv`
- Configuration is maintained in `pyproject.toml`
- Dependencies are locked in `uv.lock`
- CLI tools are implemented using the argparse library
- PDF processing follows a pipeline pattern with clear separation of concerns
