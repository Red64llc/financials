# Financials

A financial data analysis and visualization application.

## CLI Tool

The package includes a command-line interface (CLI) for processing PDF documents.

### Running the CLI from Host Machine

If you've installed the package locally:

```bash
# Process a PDF file
finanalyze process path-to-file.pdf
```

If you haven't installed the package, you can run the CLI directly using the Python module:

```bash
# From the project root directory
python -m financials.cli.cli process path-to-file.pdf
```

### Environment Setup

Before using the CLI, make sure to set the required environment variables:

```bash
# Required for OpenAI embeddings
export OPENAI_API_KEY="your-api-key"

# If connecting to local Weaviate (started via docker-compose)
export WEAVIATE_HOST="localhost"
export WEAVIATE_PORT="8080"
export WEAVIATE_GRPC_PORT="50051"
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
- Weaviate (Vector database)
- LangChain (Text processing pipeline)
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

##### Running with Docker Compose

To start all services (Financials app and Weaviate database):

```bash
# From the project root directory
docker-compose up
```

This will:
- Build the Financials application image if it doesn't exist
- Start the Financials service on port 8501
- Start the Weaviate vector database on port 8080
- Set up the necessary volumes and network connections

To run in detached mode (background):

```bash
docker-compose up -d
```

To stop all services:

```bash
docker-compose down
```

##### Calling CLI from Docker

Running the CLI using the installed command:
```bash
# From the project root directory
➜  financials git:(main) ✗ docker compose exec financials-ui uv run finanalyze process -h
usage: finanalyze process [-h] pdf_path

positional arguments:
  pdf_path              Path to the PDF file

optional arguments:
  -h, --help            show this help message and exit
```

###### Processing Files from Host Machine

To process a PDF file that exists on your host machine:

```bash
# Place your PDF file in the data directory
# Then run the command (assuming your PDF is at data/example.pdf):
docker-compose exec financials-cli finanalyze process /app/data/example.pdf
```

##### Running Individual Container

Alternatively, you can run just the Financials container:

```bash
docker run -p 8501:8501 financial:v0
```

## Patterns and Best Practices

- Source code is organized in the `src` directory
- Package management is handled by `uv`
- Configuration is maintained in `pyproject.toml`
- Dependencies are locked in `uv.lock`
- CLI tools are implemented using the argparse library
- PDF processing follows a pipeline pattern with clear separation of concerns
