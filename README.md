# Financials

A financial data analysis and visualization application.

## Project Structure

```
financials/
├── src/               # Source code directory
│   └── financials/    # Main package
│       ├── __init__.py
│       └── app.py     # Streamlit application
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

#### Using Gitpod

1. install gitpod cli
```
Warning: gitpod-io/tap/gitpod 0.1.4 is already installed and up-to-date.
To reinstall 0.1.4, run:
  brew reinstall gitpod
```

2. 

This will launch the Streamlit server and open the application in your default web browser.

## Patterns and Best Practices

- Source code is organized in the `src` directory
- Package management is handled by `uv`
- Configuration is maintained in `pyproject.toml`
- Dependencies are locked in `uv.lock`
