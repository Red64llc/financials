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

(check: https://www.gitpod.io/docs/flex/integrations/cli)

1. install gitpod cli
```
curl -o gitpod -fsSL "https://releases.gitpod.io/cli/stable/gitpod-$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m | sed 's/x86_64/amd64/;s/\(arm64\|aarch64\)/arm64/')" && \
chmod +x gitpod && \
sudo mv gitpod /usr/local/bin
```

2. check installation
```
(financials) ➜  financials git:(main) ✗ gitpod -h

      .-+*#+                 Gitpod: Always ready to code.
   :=*#####*.                Try the following commands to get started:
  .=*####*+-.    .--:
  +****=:     :=*####+       gitpod login              Login to Gitpod
  ****:   .-+*########.      gitpod whoami             Show information about the currently logged in user
  +***:   *****+--####.
  +***:   .-=:.  .#*##.      gitpod environment list   List your environments
  +***+-.      .-+****       gitpod environment create Create a new environment
  .=*****+=::-+*****+:       gitpod environment open   Open a running environment
  .:=+*********=-.           gitpod environment stop   Stop a running environment
      .-++++=:
```


This will launch the Streamlit server and open the application in your default web browser.

## Patterns and Best Practices

- Source code is organized in the `src` directory
- Package management is handled by `uv`
- Configuration is maintained in `pyproject.toml`
- Dependencies are locked in `uv.lock`
