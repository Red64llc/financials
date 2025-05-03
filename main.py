#!/usr/bin/env python3
"""
Main entry point for the Financials application.
This script launches the Streamlit UI.
"""
import subprocess
import sys
from pathlib import Path


def main():
    """
    Launch the Streamlit application.
    """
    app_path = Path(__file__).parent / "src" / "financials" / "app.py"
    print(f"Starting Financials Streamlit application from {app_path}")
    
    # Launch Streamlit with the app file
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        str(app_path), "--server.headless", "true", 
        "--server.enableCORS", "false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error launching Streamlit application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
