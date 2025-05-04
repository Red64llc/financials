"""
Command Line Interface for the financials package.

This module provides a CLI for the PDFProcessor to load, process, and chunk PDF files.
"""

import os
import argparse
import logging
from typing import Optional

from financials.pipeline.pdf_processor import PDFProcessor
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_processor() -> Optional[PDFProcessor]:
    """
    Set up the PDFProcessor with the OpenAI API key.
    
    Returns:
        PDFProcessor or None if API key is not available
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        return None
    
    return PDFProcessor(
        openai_api_key=openai_api_key,
        chunk_size=1000,
        chunk_overlap=200
    )


def process_command(args):
    """Handle the 'process' command."""
    processor = setup_processor()
    if not processor:
        return
    
    try:
        pdf_path = args.pdf_path
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Load, process, store embeddings and chunked documents
        nb_docs = processor.process_pdf_to_vectordb(pdf_path, collection_name="financials")
        
        logger.info(f"Successfully processed {nb_docs} documents from {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Financial document analysis CLI",
        prog="finanalyze"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a PDF file")
    process_parser.add_argument("pdf_path", help="Path to the PDF file")
    process_parser.set_defaults(func=process_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
