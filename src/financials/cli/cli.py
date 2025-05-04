"""
Command Line Interface for the financials package.

This module provides a CLI for the PDFProcessor to load, process, and chunk PDF files.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import Optional

from financials.pdf_processor import PDFProcessor
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
        chunk_overlap=200,
        db_directory="./chroma_db"
    )


def load_command(args):
    """Handle the 'load' command."""
    processor = setup_processor()
    if not processor:
        return
    
    try:
        pdf_path = args.pdf_path
        logger.info(f"Loading PDF: {pdf_path}")
        elements = processor.load_pdf_with_unstructured(pdf_path)
        logger.info(f"Successfully loaded {len(elements)} elements from {pdf_path}")
    except Exception as e:
        logger.error(f"Error loading PDF: {str(e)}")


def process_command(args):
    """Handle the 'process' command."""
    processor = setup_processor()
    if not processor:
        return
    
    try:
        pdf_path = args.pdf_path
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Load and process the PDF
        elements = processor.load_pdf_with_unstructured(pdf_path)
        documents = processor.process_elements(elements)
        
        logger.info(f"Successfully processed {len(documents)} documents from {pdf_path}")
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")


def chunk_command(args):
    """Handle the 'chunk' command."""
    processor = setup_processor()
    if not processor:
        return
    
    try:
        pdf_path = args.pdf_path
        logger.info(f"Chunking PDF: {pdf_path}")
        
        # Load, process, and chunk the PDF
        elements = processor.load_pdf_with_unstructured(pdf_path)
        documents = processor.process_elements(elements)
        chunks = processor.chunk_documents(documents)
        
        logger.info(f"Successfully created {len(chunks)} chunks from {pdf_path}")
    except Exception as e:
        logger.error(f"Error chunking PDF: {str(e)}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Financial document analysis CLI",
        prog="finanalyze"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Load command
    load_parser = subparsers.add_parser("load", help="Load a PDF file")
    load_parser.add_argument("pdf_path", help="Path to the PDF file")
    load_parser.set_defaults(func=load_command)
    
    # Process command
    process_parser = subparsers.add_parser("process", help="Process a PDF file")
    process_parser.add_argument("pdf_path", help="Path to the PDF file")
    process_parser.set_defaults(func=process_command)
    
    # Chunk command
    chunk_parser = subparsers.add_parser("chunk", help="Chunk a PDF file")
    chunk_parser.add_argument("pdf_path", help="Path to the PDF file")
    chunk_parser.set_defaults(func=chunk_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
