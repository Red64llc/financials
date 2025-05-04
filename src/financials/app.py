"""
Main Streamlit application for the financials project.
"""
import asyncio
import logging
import streamlit as st
from pdf_processor import PDFProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---- Business Logic ----

async def parse_pdf(file_path):
    """Parse PDF document and return the elements.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        list: Parsed elements from the PDF
    """
    logger.info(f"Starting to parse PDF: {file_path}")
    try:
        # Construct the full path to the file in the data directory
        full_path = f"./data/{file_path}"
        # specific strategy for PDF parsing with high resolution and table inference
        elements = partition_pdf(
            full_path, 
            strategy="hi_res",
            extract_image_block_types=["Image", "Table"],
            extract_image_block_to_payload=True,
            extract_image_block_output_dir="./output/images",
        )
        logger.info(f"Successfully parsed PDF with {len(elements)} elements")
        return elements
    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        raise

# ---- UI Components ----

def create_header():
    """Create the application header."""
    st.title("Financials Dashboard")

def create_control_panel(left_col):
    """Create the control panel in the left column.
    
    Args:
        left_col: Streamlit column for controls
        
    Returns:
        tuple: UI elements (selected_files, parse_button, document_info)
    """
    with left_col:
        st.subheader("Controls")
        
        # Get list of PDF files in the data directory
        import os
        data_dir = "./data"
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            st.warning("No PDF files found in the data directory. Please add PDF files to the './data' folder.")
        
        # File selector for PDF documents
        st.write("Select files to process:")
        selected_files = st.multiselect(
            "Choose PDF files", 
            options=pdf_files,
            help="Select one or more PDF files to process"
        )
        
        # Parse button
        parse_button = st.button(
            "Parse Selected Files", 
            disabled=not selected_files,
            help="Click to start processing the selected files"
        )
        
        st.divider()
        st.write("Document Information")
        document_info = st.empty()
        
        return selected_files, parse_button, document_info

def create_results_panel(right_col):
    """Create the results panel in the right column.
    
    Args:
        right_col: Streamlit column for results
        
    Returns:
        tuple: UI elements (results_container, log_container)
    """
    with right_col:
        st.subheader("Parsing Results")
        results_container = st.container()
        
        st.divider()
        st.subheader("Console Log")
        log_container = st.container()
        
        return results_container, log_container

def display_elements(container, elements, file_name=None):
    """Display parsed elements in the provided container.
    
    Args:
        container: Streamlit container
        elements: List of parsed elements
        file_name: Optional name of the file being displayed
    """
    with container:
        if file_name:
            st.success(f"{file_name} parsed successfully!")
        else:
            st.success("PDF parsed successfully!")
        
        for element in elements:
            element_label = f"Element from {file_name}" if file_name else "Element"
            with st.expander(element_label):
                st.write(str(element))

# ---- Log Capture Handler ----

class StreamlitLogHandler(logging.Handler):
    """Custom logging handler that writes logs to a Streamlit container."""
    
    def __init__(self, container):
        super().__init__()
        self.container = container
        self.logs = []
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    def emit(self, record):
        log_entry = self.formatter.format(record)
        self.logs.append(log_entry)
        
        # Display logs in the container
        with self.container:
            st.code('\n'.join(self.logs), language="bash")

# ---- Main Application ----

async def main():
    """Main function to run the Streamlit application."""
    try:
        # Configure the page
        st.set_page_config(
            page_title="Financials",
            page_icon="💰",
            layout="wide"
        )
        
        # Create UI structure
        create_header()
        
        # Create two columns for the layout
        left_col, right_col = st.columns([1, 2])
        
        # Set up UI components
        selected_files, parse_button, document_info = create_control_panel(left_col)
        results_container, log_container = create_results_panel(right_col)
        
        # Set up log handler
        log_handler = StreamlitLogHandler(log_container)
        logger.addHandler(log_handler)
        
        # Only parse when the button is clicked and files are selected
        if parse_button and selected_files:
            logger.info(f"Parse button clicked with {len(selected_files)} file(s) selected")
            
            # Clear previous results
            results_container.empty()
            
            # Process each selected file
            all_elements = []
            
            for i, file_name in enumerate(selected_files):
                logger.info(f"Processing file {i+1}/{len(selected_files)}: {file_name}")
                
                with st.spinner(f"Processing {file_name}..."):
                    try:
                        # Parse the PDF
                        elements = await PDFProcessor.process_pdf(file_name)
                        all_elements.extend(elements)
                        
                        # Add file-specific results
                        with results_container.container():
                            st.subheader(f"Results for {file_name}")
                            st.success(f"Successfully parsed with {len(elements)} elements")
                            
                            # Display the elements for this file
                            for element in elements:
                                with st.expander(f"Element from {file_name}"):
                                    st.write(str(element))
                            
                            st.divider()
                        
                    except Exception as e:
                        logger.error(f"Error processing {file_name}: {str(e)}")
                        with results_container:
                            st.error(f"Error processing {file_name}: {str(e)}")
            
            # Update document info with summary of all processed files
            if all_elements:
                document_info.write(f"Processed {len(selected_files)} file(s) with {len(all_elements)} total elements")
    
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
