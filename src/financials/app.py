"""
Main Streamlit application for the financials project.
"""
import asyncio
import logging
import streamlit as st
from unstructured.partition.auto import partition

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
        file_path (str): Path to the PDF file
        
    Returns:
        list: Parsed elements from the PDF
    """
    logger.info(f"Starting to parse PDF: {file_path}")
    try:
        elements = partition(file_path)
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
        tuple: UI elements (parse_button, document_info)
    """
    with left_col:
        st.subheader("Controls")
        parse_button = st.button("Parse PDF")
        
        st.divider()
        st.write("Document Information")
        document_info = st.empty()
        
        return parse_button, document_info

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

def display_elements(container, elements):
    """Display parsed elements in the provided container.
    
    Args:
        container: Streamlit container
        elements: List of parsed elements
    """
    with container:
        st.success("PDF parsed successfully!")
        
        for element in elements:
            with st.expander(f"Element"):
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
    parse_button, document_info = create_control_panel(left_col)
    results_container, log_container = create_results_panel(right_col)
    
    # Set up log handler
    log_handler = StreamlitLogHandler(log_container)
    logger.addHandler(log_handler)
    
    # File path for processing
    file_path = "./data/EIN-Red64-CP_575_G_Notice.pdf"
    
    # Only parse when the button is clicked
    if parse_button:
        logger.info("Parse button clicked")
        with st.spinner("Parsing document..."):
            try:
                # Parse the PDF
                elements = await parse_pdf(file_path)
                
                # Display document info in the left column
                document_info.write(f"Total elements: {len(elements)}")
                
                # Display results in the right column
                display_elements(results_container, elements)
                
            except Exception as e:
                logger.error(f"Error in main process: {str(e)}")
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
