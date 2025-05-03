"""
Main Streamlit application for the financials project.
"""
import asyncio
import streamlit as st
from unstructured.partition.auto import partition

async def _parse_pdf():
    """Parse PDF document and return the elements."""
    elements = partition("./data/EIN-Red64-CP_575_G_Notice.pdf")
    return elements

def _weigh_tree(elements):
    """Weigh the document tree based on the elements."""
    # This is a placeholder for the actual weighing logic
    # In a real implementation, you would analyze the elements and assign weights
    weighted_elements = []
    for i, element in enumerate(elements):
        weighted_elements.append({
            'element': element,
            'weight': i + 1,  # Simple placeholder weight
            'content': str(element)
        })
    return weighted_elements

async def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Financials",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("Financials Dashboard")
    
    # Create two columns for the layout
    left_col, right_col = st.columns([1, 2])
    
    # Left column for controls
    with left_col:
        st.subheader("Controls")
        parse_button = st.button("Parse PDF")
        
        # Additional controls can be added here
        st.divider()
        st.write("Document Information")
        document_info = st.empty()
    
    # Right column for displaying results
    with right_col:
        st.subheader("Parsing Results")
        results_container = st.container()
        
    # Only parse when the button is clicked
    if parse_button:
        with st.spinner("Parsing document..."):
            # Parse the PDF
            elements = await _parse_pdf()
            
            # Weigh the tree
            weighted_elements = _weigh_tree(elements)
            
            # Display document info in the left column
            document_info.write(f"Total elements: {len(elements)}")
            
            # Display results in the right column
            with results_container:
                st.success("PDF parsed successfully!")
                
                # Display the weighted elements
                for item in weighted_elements:
                    with st.expander(f"Element (Weight: {item['weight']})"):
                        st.write(item['content'])


if __name__ == "__main__":
    asyncio.run(main())
