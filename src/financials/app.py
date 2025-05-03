"""
Main Streamlit application for the financials project.
"""
import streamlit as st

def main():
    """
    Main function to run the Streamlit application.
    """
    st.set_page_config(
        page_title="Financials",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("Financials Dashboard")
    st.subheader("Welcome to the Financials application")
    
    st.write("This is a basic Streamlit UI to verify that the Git pod is running properly.")
    
    # Display some sample content
    st.info("Everything is working correctly!")
    
    # Add a simple interactive element
    if st.button("Click me!"):
        st.success("Button clicked successfully!")

if __name__ == "__main__":
    main()
