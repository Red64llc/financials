import streamlit as st
import pandas as pd
from pathlib import Path
import logging
from financials.chat.chat_system import ChatSystem

# Page configuration
st.set_page_config(
    page_title="Financial Data Assistant",
    page_icon="💹",
    layout="wide"
)


@st.cache_resource
def configure_logging(file_path, level=logging.INFO):
    """Configure logging to both console and file"""
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger

# Initialize ChatSystem
@st.cache_resource
def initialize_systems():
    """Initialize Weaviate client and financial extraction system"""
    logger = configure_logging("financials.log")
    with st.spinner("Connecting to Weaviate and loading models..."):
        chat_system = ChatSystem(logger=logger)
        return chat_system

# Initialize session state  
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your financial data assistant. Ask me about revenue, growth rates, or other financial metrics."}
    ]

# Main app layout
st.title("💹 Financial Data Chat Assistant")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    chat_system = initialize_systems()
    st.write(chat_system.info())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input and response handling
if prompt := st.chat_input("Ask me about your financial data"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show thinking animation
        with st.spinner("Analyzing financial data..."):
            # Process query through Weaviate and financial extraction
            full_response = chat_system.chat(prompt)
        
        # Display response (same as previous example)
        message_placeholder.markdown(full_response)
        
        # Add to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
