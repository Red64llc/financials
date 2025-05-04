import streamlit as st
import pandas as pd
from pathlib import Path

from financials.chat.chat_system import ChatSystem

# Page configuration
st.set_page_config(
    page_title="Financial Data Assistant",
    page_icon="💹",
    layout="wide"
)

# Initialize ChatSystem
@st.cache_resource
def initialize_systems():
    """Initialize Weaviate client and financial extraction system"""
    with st.spinner("Connecting to Weaviate and loading models..."):
        chat_system = ChatSystem()
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

# Add visualization section if data is available
if len(st.session_state.messages) > 1:
    # Check if we have financial data to visualize
    data_path = Path("financial_analysis/revenue_data_analyzed.csv")
    if data_path.exists():
        st.markdown("---")
        st.header("Revenue Data Visualization")
        
        # Load the extracted data
        df = pd.read_csv(data_path)
        
        # Display interactive chart
        if not df.empty and 'period' in df.columns and 'value' in df.columns:
            chart_data = df[['period', 'value']].rename(columns={'value': 'Revenue'})
            
            # Add growth rate if available
            if 'growth_rate' in df.columns:
                chart_data['Growth Rate (%)'] = df['growth_rate'] * 100
            
            # Create tabs for different visualizations
            tab1, tab2 = st.tabs(["Revenue Over Time", "Growth Rate"])
            
            with tab1:
                st.bar_chart(chart_data, x='period', y='Revenue')
                
            with tab2:
                if 'Growth Rate (%)' in chart_data.columns:
                    st.line_chart(chart_data, x='period', y='Growth Rate (%)')
                else:
                    st.info("Growth rate data not available.")