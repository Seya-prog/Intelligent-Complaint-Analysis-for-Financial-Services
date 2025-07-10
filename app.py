"""Streamlit app for interactive RAG chat interface.
Run with:
    streamlit run app.py
"""
import streamlit as st
import sys
import importlib
from pathlib import Path

# Ensure `src/` is on PYTHONPATH so that `import rag` works when running via Streamlit
root = Path(__file__).resolve().parent
src_path = root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Force reload all rag modules
import rag
import rag.pipeline
import rag.generator
import rag.retriever
importlib.reload(rag.generator)
importlib.reload(rag.retriever)
importlib.reload(rag.pipeline)
importlib.reload(rag)

from rag.pipeline import RAGPipeline

# Set page config
st.set_page_config(
    page_title="CrediTrust Complaint Analyst",
    page_icon="✨",
    layout="wide"
)

# Custom CSS for a ChatGPT-like interface
st.markdown("""
<style>
.main {
    background-color: #f9f9f9;
}
.chat-message {
    padding: 1rem;
    margin-bottom: 1rem;
    line-height: 1.5;
}
.user-message {
    background-color: #f0f2f5;
    border-radius: 0.5rem;
}
.assistant-message {
    background-color: white;
    border-radius: 0.5rem;
}
.source-item {
    padding: 0.5rem;
    background-color: #f8f9fa;
    border-radius: 0.3rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid #a5b1c2;
}
.stButton>button {
    background-color: #2e86de;
    color: white;
    border-radius: 0.3rem;
    border: none;
    padding: 0.5rem 1rem;
}
.stButton>button:hover {
    background-color: #1c6dc9;
}
.chat-container {
    max-width: 800px;
    margin: 0 auto;
}
.sidebar-history {
    padding: 0.5rem;
}
.sidebar-history-item {
    padding: 0.5rem;
    margin-bottom: 0.5rem;
    border-radius: 0.3rem;
    background-color: #f0f2f5;
    cursor: pointer;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.sidebar-history-item:hover {
    background-color: #e4e6e9;
}
.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 1rem;
    background-color: white;
    border-top: 1px solid #eee;
    z-index: 100;
}
.chat-area {
    margin-bottom: 5rem;
    padding-bottom: 2rem;
}
.stExpander {
    border: none !important;
    box-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

# Cache the pipeline to avoid reloading on every interaction
@st.cache_resource(show_spinner=False)
def load_pipeline():
    return RAGPipeline()

# Force clear the cache
st.cache_resource.clear()

pipeline = load_pipeline()

# Chat history stored in session state
if "history" not in st.session_state:
    st.session_state.history = []  # list[tuple[str, str, list[str]]]

# Initialize current conversation if not exists
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = []

# Sidebar with history
with st.sidebar:
    st.title("CrediTrust Analyst ✨")
    
    if st.button("New Chat", key="new_chat"):
        st.session_state.current_conversation = []
        st.rerun()
        
    if st.button("Clear All History", key="clear_btn"):
    st.session_state.history = []
        st.session_state.current_conversation = []
        st.rerun()
    
    if st.session_state.history:
        st.markdown("### Previous Conversations")
        for i, (q, _, _) in enumerate(st.session_state.history):
            # Truncate long questions
            display_q = q if len(q) < 40 else q[:37] + "..."
            st.markdown(f"""
            <div class="sidebar-history-item" onclick="parent.postMessage({{command: 'streamlit:setComponentValue', componentValue: {i}, dataType: 'number'}}, '*')">
                {display_q}
            </div>
            """, unsafe_allow_html=True)

# Main chat container
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Display current conversation
st.markdown('<div class="chat-area">', unsafe_allow_html=True)
for q, a, srcs in st.session_state.current_conversation:
    # User message
    st.markdown(f"""
    <div class="chat-message user-message">
        <div>{q}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Assistant message
    st.markdown(f"""
    <div class="chat-message assistant-message">
        <div>{a}</div>
    """, unsafe_allow_html=True)
    
    # Sources in a collapsible section
    if srcs:
        with st.expander("View Sources"):
            for i, s in enumerate(srcs):
                st.markdown(f"""
                <div class="source-item">
                    <strong>Source {i+1}:</strong> {s}
                </div>
                """, unsafe_allow_html=True)
    
    # Close the assistant message div
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input form at the bottom
st.markdown('<div class="input-container">', unsafe_allow_html=True)
with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([6, 1])
    with cols[0]:
        user_query = st.text_input(
            "Question",  # Added a non-empty label
            placeholder="Ask a question about credit card complaints...",
            key="input",
            label_visibility="collapsed"  # Hide the label but keep it for accessibility
        )
    with cols[1]:
        submitted = st.form_submit_button("Send", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Handle form submission
if submitted and user_query.strip():
    # Add user message to conversation immediately
    if "temp_conversation" not in st.session_state:
        st.session_state.temp_conversation = []
    
    st.session_state.temp_conversation = st.session_state.current_conversation.copy()
    st.session_state.temp_conversation.append((user_query, "", []))
    
    # Display updated conversation with empty assistant response
    for i, (q, a, srcs) in enumerate(st.session_state.temp_conversation):
        if i == len(st.session_state.temp_conversation) - 1:
            # This is the latest message, show streaming response
            st.markdown(f"""
            <div class="chat-message user-message">
                <div>{q}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="chat-message assistant-message">', unsafe_allow_html=True)
            
            with st.spinner("Thinking..."):
        # Retrieve chunks
        chunks = pipeline.retriever.retrieve(user_query, k=5)
                
                # Stream generate response
        placeholder = st.empty()
        answer_acc = ""
                
        for delta in pipeline.generator.stream_generate(question=user_query, chunks=chunks):
            answer_acc += delta
            placeholder.markdown(answer_acc)
                
                # Prepare sources
                source_snippets = [f"{c.content}" for c in chunks[:3]]

                # Add to current conversation
                st.session_state.current_conversation.append((user_query, answer_acc, source_snippets))

                # Also add to history if not already there
                if not st.session_state.history or st.session_state.history[-1][0] != user_query:
                    st.session_state.history.append((user_query, answer_acc, source_snippets))
            
            # Sources
            if source_snippets:
                with st.expander("View Sources"):
                    for i, s in enumerate(source_snippets):
                        st.markdown(f"""
                        <div class="source-item">
                            <strong>Source {i+1}:</strong> {s}
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Clean up temporary conversation
    if "temp_conversation" in st.session_state:
        del st.session_state.temp_conversation
st.markdown('</div>', unsafe_allow_html=True)

