import streamlit as st
from pathlib import Path
import sys

# Add the 'src' and 'src/synthesis' directories to the Python path
# This is necessary to resolve nested imports when running from the root directory
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root / "src"))
# Adding the synthesis directory specifically to solve the 'prompts' import issue
sys.path.append(str(project_root / "src" / "synthesis"))


# Now we can import our agent creation function
from synthesis.agent import create_qa_agent

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Financial Synthesis Agent",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Financial Synthesis Agent for Reliance Industries")
st.markdown("""
Welcome! This agent can answer your questions about Reliance Industries based on its recent financial reports, earnings calls, and news articles. 
To perform a detailed analysis, use keywords like 'SWOT', 'strengths', or 'risks' in your query.
""")

# --- Model and Agent Loading ---
# Use Streamlit's caching to load the agent only once.
@st.cache_resource
def load_agent(use_swot):
    """Loads the appropriate QA agent based on the query type."""
    st.write("Loading agent and AI models... This may take a few moments.")
    agent = create_qa_agent("RELIANCE_INDUSTRIES", use_swot_prompt=use_swot)
    return agent

# --- User Interface ---
query = st.text_input("Enter your question or analysis request:", placeholder="e.g., What were the key financial highlights last quarter?")

if query:
    # Detect if the user wants a SWOT analysis
    swot_keywords = ['swot', 'strengths', 'weaknesses', 'opportunities', 'threats']
    is_swot_query = any(keyword in query.lower() for keyword in swot_keywords)
    
    # Load the correct agent (this will be cached)
    qa_agent = load_agent(is_swot_query)

    if qa_agent:
        with st.spinner("Thinking... (This may take a moment due to the advanced model)"):
            try:
                # Get the result from the agent
                result = qa_agent.invoke({"query": query})

                # Display the results
                st.subheader("Result")
                st.write(result['result'])

                st.subheader("Sources Used")
                # Display the source documents
                unique_sources = {source.metadata.get('source', 'Unknown source') for source in result['source_documents']}
                for source in sorted(list(unique_sources)):
                    st.info(f"- {Path(source).name}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Could not initialize the QA Agent. Please check the terminal for errors.")

