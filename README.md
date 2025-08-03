Financial Synthesis Agent 🤖
This project is a sophisticated, multi-stage AI agent designed to analyze and synthesize information from unstructured financial documents. It transforms dense annual reports, quarterly reports, and earnings call transcripts into a queryable and intelligent system, capable of performing tasks from simple Q&A to complex analytical summaries.

This repository documents the journey of building this agent, highlighting the real-world challenges and engineering solutions involved in making AI work with messy, domain-specific data.

Features
Robust Data Processing Pipeline: Ingests and parses complex PDF documents, handling scanned pages (OCR), tables, and inconsistent layouts using unstructured, Poppler, and Tesseract.

Intelligent RAG Pipeline:

Builds a semantic search index using FAISS and sentence-transformers for meaning-based information retrieval.

Implements an advanced Map-Reduce strategy to handle large contexts and avoid the limitations of smaller LLMs.

Analytical Modes: The agent can perform both general Q&A and more complex, structured analyses like generating a SWOT summary.

Interactive Web Interface: A user-friendly web app built with Streamlit allows for easy interaction with the agent.

Project Status
The foundational RAG pipeline is complete and functional. The agent can successfully analyze documents for a single company (Reliance Industries) and provide source-backed answers.

The next steps on the roadmap include:

Knowledge Graph Integration: Building a Neo4j Knowledge Graph to enable the agent to perform multi-hop reasoning and answer questions that require connecting information across documents.

Deployment: Deploying the Streamlit application to a cloud platform.

Scalability: Architecting an automated data ingestion pipeline to expand the agent's capabilities across a wider range of stocks.

Setup and Installation
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

2. Create a Python Environment
It is highly recommended to use a virtual environment.

# Create the environment
python -m venv rag

# Activate the environment
# On Windows:
rag\Scripts\activate
# On macOS/Linux:
source rag/bin/activate

3. Install System Dependencies
This project requires several system-level dependencies for PDF processing.

Poppler: Follow installation instructions for your OS.

Tesseract: Follow installation instructions for your OS.

4. Install Python Packages
Once the system dependencies are installed, install the required Python packages:

pip install -r requirements.txt

5. Set Up API Keys
Create a .env file inside the config/ directory.

Add your Groq API key to this file:

GROQ_API_KEY="gsk_..."

6. Gather Data
Due to their size, the raw data files are not included in this repository. You will need to manually download the Annual Reports, Quarterly Reports, and Earnings Call Transcripts for your target company and place them in the appropriate subdirectories within data/raw/COMPANY_NAME/.

How to Run
The project is broken down into a professional, decoupled pipeline. Run the scripts in the following order:

Step 1: Process Raw Documents
First, run the parsers to clean the raw data.

python src/processing/document_parser.py
python src/processing/transcript_parser.py
python src/processing/news_parser.py

Step 2: Build the Vector Store
This script will chunk the processed text and build the FAISS index.

python src/retrieval/build_vector_store.py

Step 3: Run the Web Application
Launch the Streamlit app to interact with the agent.

streamlit run app.py
