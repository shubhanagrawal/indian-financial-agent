

````markdown
# 📊 Financial Synthesis Agent 🤖

**Financial Synthesis Agent** is a sophisticated, multi-stage AI system built to analyze and synthesize insights from unstructured financial documents such as annual reports, quarterly filings, and earnings call transcripts.

It transforms dense, domain-specific text into a **queryable, intelligent assistant** capable of handling both general Q&A and advanced analytical tasks such as SWOT analysis. This repository documents the real-world engineering decisions and challenges encountered while building such an AI system.


## 🚀 Features

### 🔁 Robust Data Processing Pipeline
- Ingests and parses complex PDF documents
- Handles:
  - Scanned pages using **Tesseract OCR**
  - Tabular data and inconsistent layouts using **Unstructured** and **Poppler**

### 🧠 Intelligent Retrieval-Augmented Generation (RAG) Pipeline
- Semantic search powered by **FAISS** and **Sentence Transformers**
- **Map-Reduce** summarization strategy for managing long-context documents
- Modular design for scalability and adaptability

### 📊 Analytical Modes
- Supports:
  - Natural language Q&A from financial documents
  - Structured insights such as **SWOT summaries**

### 💬 Interactive Web Interface
- Built with **Streamlit** for an intuitive and accessible user experience

---

## 📌 Project Status

✅ The foundational **RAG pipeline** is fully implemented and operational  
✅ The system can process and respond to questions based on documents from **Reliance Industries**  
✅ Answers are **source-backed** with semantic retrieval  

### 🔮 Next Steps
- 🔗 **Knowledge Graph Integration** using **Neo4j** for multi-hop reasoning
- ☁️ **Deployment** of the Streamlit app to a cloud platform
- 📈 **Scalability** through an automated ingestion pipeline for multi-company document handling

---

## 🛠️ Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
````

### 2. Create and Activate a Python Environment

```bash
# Create virtual environment
python -m venv rag

# Activate environment
# On Windows:
rag\Scripts\activate

# On macOS/Linux:
source rag/bin/activate
```

### 3. Install System Dependencies

* **Poppler** → [Installation Guide](https://poppler.freedesktop.org/)
* **Tesseract OCR** → [Installation Guide](https://github.com/tesseract-ocr/tesseract)

Make sure both tools are available in your system's PATH.

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Set Up API Keys

Create a `.env` file inside the `config/` directory with the following content:

```env
GROQ_API_KEY="gsk_..."
```

### 6. Gather Financial Data

Download and place the following documents in the appropriate subdirectory:

```
data/raw/COMPANY_NAME/
├── Annual_Reports/
├── Quarterly_Reports/
└── Earnings_Call_Transcripts/
```

---

## ▶️ How to Run the Pipeline

### ✅ Step 1: Process Raw Documents

```bash
python src/processing/document_parser.py
python src/processing/transcript_parser.py
python src/processing/news_parser.py
```

### ✅ Step 2: Build Vector Store

```bash
python src/retrieval/build_vector_store.py
```

### ✅ Step 3: Launch the Web App

```bash
streamlit run app.py
```

This will open an interactive Streamlit interface where you can upload queries and receive source-backed answers and summaries.

---


---

## 🧠 Tech Stack

* **Python**
* **Tesseract OCR** for scanned documents
* **Poppler + Unstructured** for PDF parsing
* **FAISS + Sentence Transformers** for semantic search
* **LangChain / Groq API** for LLM inference
* **Streamlit** for UI
* *(Coming Soon)* **Neo4j** for Knowledge Graph reasoning

---

## 👥 Contributing

Contributions are welcome!
If you have suggestions, find bugs, or want to extend the system, feel free to:

* Fork the repo
* Create a branch
* Submit a Pull Request

For major changes, please open an issue first to discuss your idea.

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📬 Contact

For questions or collaborations, please reach out via [GitHub Issues](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME/issues) or connect on [LinkedIn](https://www.linkedin.com/in/YOUR_PROFILE/).

---

```

---


