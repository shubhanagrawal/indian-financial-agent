from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Updated LangChain imports to resolve deprecation warnings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import traceback

def create_vector_store(company_name: str):
    """
    Chunks the processed text files, creates embeddings, and builds a FAISS vector store.

    Args:
        company_name (str): The name of the company folder (e.g., "RELIANCE_INDUSTRIES").
    """
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / f'data/processed/{company_name}'
    vector_store_path = project_root / f'data/vector_store/{company_name}'
    vector_store_path.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting to build vector store for {company_name} ---")
    print(f"Loading documents from: {processed_data_dir}")

    if not processed_data_dir.exists():
        print(f"Error: Processed data directory not found: {processed_data_dir}")
        return

    try:
        # --- FIX: Specify UTF-8 encoding for the TextLoader ---
        # This resolves the UnicodeDecodeError.
        loader_kwargs = {'encoding': 'utf-8'}
        
        # Load all .txt files from the processed data directory and its subdirectories
        loader = DirectoryLoader(
            str(processed_data_dir), 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            loader_kwargs=loader_kwargs, # Pass the encoding argument here
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()

        if not documents:
            print("[ERROR] No documents were loaded. Please check the processed data directory.")
            return
            
        print(f"Successfully loaded {len(documents)} documents.")

        # --- Chunk the documents ---
        print("Chunking documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split documents into {len(texts)} chunks.")

        # --- Create embeddings ---
        # We use a popular, high-quality open-source model.
        # The first time this runs, it will download the model (approx. 90MB).
        print("Initializing embedding model...")
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'} # Use CPU for broad compatibility
        )
        print("Embedding model loaded.")

        # --- Build and save the FAISS vector store ---
        print("Building FAISS vector store... (This may take a few minutes)")
        db = FAISS.from_documents(texts, embeddings)
        
        # Save the vector store to disk
        db.save_local(str(vector_store_path))
        print(f"[SUCCESS] Vector store created and saved at: {vector_store_path}")

    except Exception as e:
        print(f"[ERROR] An error occurred during vector store creation.")
        print(f"[ERROR] Details: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    target_company = "RELIANCE_INDUSTRIES"
    create_vector_store(target_company)

