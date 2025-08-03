import os
import re
import yaml
import json
from pathlib import Path
import time
import concurrent.futures
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --- Load Configuration ---
config_path = Path(__file__).resolve().parents[2] / 'config/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# --- Load Groq API Key from .env ---
env_path = Path(__file__).resolve().parents[2] / 'config/.env'
load_dotenv(dotenv_path=env_path)
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Initialize the Groq LLM ---
try:
    llm = ChatGroq(
        temperature=0, 
        model_name=config['llm']['model_name'],
        groq_api_key=groq_api_key
    )
    print(f"[SUCCESS] Groq client initialized for model: {config['llm']['model_name']}")
except Exception as e:
    print(f"[FATAL ERROR] Could not initialize Groq client. Check your API key. Error: {e}")
    llm = None

# --- Prompt for Knowledge Graph Extraction ---
graph_extraction_prompt = PromptTemplate(
    template="""
You are an expert data analyst. Your task is to extract structured information from the text below.
Identify key entities and their relationships as a list of triples.
Present your findings as a list of triples. Each triple must be on a new line and in the format: (Subject, Relationship, Object).
**CRITICAL:** Adhere strictly to this format. Do not add any text before or after the list of triples.
---
Text: {text}
Output:
""",
    input_variables=["text"],
)

def load_processed_files_log(log_path: Path) -> set:
    if not log_path.exists():
        return set()
    with open(log_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)

def log_processed_file(log_path: Path, filename: str):
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(f"{filename}\n")

def save_progress(output_file: Path, triples: list):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(triples, f, indent=2)

def process_chunk(chunk: str) -> list:
    """Processes a single chunk of text to extract triples."""
    if not llm:
        return []
    try:
        chain = LLMChain(llm=llm, prompt=graph_extraction_prompt)
        result = chain.invoke({"text": chunk})
        extracted_text = result.get('text', '')
        triples = re.findall(r'[\[\(](.*?),\s*\'(.*?)\',\s*(.*?)[\]\)]', extracted_text)
        return triples
    except Exception as e:
        # This will catch rate limit errors for a single chunk
        print(f"[WARNING] A chunk failed to process, likely due to rate limits. Error: {e}")
        # Re-raise the exception to be caught by the main loop
        raise e

def extract_triples_from_files():
    """
    Reads processed text files, extracts knowledge triples in parallel using the Groq API,
    and saves them to an intermediate JSON file. This version is resumable.
    """
    if not llm:
        print("LLM not initialized. Aborting.")
        return

    company_name = config['processing']['company_name']
    project_root = Path(__file__).resolve().parents[2]
    processed_data_dir = project_root / f'data/processed/{company_name}'
    output_file = project_root / f'data/processed/{company_name}_extracted_triples.json'
    log_file = project_root / f'data/processed/{company_name}_processed_files.log'

    processed_files = load_processed_files_log(log_file)
    all_triples = []
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            try:
                all_triples = json.load(f)
            except json.JSONDecodeError:
                all_triples = []
    
    print(f"--- Starting PARALLEL Triple Extraction for {company_name} using Groq ---")
    print(f"Found {len(processed_files)} previously processed files. They will be skipped.")
    print(f"Loaded {len(all_triples)} existing triples.")
    
    dirs_to_process = config['processing']['dirs_to_process']
    stop_processing = False

    for subdir in dirs_to_process:
        if stop_processing: break
        subdir_path = processed_data_dir / subdir
        if not subdir_path.exists():
            continue
            
        print(f"\n--- Processing directory: {subdir} ---")
        for txt_path in sorted(list(subdir_path.glob("*.txt"))):
            if stop_processing: break
            
            if txt_path.name in processed_files:
                print(f"\n[INFO] Skipping already processed file: {txt_path.name}")
                continue

            print(f"\n[INFO] Processing file: {txt_path.name}...")
            triples_from_this_file = []
            try:
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                chunks = [text_content[i:i+config['processing']['chunk_size']] for i in range(0, len(text_content), config['processing']['chunk_size'])]
                chunks = [c for c in chunks if c.strip()]

                # --- OPTIMIZATION: Process chunks in parallel ---
                # We set max_workers to a reasonable number to avoid overwhelming the API
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        try:
                            result_triples = future.result()
                            if result_triples:
                                triples_from_this_file.extend(result_triples)
                                print(f"[SUCCESS] Extracted {len(result_triples)} triples from a chunk.")
                        except Exception as e:
                            # If any chunk fails (e.g., rate limit), we stop processing this file
                            raise e

                all_triples.extend(triples_from_this_file)
                save_progress(output_file, all_triples)
                log_processed_file(log_file, txt_path.name)
                print(f"[SAVE] Successfully processed {len(triples_from_this_file)} triples from {txt_path.name} and saved progress.")

            except Exception as e:
                print(f"[ERROR] An API or other error occurred while processing {txt_path.name}: {e}")
                print("Stopping extraction. Will resume from this file on the next run.")
                stop_processing = True
                break

    print(f"\n--- Triple Extraction Run Complete ---")
    print(f"Total triples extracted so far: {len(all_triples)}.")
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    extract_triples_from_files()
