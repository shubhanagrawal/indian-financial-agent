from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import traceback

def parse_transcripts(company_name: str):
    """
    Parses PDF earnings call transcripts for a given company.
    Uses the 'fast' strategy for efficiency as transcripts are text-based.

    Args:
        company_name (str): The name of the company folder (e.g., "RELIANCE_INDUSTRIES").
    """
    project_root = Path(__file__).resolve().parents[2]
    
    # Define Input and Output Directories
    raw_transcripts_dir = project_root / f'data/raw/{company_name}/earnings_transcripts'
    processed_output_dir = project_root / f'data/processed/{company_name}/earnings_transcripts'
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting transcript parsing for {company_name} (fast strategy) ---")
    print(f"Input directory: {raw_transcripts_dir}")

    if not raw_transcripts_dir.exists():
        print(f"Error: Raw data directory not found: {raw_transcripts_dir}")
        return

    # Process each PDF in the directory
    for pdf_path in raw_transcripts_dir.glob("*.pdf"):
        print(f"\n[INFO] Processing file: {pdf_path.name}...")
        
        try:
            # The "fast" strategy is sufficient and quick for text-heavy transcripts.
            elements = partition_pdf(
                filename=str(pdf_path), 
                strategy="fast"
            )

            full_text_content = [str(el) for el in elements if str(el).strip()]

            if not full_text_content:
                print(f"[WARNING] No content extracted from {pdf_path.name}. Skipping.")
                continue

            # Save the processed content to a new text file
            output_filename = processed_output_dir / f"{pdf_path.stem}_processed.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write("\n\n".join(full_text_content))
            
            print(f"[SUCCESS] Successfully processed and saved to {output_filename}")

        except Exception as e:
            print(f"[ERROR] A critical error occurred while processing {pdf_path.name}.")
            print(f"[ERROR] Details: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    target_company = "RELIANCE_INDUSTRIES"
    parse_transcripts(target_company)
    print("\n--- Transcript processing complete. ---")
