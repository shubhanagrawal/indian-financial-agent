import os
from pathlib import Path
from unstructured.partition.pdf import partition_pdf
from unstructured_pytesseract import pytesseract
import traceback

def parse_pdf_documents(company_name: str, document_type: str):
    """
    Parses PDF documents using the 'hi_res' strategy.
    This version includes path verification and runtime PATH modification
    to provide the most robust solution for Windows/Anaconda environments.

    Args:
        company_name (str): The name of the company folder (e.g., "RELIANCE_INDUSTRIES").
        document_type (str): The type of document to parse ('annual_reports' or 'quarterly_reports').
    """
    # --- IMPORTANT: Set the paths to your Poppler and Tesseract installations ---
    poppler_bin_path = Path(r"D:\Poppler\poppler-24.08.0\Library\bin")
    tesseract_exe_path = Path(r"D:\Tesseract\tesseract.exe")

    # --- Step 1: Verify that the executables exist at the specified paths ---
    if not (poppler_bin_path / "pdfinfo.exe").is_file():
        print(f"[FATAL ERROR] Poppler executable not found at: {poppler_bin_path / 'pdfinfo.exe'}")
        print("Please double-check the 'poppler_bin_path' variable in the script.")
        return
        
    if not tesseract_exe_path.is_file():
        print(f"[FATAL ERROR] Tesseract executable not found at: {tesseract_exe_path}")
        print("Please double-check the 'tesseract_exe_path' variable in the script.")
        return
        
    print("[SUCCESS] Poppler and Tesseract paths verified.")

    # --- Step 2: Directly tell the libraries where to find their executables ---
    pytesseract.tesseract_cmd = str(tesseract_exe_path)

    # --- Step 3: Temporarily add Poppler to the PATH for this script's execution ---
    os.environ["PATH"] += os.pathsep + str(poppler_bin_path)

    project_root = Path(__file__).resolve().parents[2]
    
    raw_reports_dir = project_root / f'data/raw/{company_name}/{document_type}'
    processed_output_dir = project_root / f'data/processed/{company_name}/{document_type}'
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting parsing for {company_name} - {document_type} (hi_res strategy) ---")
    print(f"Input directory: {raw_reports_dir}")

    for pdf_path in raw_reports_dir.glob("*.pdf"):
        print(f"\n[INFO] Processing file: {pdf_path.name}...")
        
        try:
            # We no longer need the poppler_path argument, as it's now in the runtime PATH
            elements = partition_pdf(
                filename=str(pdf_path), 
                strategy="hi_res",
                infer_table_structure=True,
                extract_images_in_pdf=False
            )

            if not elements:
                print(f"[WARNING] No content extracted from {pdf_path.name}. Skipping.")
                continue

            full_text_content = [str(el) for el in elements if str(el).strip()]

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
    
    parse_pdf_documents(target_company, "annual_reports")
    parse_pdf_documents(target_company, "quarterly_reports")

    print("\n--- All PDF processing complete. ---")
