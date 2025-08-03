import json
from pathlib import Path
import traceback

def parse_news_articles(company_name: str):
    """
    Parses the raw JSON news articles for a given company.
    It extracts the title and content from each article and saves it
    as a clean text file.

    Args:
        company_name (str): The name of the company folder (e.g., "RELIANCE_INDUSTRIES").
    """
    project_root = Path(__file__).resolve().parents[2]
    
    # Define Input and Output Directories
    raw_news_dir = project_root / f'data/raw/{company_name}/news_articles'
    processed_output_dir = project_root / f'data/processed/{company_name}/news_articles'
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Starting news article parsing for {company_name} ---")
    print(f"Input directory: {raw_news_dir}")

    if not raw_news_dir.exists():
        print(f"Error: Raw data directory not found: {raw_news_dir}")
        return

    # Process each JSON file in the directory
    for json_path in raw_news_dir.glob("*.json"):
        print(f"\n[INFO] Processing file: {json_path.name}...")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                article_data = json.load(f)

            # Extract the most important fields: title and content
            # The 'content' field from NewsAPI can sometimes be truncated,
            # but it's the best we have without full web scraping.
            title = article_data.get("title", "")
            content = article_data.get("content", "")
            
            # Combine title and content for the final text file
            full_text_content = f"{title}\n\n{content}"

            if not title and not content:
                print(f"[WARNING] No title or content found in {json_path.name}. Skipping.")
                continue

            # Save the processed content to a new text file
            output_filename = processed_output_dir / f"{json_path.stem}_processed.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(full_text_content)
            
            print(f"[SUCCESS] Successfully processed and saved to {output_filename}")

        except json.JSONDecodeError:
            print(f"[ERROR] Could not decode JSON from {json_path.name}. File might be corrupted.")
        except Exception as e:
            print(f"[ERROR] A critical error occurred while processing {json_path.name}.")
            print(f"[ERROR] Details: {e}")
            traceback.print_exc()

if __name__ == '__main__':
    target_company = "RELIANCE_INDUSTRIES"
    parse_news_articles(target_company)
    print("\n--- News article processing complete. ---")

