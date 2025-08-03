import os
import json
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path

def fetch_news_articles(company_name: str, num_articles: int = 100):
    """
    Fetches news articles for a specific company and saves them
    into a dedicated folder for that company.

    Args:
        company_name (str): The name of the company (e.g., "Reliance Industries").
        num_articles (int): The number of articles to fetch.
    """
    # Load environment variables from the .env file in the config directory
    # This path navigation goes up two levels from src/ingestion to the root, then into config
    env_path = Path(__file__).resolve().parents[2] / 'config/.env'
    load_dotenv(dotenv_path=env_path)
    api_key = os.getenv("NEWS_API_KEY")

    if not api_key:
        print(f"Error: NEWS_API_KEY not found in {env_path}")
        print("Please create a .env file in the 'config' directory and add your key.")
        return

    # The save path is dynamic based on the company name
    company_folder_name = company_name.upper().replace(" ", "_")
    save_dir = Path(__file__).resolve().parents[2] / f'data/raw/{company_folder_name}/news_articles'
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching up to {num_articles} articles for '{company_name}'...")
    print(f"Saving articles to: {save_dir}")

    # The API query is also dynamic
    # We look for articles in the last 30 days for relevancy
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    url = (f'https://newsapi.org/v2/everything?'
           f'q="{company_name}"&'
           f'from={from_date}&'
           f'sortBy=relevancy&'
           f'language=en&'
           f'apiKey={api_key}&'
           f'pageSize={num_articles}')

    try:
        response = requests.get(url)
        response.raise_for_status() # This will raise an error for bad status codes (4xx or 5xx)
        data = response.json()
        articles = data.get("articles", [])

        if not articles:
            print(f"No articles found for '{company_name}'. The API might be rate-limited or the query returned no results.")
            return

        for i, article in enumerate(articles):
            # Create a unique filename to avoid overwriting
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = save_dir / f"{timestamp}_article_{i+1}.json"
            
            # Save the article data as a JSON file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(article, f, ensure_ascii=False, indent=4)

        print(f"Successfully saved {len(articles)} articles for {company_name}.")

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the API request: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # This block allows you to run the script directly from the terminal
    # It will automatically target "Reliance Industries" as planned.
    target_company = "Reliance Industries"
    
    fetch_news_articles(company_name=target_company, num_articles=100)
    print("-" * 20)
    print("News fetching process complete.")
