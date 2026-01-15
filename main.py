import requests
from albert_model import _load_model
from albert_model import rank_results # Import the ranking function from the ALBERT model
from albert_history import main

# --- Placeholder API Keys ---
SERPAPI_KEY = "f76b30bd9e63785dfacca776677e15736a53b529aa3cd4eb4f772759cf16b669"
SERPAPI_KEY2 = "a282d6cfa46a8906c3f6f29f3e963d8b62e08342f3d8a7d601bb26f39980fecc"
NEWSAPI_KEY = "18717fb28f154119acb22ebefed88e97"

def albert_rank(query, results):
    """Placeholder for ALBERT ranking logic."""
    # Normally you’d import and call your ALBERT ranking model here.
    return results  # returning unmodified results for now

def Main():
    query = input("Enter your search query: ").strip()
    print(f"\n🔍 Searching for: {query}\n")

    if not query:
        print("No query entered. Exiting.")
        return

    results = []
    related_searches = []
    inline_images = []
    top_news = []
    summary_html = ""

    # --- Search via SerpAPI ---
    params = {"q": query, "engine": "duckduckgo", "api_key": SERPAPI_KEY}
    response = requests.get("https://serpapi.com/search.json", params=params)

    if response.status_code == 200:
        data = response.json()
        results = data.get("organic_results", [])
        related_searches = data.get("related_searches", [])[:5]
        inline_images = data.get("inline_images", [])[:7]
    elif 400 <= response.status_code < 500:
        params["api_key"] = SERPAPI_KEY2
        response = requests.get("https://serpapi.com/search.json", params=params)
        if response.status_code == 200:
            data = response.json()
            results = data.get("organic_results", [])
            related_searches = data.get("related_searches", [])[:5]
            inline_images = data.get("inline_images", [])[:5]
        else:
            results = [{"title": "Error", "snippet": f"Search failed: {response.status_code}"}]
    else:
        print(f"⚠️ Unexpected response from SerpAPI: {response.status_code}")

    # --- Fetch related news ---
    news_params = {
        "q": query,
        "sortBy": "publishedAt",
        "apiKey": NEWSAPI_KEY,
        "pageSize": 3,
        "language": "en"
    }
    news_response = requests.get("https://newsapi.org/v2/everything", params=news_params)
    if news_response.status_code == 200:
        news_data = news_response.json()
        top_news = news_data.get("articles", [])[:3]
    else:
        top_news = [{"title": "News unavailable", "description": "Couldn't fetch news."}]

    # --- Rank results using ALBERT (if available) ---
    if results:
        model = _load_model()
        # need to plug in model for history to encode past data
        history = main(query,model)
        results = rank_results(history, results)

    # --- Print Results ---
    print("\n📄 Search Results:")
    for i, r in enumerate(results, start=1):
        print(f"{i}. {r.get('title', 'No title')}")
        print(f"   {r.get('link', '')}")
        print(f"   {r.get('snippet', '')}\n")

    print("📎 Related Searches:", [rs.get("query", "") for rs in related_searches])
    print("🖼️ Inline Images:", [img.get("thumbnail", "") for img in inline_images])

    print("\n📰 Top News:")
    for article in top_news:
        print(f"• {article.get('title', 'No title')} - {article.get('source', {}).get('name', 'Unknown')}")
        print(f"  {article.get('description', '')}\n")

if __name__ == "__main__":
    main()
