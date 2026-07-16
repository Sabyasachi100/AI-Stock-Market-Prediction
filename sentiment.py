import os
from newsapi import NewsApiClient
from transformers import pipeline
from dotenv import load_dotenv

# Load .env
load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

newsapi = NewsApiClient(api_key=API_KEY)

classifier = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

articles = newsapi.get_everything(
    q="Apple",
    language="en",
    sort_by="publishedAt",
    page_size=5
)

print("\nLatest Financial News\n")

for article in articles["articles"]:

    title = article["title"]

    result = classifier(title)[0]

    print("-----------------------------------")
    print("News:", title)
    print("Sentiment:", result["label"])
    print("Confidence:", round(result["score"]*100,2),"%")