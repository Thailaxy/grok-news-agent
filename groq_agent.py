import os
from duckduckgo_search import DDGS
import groq

def search_news():
    # Search for news about S&P500 using DuckDuckGo
    results = DDGS.search('S&P 500 market news', region='wt-wt', safesearch='Moderate', time='y', max_results=3)
    return '\n'.join([result.title + ': ' + result.url for result in results])

def write_article(news_context):
    # Write an article based on the search results
    client = groq.Client(os.environ.get('GROQ_API_KEY'))
    model = client.get_model('llama-3.3-70b-versatile')
    prompt = f'Act as a financial content creator and write an engaging Facebook post in Thai language summarizing the provided news: {news_context}. Keep it concise, no longer than 3 short paragraphs (approx 10-15 lines). Ensure the first sentence is a very catchy hook.'
    response = model.generate(prompt)
    return response.text

def review_post(draft):
    # Review and prepare the post
    client = groq.Client(os.environ.get('GROQ_API_KEY'))
    model = client.get_model('llama-3.3-70b-versatile')
    prompt = f'Act as an editor to review the Thai draft: {draft}. Fix typos, ensure a professional tone, and add relevant emojis and hashtags.'
    response = model.generate(prompt)
    return response.text

if __name__ == "__main__":
    news_context = search_news()
    draft = write_article(news_context)
    final_post = review_post(draft)
    print(final_post)
