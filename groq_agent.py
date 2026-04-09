import os
from duckduckgo_search import DDGS
from groq import Groq

def search_news():
    # Search for news about S&P500 using DuckDuckGo
    with DDGS() as ddgs:
        results = list(ddgs.text('S&P 500 market news', max_results=3))
    return '\n'.join([result['title'] + ': ' + result['href'] for result in results])

def write_article(news_context):
    # Write an article based on the search results
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    prompt = f'Act as a financial content creator and write an engaging Facebook post in Thai language summarizing the provided news: {news_context}. Keep it concise, no longer than 3 short paragraphs (approx 10-15 lines). Ensure the first sentence is a very catchy hook.'
    response = client.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}],
        model='llama-3.3-70b-versatile'
    )
    return response.choices[0].message.content

def review_post(draft):
    # Review and prepare the post
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    prompt = f'Act as an editor to review the Thai draft: {draft}. Fix typos, ensure a professional tone, and add relevant emojis and hashtags.'
    response = client.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}],
        model='llama-3.3-70b-versatile'
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    news_context = search_news()
    draft = write_article(news_context)
    final_post = review_post(draft)
    print(final_post)
