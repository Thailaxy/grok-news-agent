import os
from duckduckgo_search import DDGS
from groq import Groq

def search_news():
    with DDGS() as ddgs:
        # Search for news specifically
        results = list(ddgs.news('S&P 500 market', max_results=3))
    
    formatted_news = ""
    for r in results:
        # Handle potential missing keys gracefully
        title = r.get('title', 'No Title')
        body = r.get('body', '')
        formatted_news += f"Headline: {title}\nSummary: {body}\n\n"
    
    return formatted_news if formatted_news.strip() else "No news found today."

def write_article(news_context):
    # Write an article based on the search results
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    prompt = f'You are a financial news summarizer. Strictly summarize ONLY the provided news context below. The news is about the S&P 500. Translate and summarize this specific context into an engaging Thai Facebook post (max 3 short paragraphs). Do NOT invent general financial advice. Do NOT talk about the Thai economy unless it is in the context. Context: {news_context}'
    response = client.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}],
        model='llama-3.3-70b-versatile'
    )
    return response.choices[0].message.content

def review_post(draft):
    # Review and prepare the post
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    prompt = f'You are a strict editor. Review this Thai Facebook post draft: {draft}. Fix any typos, improve the flow, and add relevant emojis and hashtags like #SP500. IMPORTANT: Output ONLY the final Thai text ready for posting. Do NOT include any conversational filler, introductory phrases, or notes at the end.'
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
