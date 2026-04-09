import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
from duckduckgo_search import DDGS
from groq import Groq

intents = discord.Intents.default()
intents.message_content = True

load_dotenv()
bot = commands.Bot(command_prefix='!', intents=intents)

import time
from duckduckgo_search.exceptions import RatelimitException

def search_news(topic):
    try:
        with DDGS() as ddgs:
            # Search for news specifically
            results = list(ddgs.news(topic, max_results=3))
    except RatelimitException:
        time.sleep(1)
        with DDGS() as ddgs:
            # Fallback to text search if news search fails due to rate limit
            results = list(ddgs.text(topic, max_results=3))
    
    formatted_news = ""
    for r in results:
        # Handle potential missing keys gracefully
        title = r.get('title', 'No Title')
        body = r.get('body', '')
        formatted_news += f"Headline: {title}\nSummary: {body}\n\n"
    
    return formatted_news if formatted_news.strip() else "No data found"

def write_article(news_context):
    # Write an article based on the search results
    load_dotenv()
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    prompt = f'You are a financial news summarizer. Strictly summarize ONLY the provided news context below. Translate and summarize this specific context into an engaging Thai Facebook post (max 3 short paragraphs). Do NOT invent general financial advice. Do NOT talk about the Thai economy unless it is in the context. Context: {news_context}'
    response = client.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}],
        model='llama-3.3-70b-versatile'
    )
    return response.choices[0].message.content

def review_post(draft):
    # Review and prepare the post
    load_dotenv()
    client = Groq(api_key=os.environ.get('GROQ_API_KEY'))
    prompt = f'You are a strict editor. Review this Thai Facebook post draft: {draft}. Fix any typos, improve the flow, and add relevant emojis and hashtags like #SP500. IMPORTANT: Output ONLY the final Thai text ready for posting. Do NOT include any conversational filler, introductory phrases, or notes at the end.'
    response = client.chat.completions.create(
        messages=[{'role': 'user', 'content': prompt}],
        model='llama-3.3-70b-versatile'
    )
    return response.choices[0].message.content

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

@bot.command(name='news')
async def news(ctx, topic: str):
    news_context = search_news(topic)
    await ctx.send(f'🔍 Searching for news about {topic}...')
    await ctx.send(news_context)
    draft = write_article(news_context)
    await ctx.send(f'✍️ Drafted article based on the research...')
    await ctx.send(draft)
    final_post = review_post(draft)
    await ctx.send(f'✅ Finalized and Polished Post:')
    await ctx.send(final_post)

load_dotenv()
bot.run(os.environ.get('DISCORD_TOKEN'))
