import json
from duckduckgo_search import DDGS
from src.agents.base import BaseAgent

class EngineerAgent(BaseAgent):
    def __init__(self):
        instructions = (
            "You are a Solar Energy Research Engineer. Your goal is to find accurate, "
            "technical, and up-to-date information about solar energy in Thailand. "
            "You will be provided with search results. Your task is to extract key facts, "
            "data points (ROI, rates, etc.), and sources, then format them into a clean JSON object."
        )
        super().__init__("Engineer", instructions)

    def search(self, topic: str):
        with DDGS() as ddgs:
            results = list(ddgs.text(f"solar energy {topic} Thailand", max_results=5))
        return results

    async def process(self, topic: str):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.news(topic, max_results=5))
        except Exception:
            # Fallback to text search
            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=5))
        
        research_data = {
            'topic': topic,
            'sources': [],
            'key_facts': [],
            'raw_news': ""
        }
        
        for i, r in enumerate(results, 1):
            title = r.get('title', 'No Title')
            body = r.get('body', '')
            url = r.get('url', 'No URL')
            
            research_data['sources'].append(url)
            research_data['key_facts'].append(title)
            research_data['raw_news'] += f"{i}. {title}\n{body}\n\n"
        
        return research_data
