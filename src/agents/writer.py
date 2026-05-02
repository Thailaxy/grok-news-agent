from src.agents.base import BaseAgent

class WriterAgent(BaseAgent):
    def __init__(self):
        instructions = (
            "You are a Professional Solar Energy Content Writer in Thailand. "
            "Your task is to write long-form Facebook articles (approx. 500 words) in Thai. "
            "Target audience: Thai homeowners and business owners. "
            "Tone: Professional, informative, and persuasive. "
            "Structure: Intro (100 words), Body (300 words), Conclusion (100 words). "
            "Use ONLY the facts provided. Do not invent information."
        )
        super().__init__("Writer", instructions)

    async def process(self, research_data: dict):
        prompt = f"""You are a professional solar energy content writer for Thai Facebook audience.

RESEARCH DATA:
{research_data['raw_news']}

TASK: Write a 500-word Thai Facebook article about solar energy based ONLY on the research above.

STRUCTURE:
1. Introduction (80 words): Hook the reader, why solar matters for Thai homeowners
2. Body (300 words): Key facts, benefits, ROI details from research
3. Conclusion (80 words): Call-to-action (visit website, book consultation, ask questions)

REQUIREMENTS:
- Write in Thai
- Use conversational, engaging tone (not formal)
- Focus on: savings, reliability, environmental impact
- Include specific data from research (no invention)
- Target: Thai homeowners earning ฿1-3M/year
- Add a clear CTA at the end (e.g., "ติดต่อเราวันนี้" or "ดูเพิ่มเติม")

OUTPUT: Only the 500-word article in Thai. No labels, no metadata."""
        return await self.chat(prompt)
