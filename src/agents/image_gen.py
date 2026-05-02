from src.agents.base import BaseAgent

class ImageAgent(BaseAgent):
    def __init__(self) -> None:
        instructions = (
            "You are a Creative Visual Prompt Engineer. "
            "Based on a solar energy article, you create detailed, high-quality prompts "
            "for AI image generators like DALL-E or Midjourney. "
            "The style should be modern, clean, and professional, with a Thai context."
        )
        super().__init__("ImageGen", instructions)

    async def process(self, article_text: str, topic: str) -> str:
        prompt = f"""You are a visual content specialist for solar energy marketing.

ARTICLE:
{article_text}

TOPIC: {topic}

TASK: Create a detailed DALL-E image generation prompt based on this article.

REQUIREMENTS:
- Include key visual elements (solar panels, Thai house, ROI chart, sun, etc.)
- Thai context (Thai architecture, Thai landscape if applicable)
- Professional, modern style
- High quality, realistic, clean aesthetic
- Include relevant colors (gold, green, blue, white)
- 50-100 words

FORMAT: Just the image prompt, ready to send to DALL-E.
Example: "Professional illustration of a Thai house with solar panels on roof during sunset, showing electricity cost comparison chart, modern flat design, high quality, 4k"

OUTPUT: Only the prompt. No other text."""
        return await self.chat(prompt)
