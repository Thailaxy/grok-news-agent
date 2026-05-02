import google.generativeai as genai
from groq import Groq
from src.config import Config

class BaseAgent:
    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions
        
        if Config.DEFAULT_LLM == "gemini":
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=Config.GEMINI_MODEL,
                system_instruction=instructions
            )
        else:
            self.client = Groq(api_key=Config.GROQ_API_KEY)

    async def chat(self, prompt: str) -> str:
        if Config.DEFAULT_LLM == "gemini":
            response = await self.model.generate_content_async(prompt)
            return response.text
        else:
            # Note: Groq doesn't have a native async client in the standard way for this example, 
            # but we can wrap it or use their async support if available.
            # For simplicity, we'll use synchronous call for now or assume a wrapper.
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt}
                ],
                model=Config.GROQ_MODEL
            )
            return response.choices[0].message.content
