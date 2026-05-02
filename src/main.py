import discord
from discord.ext import commands
import logging
import asyncio
from src.config import Config
from src.agents.engineer import EngineerAgent
from src.agents.writer import WriterAgent
from src.agents.editor import EditorAgent
from src.agents.image_gen import ImageAgent
from src.database import Database

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolarBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        super().__init__(command_prefix="!", intents=intents)
        
        # Initialize agents
        self.engineer = EngineerAgent()
        self.writer = WriterAgent()
        self.editor = EditorAgent()
        self.image_gen = ImageAgent()
        
        # Initialize database
        self.db = Database()

    async def on_ready(self):
        print(f'Logged in as {self.user} (ID: {self.user.id})')
        print('------')

bot = SolarBot()

async def boss_approval(ctx, content):
    """Boss agent asks for approval before proceeding"""
    msg = await ctx.send(f'👨‍💼 **BOSS REVIEW**: Content ready. React with ✅ to approve or ❌ to reject.\n\n{content}')
    await msg.add_reaction('✅')
    await msg.add_reaction('❌')
    
    def check(reaction, user):
        return user == ctx.author and str(reaction.emoji) in ['✅', '❌'] and reaction.message.id == msg.id
    
    try:
        reaction, user = await bot.wait_for('reaction_add', timeout=300.0, check=check)
        if str(reaction.emoji) == '✅':
            return True
        else:
            await ctx.send('❌ Boss rejected. Content not posted.')
            return False
    except asyncio.TimeoutError:
        await ctx.send('⏱️ Approval timeout.')
        return False

@bot.command(name='solar')
async def solar(ctx, *, topic: str):
    """5-agent workflow: Boss → Engineer → Writer → Editor → Image Agent"""
    
    # Agent 1: BOSS - Initialize
    await ctx.send(f'👨‍💼 **BOSS INITIATED**: Processing "{topic}"...\n')
    
    try:
        # Agent 2: ENGINEER - Research
        await ctx.send(f'🔬 **ENGINEER**: Researching "{topic}"...')
        research_data = await bot.engineer.process(topic)
        if not research_data:
            await ctx.send("❌ Engineer failed to gather research.")
            return
            
        facts_display = "\n".join([f"- {fact}" for fact in research_data["key_facts"][:3]])
        await ctx.send(f'✅ Research complete. Key facts found:\n{facts_display}')
        
        # Agent 3: WRITER - Draft
        await ctx.send(f'✍️ **WRITER**: Drafting 500-word article...')
        article = await bot.writer.process(research_data)
        await ctx.send(f'✅ Draft complete:\n\n{article[:1500]}...') # Truncate if too long for Discord
        
        # Agent 4: EDITOR - Polish
        await ctx.send(f'🎨 **EDITOR**: Polishing for Facebook...')
        polished_post = await bot.editor.process(article)
        await ctx.send(f'✅ Final post ready:\n\n{polished_post[:1500]}...')
        
        # Boss Approval
        approved = await boss_approval(ctx, polished_post)
        if not approved:
            return
        
        # Agent 5: IMAGE AGENT - Generate prompt
        await ctx.send(f'🖼️ **IMAGE AGENT**: Creating image prompt...')
        image_prompt = await bot.image_gen.process(polished_post, topic)
        await ctx.send(f'✅ Image prompt ready:\n\n"{image_prompt}"')
        
        # Log to database
        bot.db.log_post(topic, polished_post, research_data)
        
        # Final output
        await ctx.send(f'🚀 **READY TO POST**:\n\n**Facebook Post:**\n{polished_post}')
        await ctx.send(f'**Image Prompt (for DALL-E):**\n```{image_prompt}```')

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        await ctx.send(f"❌ **Error during workflow:** {str(e)}")

if __name__ == "__main__":
    bot.run(Config.DISCORD_TOKEN)
