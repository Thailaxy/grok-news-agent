# Wanakorn News Agent (Multi-Agent AI)

A Discord bot powered by Groq (Llama 3.3) and DuckDuckGo Search that automates financial news research and Facebook post creation using 3 specialized agents.

## Features

* News Researcher: Searches for financial news using DuckDuckGo Search
* Content Writer (Thai): Writes a Thai Facebook post based on the research using Groq (Llama 3.3)
* Professional Editor: Reviews and polishes the post using Groq (Llama 3.3)

## Setup Instructions

1. Create a `.env` file with the following environment variables:
   - `GROQ_API_KEY`: Your Groq API key
   - `DISCORD_TOKEN`: Your Discord bot token
2. Install the required dependencies using `pip install -r requirements.txt`

## Usage

To use the bot, simply type `!news [topic]` in a Discord channel where the bot is present. Replace `[topic]` with the financial news topic you want to research.

Example: `!news Apple Stock`
