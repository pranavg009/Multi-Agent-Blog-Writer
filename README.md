# Multi-Agent Blog Writer
 
A 3-agent AI pipeline that takes a topic and returns a finished, publication-ready blog post with sources, SEO metadata, and configurable tone. No back-and-forth. One click.
 
**Live demo:** [multi-agent-blog-writer-01.streamlit.app](https://multi-agent-blog-writer-01.streamlit.app/)
 
---
 
## What it actually does
 
Most AI writing tools give you a draft and leave the rest to you. This one doesn't stop there.
 
Three agents run in sequence, each one picking up where the last left off:
 
1. **Researcher** runs 3 separate Tavily web searches on different angles of your topic (facts/stats, recent news, expert views). Returns structured notes with source URLs. It never makes stats up if it can't find a number in the search results, it doesn't include one.
 
2. **Writer** receives those research notes as context and writes a full blog post in your chosen tone and word count. Every statistic in the post traces back to a real URL from step 1.
 
3. **Editor** polishes the draft, fixes grammar and flow, and appends SEO metadata: a title under 60 characters, a meta description under 160, and 5 tags.
 
The whole thing runs in 4–8 minutes.
 
## Tech stack
 
| Layer | What |
|---|---|
| Agent framework | CrewAI 1.9+ |
| Primary LLM | Claude Haiku |
| Fallback LLM | Gemini 2.5 Flash |
| Search | Tavily built for AI agents, Does not give web scraped results |
| UI | Streamlit |
| Deployment | Streamlit Cloud |
 
---
 
## Run it locally
 
```bash
git clone https://github.com/YOUR_USERNAME/multi-agent-blog-writer
cd multi-agent-blog-writer
pip install -r requirements.txt
```
 
Set your API keys
 
Then run:
 
```bash
streamlit run app.py
```
 
You need at least one LLM key (Anthropic or Gemini) and the Tavily key. If Anthropic fails or isn't set, the app automatically falls back to Gemini.
 
---
 
 6. Deploy
   
## How the agents hand off context
 
CrewAI passes task outputs as context to the next agent. The Writer never touches the web it only sees what the Researcher found. The Editor only sees the Writer's draft. This keeps each agent focused and means the final post is grounded in real search data rather than whatever the LLM remembers from training.
 
```
User input (topic, tone, word count)
        │
        ▼
┌─────────────────┐
│   Researcher    │  3x Tavily searches → structured notes + URLs
└────────┬────────┘
         │ context
         ▼
┌─────────────────┐
│     Writer      │  notes → full Markdown blog post
└────────┬────────┘
         │ context
         ▼
┌─────────────────┐
│     Editor      │  draft → polished post + SEO metadata
└─────────────────┘
         │
         ▼
  Streamlit UI output
  (rendered Markdown + download button)
```
 
---
 
## What you can configure
 
From the sidebar before generating:
 
- **Tone** Professional, Casual, Technical, or Academic
- **Target word count** 300 to 2000 words
- **Topic** 6 suggested topics, or enter your own
 
The app also keeps a history of your last 5 generated posts in the session.
