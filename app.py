import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from tavily import TavilyClient
from pydantic import BaseModel, Field
from typing import Type
import anthropic as ant
import google.generativeai as genai
import time, re, os, json
from datetime import datetime

st.set_page_config(
    page_title="Multi-Agent Blog Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── LLM Setup — Claude primary, Gemini fallback ─────────────
# NOTE: no @st.cache_resource — avoids caching None on cold start
def get_llm():
    try:
        openrouter_key = st.secrets.get("OPENROUTER_API_KEY", "")
    except:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY", "")

    if openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        return (
            LLM(
                model="openrouter/meta-llama/llama-3.1-8b-instruct:free",
                api_key=openrouter_key,
                api_base="https://openrouter.ai/api/v1",
                max_tokens=4096,
                temperature=0.7
            ),
            "OpenRouter (Llama 3.1 8B Free)"
        )
    return None, "None"
    
# ── Tavily Search Tool ───────────────────────────────────────
class SearchInput(BaseModel):
    query: str = Field(description="Search query")

class TavilySearchTool(BaseTool):
    name: str = "web_search"
    description: str = "Search the internet for facts and sources."
    args_schema: Type[BaseModel] = SearchInput
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, query: str) -> str:
        api_key = os.environ.get("TAVILY_API_KEY", "")
        if not api_key:
            return "TAVILY_API_KEY not set in Streamlit secrets."
        for attempt in range(3):
            try:
                client = TavilyClient(api_key=api_key)
                response = client.search(
                    query=query, search_depth="advanced",
                    max_results=6, include_answer=True
                )
                out = [f"Results for '{query}':\n"]
                if response.get("answer"):
                    out.append(f"Quick answer: {response['answer']}\n")
                for i, r in enumerate(response.get("results", []), 1):
                    out.append(
                        f"{i}. {r.get('title','')}\n"
                        f"   {r.get('url','')}\n"
                        f"   {r.get('content','')[:400]}\n"
                    )
                return "\n".join(out)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return f"Search failed: {e}"
        return "Search unavailable."

search_tool = TavilySearchTool()

# ── Output Parser ────────────────────────────────────────────
def parse_output(raw: str) -> dict:
    res = {"blog_post": "", "seo_title": "", "meta_desc": "", "tags": []}
    for field, marker in [("seo_title", "SEO_TITLE"), ("meta_desc", "META_DESC")]:
        m = re.search(rf'\[{marker}\](.*?)(?:\n|\[|$)', raw, re.IGNORECASE | re.DOTALL)
        if m:
            res[field] = m.group(1).strip()[:160]
    m = re.search(r'\[TAGS\](.*?)(?:\n|\[|$)', raw, re.IGNORECASE)
    if m:
        res["tags"] = [t.strip() for t in m.group(1).split(",") if t.strip()][:5]
    end = len(raw)
    for mk in ["[SEO_TITLE]", "[META_DESC]", "[TAGS]"]:
        idx = raw.upper().find(mk)
        if idx != -1 and idx < end:
            end = idx
    res["blog_post"] = raw[:end].strip()
    if not res["seo_title"]:
        res["blog_post"] = raw.strip()
        res["seo_title"] = "Generated Post"
        res["meta_desc"] = "AI blog"
        res["tags"] = ["ai", "blog", "content"]
    return res

def count_words(text: str) -> int:
    return len(re.sub(r'[#*`_\[\]()]', '', text).split())

# ── Crew Factory ─────────────────────────────────────────────
def run_crew_pipeline(topic, tone, word_count, llm):
    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research '{topic}'. Do 3 searches on different angles. "
            "Return structured notes with facts, stats, and source URLs. "
            "Never fabricate data."
        ),
        backstory="Meticulous analyst, 10 years investigative journalism.",
        tools=[search_tool], llm=llm, verbose=False,
        allow_delegation=False, max_iter=5
    )
    writer = Agent(
        role="Professional Blog Writer",
        goal=(
            f"Write a {word_count}-word {tone} blog about '{topic}' using "
            "ONLY the research notes. H1 title, hook intro, 3-5 H2 sections, "
            "CTA conclusion. No hallucination."
        ),
        backstory="Senior writer, TechCrunch and Wired. Data-driven storyteller.",
        tools=[], llm=llm, verbose=False, allow_delegation=False, max_iter=3
    )
    editor = Agent(
        role="Senior Editor and SEO Specialist",
        goal=(
            f"Polish blog. Ensure {tone} tone. Fix grammar and flow. "
            "Append at end:\n"
            "[SEO_TITLE] under 60 chars\n"
            "[META_DESC] under 160 chars\n"
            "[TAGS] exactly 5 comma-separated tags"
        ),
        backstory="15-year veteran editor. Expert in SEO and readability.",
        tools=[], llm=llm, verbose=False, allow_delegation=False, max_iter=3
    )
    t1 = Task(
        description=(
            f"Research '{topic}'. Search 3 angles:\n"
            f"1. '{topic} facts statistics 2025'\n"
            f"2. '{topic} latest developments'\n"
            f"3. '{topic} expert views challenges'\n"
            "Return: Key Facts, Developments, Expert Views, Sources."
        ),
        expected_output="Structured research notes with facts and URLs.",
        agent=researcher
    )
    t2 = Task(
        description=(
            f"Write a {word_count}-word {tone} blog about '{topic}'. "
            "Use ONLY research notes. Markdown H1/H2 headings. "
            "Min 3 statistics from research."
        ),
        expected_output=f"Complete blog post ~{word_count} words in Markdown.",
        agent=writer, context=[t1]
    )
    t3 = Task(
        description=(
            f"Polish the blog. Ensure {tone} tone. "
            "Append at end:\n"
            "[SEO_TITLE] ...\n[META_DESC] ...\n[TAGS] ..., ..., ..., ..., ..."
        ),
        expected_output="Polished post + SEO markers.",
        agent=editor, context=[t2]
    )
    crew = Crew(
        agents=[researcher, writer, editor],
        tasks=[t1, t2, t3],
        process=Process.sequential,
        verbose=False, memory=False, max_rpm=8
    )
    result = crew.kickoff()
    research = ""
    try:
        research = str(t1.output.raw if hasattr(t1.output, "raw") else t1.output)
    except Exception:
        pass
    return str(result), research

# ── Suggested topics ─────────────────────────────────────────
SUGGESTED_TOPICS = [
    "How AI agents are changing software development in 2025",
    "The future of remote work: trends shaping 2025 and beyond",
    "Quantum computing: what businesses need to know now",
    "Sustainable tech: how green AI is reducing carbon footprints",
    "Cybersecurity threats every startup should prepare for",
    "The rise of no-code tools and what it means for developers",
]

# ── Session state ────────────────────────────────────────────
if "history"  not in st.session_state: st.session_state.history  = []
if "result"   not in st.session_state: st.session_state.result   = None
if "running"  not in st.session_state: st.session_state.running  = False
if "topic_in" not in st.session_state: st.session_state.topic_in = ""

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("✍️ Blog Writer")
    llm, provider = get_llm()
    if llm:
        st.success(f"✅ {provider} connected")
    else:
        st.error("❌ No API key found.")
        st.info("Add to Streamlit secrets:\nANTHROPIC_API_KEY\nor\nGEMINI_API_KEY\nand\nTAVILY_API_KEY")
    st.divider()

    st.subheader("⚙️ Settings")
    tone       = st.selectbox("Tone", ["Professional", "Casual", "Technical", "Academic"])
    word_count = st.slider("Target words", 300, 2000, 800, step=100)
    st.info("⏱ Estimated: 4–8 min\n(3 agents in sequence)")
    st.divider()

    st.subheader("📂 History")
    if st.session_state.history:
        for h in reversed(st.session_state.history[-5:]):
            label = h["topic"][:28] + ("..." if len(h["topic"]) > 28 else "")
            if st.button(f"📄 {label}", key=h["id"], use_container_width=True):
                st.session_state.result = h["result"]
                st.rerun()
    else:
        st.caption("Generated posts appear here")
    st.divider()

    st.subheader("🤖 Pipeline")
    st.markdown("🔬 **Researcher** — Tavily web search")
    st.markdown("✍️ **Writer** — drafts blog post")
    st.markdown("📝 **Editor** — polish + SEO metadata")

# ── Main ─────────────────────────────────────────────────────
st.title("✍️ Multi-Agent Blog Writer")
st.caption("Researcher → Writer → Editor · CrewAI + Tavily")

st.markdown("**💡 Try a suggested topic:**")
cols = st.columns(3)
for i, ts in enumerate(SUGGESTED_TOPICS):
    with cols[i % 3]:
        label = ts[:35] + "..." if len(ts) > 35 else ts
        if st.button(label, key=f"s{i}", use_container_width=True,
                     disabled=st.session_state.running):
            st.session_state.topic_in = ts
            st.rerun()

topic = st.text_input(
    "Or enter your own topic:",
    value=st.session_state.topic_in,
    placeholder="e.g. How quantum computing will reshape cybersecurity",
    disabled=st.session_state.running
)

m1, m2, m3 = st.columns(3)
m1.metric("Tone", tone)
m2.metric("Target Words", word_count)
m3.metric("Agents", "3")

generate_btn = st.button(
    "🚀 Generate Blog Post",
    disabled=st.session_state.running or not llm or not topic.strip(),
    type="primary",
    use_container_width=True
)
if not llm:
    st.warning("Add ANTHROPIC_API_KEY or GEMINI_API_KEY in Streamlit secrets.")

# ── Generation ───────────────────────────────────────────────
if generate_btn and topic.strip() and llm:
    st.session_state.running = True
    st.session_state.topic_in = topic
    st.divider()
    st.subheader("⚡ Pipeline Running")

    s1, s2, s3 = st.columns(3)
    with s1: step1 = st.empty(); step1.info("🔬 **Step 1**\nResearcher\nSearching...")
    with s2: step2 = st.empty(); step2.warning("✍️ **Step 2**\nWriter\nWaiting...")
    with s3: step3 = st.empty(); step3.warning("📝 **Step 3**\nEditor\nWaiting...")

    status  = st.empty()
    prog    = st.progress(0)
    timer_d = st.empty()
    status.info("🔬 Researcher searching the web... (1/3)")
    prog.progress(5)
    start_t = time.time()

    try:
        raw, research_notes = run_crew_pipeline(topic, tone, word_count, llm)
        elapsed = round(time.time() - start_t, 1)

        step1.success("🔬 **Step 1**\nResearcher\n✅ Done")
        step2.success("✍️ **Step 2**\nWriter\n✅ Done")
        step3.success("📝 **Step 3**\nEditor\n✅ Done")
        prog.progress(100)
        timer_d.success(f"✅ Completed in {elapsed}s")
        status.empty()

        parsed = parse_output(raw)
        parsed["topic"]             = topic
        parsed["tone"]              = tone
        parsed["word_count_target"] = word_count
        parsed["word_count_actual"] = count_words(parsed["blog_post"])
        parsed["research_notes"]    = research_notes
        parsed["time_taken"]        = elapsed

        st.session_state.history.append({
            "id": str(time.time()), "topic": topic,
            "result": parsed, "time": datetime.now().strftime("%H:%M")
        })
        st.session_state.result  = parsed
        st.session_state.running = False
        st.rerun()

    except Exception as e:
        status.error(f"❌ Error: {str(e)[:200]}")
        step1.error("🔬 Failed"); step2.error("✍️ —"); step3.error("📝 —")
        st.session_state.running = False

# ── Display Result ───────────────────────────────────────────
if st.session_state.result:
    res = st.session_state.result
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words Written",  res.get("word_count_actual", "—"))
    c2.metric("Target Words",   res.get("word_count_target", "—"))
    c3.metric("Tags",           len(res.get("tags", [])))
    c4.metric("Time",           f"{res.get('time_taken', '—')}s")

    st.subheader("📄 Generated Blog Post")

    dl_col, _ = st.columns([1, 3])
    with dl_col:
        full_md = (
            res["blog_post"] +
            f"\n\n---\n**SEO Title:** {res['seo_title']}\n"
            f"**Meta Desc:** {res['meta_desc']}\n"
            f"**Tags:** {', '.join(res['tags'])}\n"
        )
        st.download_button(
            "⬇️ Download .md", full_md,
            file_name="blog_post.md", mime="text/markdown",
            use_container_width=True
        )

    with st.expander("📋 Copy raw Markdown"):
        st.code(res["blog_post"], language="markdown")

    st.markdown(res["blog_post"])
    st.divider()

    with st.expander("🔍 SEO Metadata", expanded=True):
        st.markdown(f"**SEO Title:** `{res['seo_title']}`")
        st.markdown(f"**Meta Description:** {res['meta_desc']}")
        if res.get("tags"):
            st.markdown("**Tags:** " + "  ".join([f"`{t}`" for t in res["tags"]]))

    if res.get("research_notes"):
        with st.expander("🔬 Research Sources Used"):
            st.markdown(res["research_notes"])
