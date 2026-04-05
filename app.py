import streamlit as st
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field
from typing import Type
import time, re, os
from datetime import datetime

st.set_page_config(
    page_title="Multi-Agent Blog Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LLM SETUP — FINAL CORRECT APPROACH
#
# CrewAI >=1.9 requires Agent(llm=) to be a plain STRING.
# Passing any object (LLM(), ChatOpenAI()) causes a Pydantic
# validation error. The string goes directly to LiteLLM inside
# CrewAI — bypassing CrewAI's own provider whitelist check.
#
# LiteLLM natively supports Groq via GROQ_API_KEY env var.
# We set the env var, return the model string, done.
# ============================================================
def setup_llm():
    groq_key = ""
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_key = os.environ.get("GROQ_API_KEY", "")

    if not groq_key:
        return None, "GROQ_API_KEY not found in secrets"

    # LiteLLM reads GROQ_API_KEY directly — no other config needed
    os.environ["GROQ_API_KEY"] = groq_key

    # Return the model string — Agent(llm=) accepts this directly
    return "groq/llama-3.3-70b-versatile", "Groq · Llama 3.3 70B"


# ============================================================
# SEARCH TOOL — DuckDuckGo
# ============================================================
class SearchInput(BaseModel):
    query: str = Field(description="The search query to look up")

class DDGSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web using DuckDuckGo. Returns real results with titles, "
        "URLs, and content snippets. Use for facts, stats, and recent news."
    )
    args_schema: Type[BaseModel] = SearchInput
    model_config = {"arbitrary_types_allowed": True}

    def _run(self, query: str) -> str:
        for attempt in range(3):
            try:
                with DDGS(timeout=15) as ddgs:
                    results = list(ddgs.text(query, max_results=5))
                if not results:
                    return f"No results found for: '{query}'"
                output = [f"Search results for: '{query}'\n{'='*50}\n"]
                for i, r in enumerate(results, 1):
                    output.append(
                        f"{i}. {r.get('title', 'No title')}\n"
                        f"   URL: {r.get('href', '')}\n"
                        f"   {r.get('body', '')[:400]}\n"
                    )
                return "\n".join(output)
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    return (
                        f"Web search unavailable ({e}). "
                        "Use your training knowledge to answer. "
                        "Note clearly which facts are inferred rather than cited."
                    )
        return "Search unavailable — use training knowledge and note this clearly."


search_tool = DDGSearchTool()


# ============================================================
# OUTPUT PARSER
# ============================================================
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
        res["meta_desc"] = "AI blog post"
        res["tags"] = ["ai", "blog", "content"]

    return res


def count_words(text: str) -> int:
    return len(re.sub(r'[#*`_\[\]()]', '', text).split())


# ============================================================
# PIPELINE
# ============================================================
def run_crew_pipeline(topic, tone, word_count, model_str):

    researcher = Agent(
        role="Senior Research Analyst",
        goal=(
            f"Research '{topic}' using 3 web searches on different angles. "
            "Return structured notes with key facts, statistics, and source URLs. "
            "Never fabricate statistics — only include what you find."
        ),
        backstory=(
            "A meticulous analyst with 10 years in investigative journalism. "
            "Allergic to misinformation. Always cites sources."
        ),
        tools=[search_tool],
        llm=model_str,
        verbose=False,
        allow_delegation=False,
        max_iter=2,
        max_rpm=4,
    )

    writer_editor = Agent(
        role="Professional Blog Writer and Editor",
        goal=(
            f"Write a complete polished {tone} blog post of ~{word_count} words "
            f"about '{topic}' using ONLY the research notes. "
            "Self-edit for grammar, flow, and tone consistency. "
            "Then generate SEO metadata.\n\n"
            "Output format (follow exactly):\n"
            "1. Full polished blog post in Markdown (H1 title, H2 sections)\n"
            "2. Then on new lines:\n"
            "[SEO_TITLE] under 60 chars\n"
            "[META_DESC] under 160 chars\n"
            "[TAGS] exactly 5 comma-separated tags\n\n"
            "Rules: only use facts from research notes. "
            "At least 3 statistics from research. "
            f"Tone must be {tone} throughout. "
            "Nothing after [TAGS] line."
        ),
        backstory=(
            "Senior writer published in TechCrunch and Wired. "
            "15 years of editing experience. "
            "Every stat is sourced, every sentence earns its place."
        ),
        tools=[],
        llm=model_str,
        verbose=False,
        allow_delegation=False,
        max_iter=2,
        max_rpm=4,
    )

    research_task = Task(
        description=(
            f"Research '{topic}'. Run exactly 3 web searches:\n"
            f"1. '{topic} facts statistics 2025'\n"
            f"2. '{topic} latest developments'\n"
            f"3. '{topic} expert opinion challenges'\n\n"
            "Return structured Markdown notes:\n"
            "## Key Facts & Statistics (with source URLs)\n"
            "## Recent Developments\n"
            "## Expert Perspectives\n"
            "## Sources (all URLs found)\n\n"
            "Only include facts actually found in search results."
        ),
        expected_output=(
            "Structured Markdown research notes, 4 sections, min 300 words. "
            "All statistics have source URLs."
        ),
        agent=researcher,
    )

    write_edit_task = Task(
        description=(
            f"Using the research notes, write a complete {tone} blog post "
            f"about '{topic}' (~{word_count} words). "
            "Self-edit for grammar and flow. "
            "Then append SEO metadata using the exact markers from your goal. "
            "Do not search the web — use only the research notes provided."
        ),
        expected_output=(
            f"Complete polished blog post ~{word_count} words in Markdown, then:\n"
            "[SEO_TITLE] ...\n[META_DESC] ...\n[TAGS] ..., ..., ..., ..., ..."
        ),
        agent=writer_editor,
        context=[research_task],
    )

    crew = Crew(
        agents=[researcher, writer_editor],
        tasks=[research_task, write_edit_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
        max_rpm=4,
    )

    result = crew.kickoff()

    research_notes = ""
    try:
        research_notes = str(
            research_task.output.raw
            if hasattr(research_task.output, "raw")
            else research_task.output
        )
    except Exception:
        research_notes = "Research notes unavailable."

    return str(result), research_notes


# ============================================================
# SUGGESTED TOPICS
# ============================================================
SUGGESTED_TOPICS = [
    "How AI agents are changing software development in 2025",
    "The future of remote work: trends shaping 2025 and beyond",
    "Quantum computing: what businesses need to know now",
    "Sustainable tech: how green AI is reducing carbon footprints",
    "Cybersecurity threats every startup should prepare for",
    "The rise of no-code tools and what it means for developers",
]

# ============================================================
# SESSION STATE
# ============================================================
if "history"  not in st.session_state: st.session_state.history  = []
if "result"   not in st.session_state: st.session_state.result   = None
if "running"  not in st.session_state: st.session_state.running  = False
if "topic_in" not in st.session_state: st.session_state.topic_in = ""

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("✍️ Blog Writer")
    model_str, provider = setup_llm()

    if model_str:
        st.success(f"✅ {provider} connected")
    else:
        st.error(f"❌ {provider}")
        st.info(
            "Add your Groq key to Streamlit secrets:\n\n"
            "```\nGROQ_API_KEY = 'gsk_...'\n```\n\n"
            "Free at console.groq.com — no credit card needed."
        )
    st.divider()
    st.subheader("⚙️ Settings")
    tone       = st.selectbox("Tone", ["Professional", "Casual", "Technical", "Academic"])
    word_count = st.slider("Target words", 300, 2000, 800, step=100)
    st.info("⏱ Estimated: 2–5 min\n(2 agents · Groq speed)")
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
    st.markdown("🔬 **Step 1 — Researcher** · DuckDuckGo search")
    st.markdown("✍️ **Step 2 — Writer** · Drafts the post")
    st.markdown("📝 **Step 3 — Editor** · Polishes + SEO")

# ============================================================
# MAIN UI
# ============================================================
st.title("✍️ Multi-Agent Blog Writer")
st.caption("Researcher → Writer → Editor · CrewAI + Groq + DuckDuckGo")

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
    disabled=st.session_state.running or not model_str or not topic.strip(),
    type="primary",
    use_container_width=True
)
if not model_str:
    st.warning("Add GROQ_API_KEY in Streamlit secrets. Free at console.groq.com")

# ============================================================
# GENERATION
# ============================================================
if generate_btn and topic.strip() and model_str:
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
    start_t = time.time()

    status.info("🔬 Researcher searching DuckDuckGo... (1/3)")
    prog.progress(10)

    try:
        raw, research_notes = run_crew_pipeline(topic, tone, word_count, model_str)
        elapsed = round(time.time() - start_t, 1)

        step1.success("🔬 **Step 1**\nResearcher\n✅ Done")
        step2.success("✍️ **Step 2**\nWriter\n✅ Done")
        step3.success("📝 **Step 3**\nEditor\n✅ Done")
        prog.progress(100)
        status.success(f"✅ Completed in {elapsed}s")

        parsed = parse_output(raw)
        parsed["topic"]             = topic
        parsed["tone"]              = tone
        parsed["word_count_target"] = word_count
        parsed["word_count_actual"] = count_words(parsed["blog_post"])
        parsed["research_notes"]    = research_notes
        parsed["time_taken"]        = elapsed

        st.session_state.history.append({
            "id":     str(time.time()),
            "topic":  topic,
            "result": parsed,
            "time":   datetime.now().strftime("%H:%M")
        })
        st.session_state.result  = parsed
        st.session_state.running = False
        st.rerun()

    except Exception as e:
        status.error(f"❌ Error: {str(e)[:300]}")
        step1.error("🔬 Failed")
        step2.error("✍️ —")
        step3.error("📝 —")
        st.session_state.running = False

# ============================================================
# DISPLAY RESULT
# ============================================================
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
            file_name="blog_post.md",
            mime="text/markdown",
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
