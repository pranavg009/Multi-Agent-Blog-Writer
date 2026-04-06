import streamlit as st
from openai import OpenAI
from ddgs import DDGS
import re, os, time
from datetime import datetime

st.set_page_config(
    page_title="Multi-Agent Blog Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# WHY NO CREWAI:
# CrewAI 1.13 has LITELLM_AVAILABLE=False, default max_iter=25,
# and an RPMController that sleeps 60s every 4 calls.
# Together they caused 15+ min hangs with no output.
#
# THIS APPROACH:
# - Step 1: DuckDuckGo → 3 searches → raw snippets
# - Step 2: Groq API call → structured research notes
# - Step 3: Groq API call → full blog post + SEO metadata
# - Total: exactly 2 LLM calls, no framework, no hidden loops
# - Typical runtime: 20–60 seconds
# ============================================================

def get_client():
    groq_key = ""
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    if not groq_key:
        groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        return None, "GROQ_API_KEY not found in Streamlit secrets"
    try:
        client = OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        return client, "Groq · Llama 3.3 70B"
    except Exception as e:
        return None, str(e)[:200]


def web_search(query: str, max_results: int = 5) -> str:
    """Run one DuckDuckGo search, return formatted string."""
    try:
        with DDGS(timeout=12) as d:
            results = list(d.text(query, max_results=max_results))
        if not results:
            return f"No results for: {query}"
        lines = [f"Results for '{query}':"]
        for r in results:
            lines.append(
                f"- {r.get('title','')}\n"
                f"  URL: {r.get('href','')}\n"
                f"  {r.get('body','')[:350]}"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"Search failed ({e}) — use training knowledge for this query."


def call_groq(client, system: str, user: str, max_tokens: int = 4096) -> str:
    """Single Groq API call. Raises on failure."""
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()


def run_pipeline(client, topic: str, tone: str, word_count: int):
    """
    3-step pipeline — all visible, no hidden loops.
    Returns (blog_post, seo_title, meta_desc, tags, research_notes).
    """

    # ── STEP 1: Web search (3 queries)
    q1 = web_search(f"{topic} facts statistics 2025")
    q2 = web_search(f"{topic} latest developments trends")
    q3 = web_search(f"{topic} expert opinion challenges")
    raw_search = f"{q1}\n\n{q2}\n\n{q3}"

    # ── STEP 2: Research synthesis (1 LLM call)
    research_notes = call_groq(
        client,
        system=(
            "You are a senior research analyst. Given raw web search snippets, "
            "produce structured research notes in Markdown. "
            "Include only facts present in the snippets. "
            "Format: ## Key Facts & Statistics | ## Recent Developments | "
            "## Expert Perspectives | ## Sources"
        ),
        user=(
            f"Topic: {topic}\n\n"
            f"Raw search results:\n{raw_search}\n\n"
            "Write structured research notes (minimum 300 words). "
            "Every statistic must have its source URL."
        ),
        max_tokens=2048,
    )

    # ── STEP 3: Blog writing + SEO (1 LLM call)
    blog_raw = call_groq(
        client,
        system=(
            f"You are a professional blog writer with 15 years of experience. "
            f"Write in a {tone} tone. Use ONLY facts from the research notes provided. "
            "Output the blog post in Markdown (H1 title, H2 sections), "
            "then on new lines append exactly:\n"
            "[SEO_TITLE] (under 60 chars)\n"
            "[META_DESC] (under 160 chars)\n"
            "[TAGS] tag1, tag2, tag3, tag4, tag5\n"
            "Nothing after [TAGS]."
        ),
        user=(
            f"Research notes:\n{research_notes}\n\n"
            f"Write a complete {tone} blog post about '{topic}' "
            f"targeting approximately {word_count} words. "
            "Include at least 3 statistics with sources. "
            "Then append the SEO metadata markers."
        ),
        max_tokens=4096,
    )

    return blog_raw, research_notes


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


# ── Suggested topics
SUGGESTED_TOPICS = [
    "How AI agents are changing software development in 2025",
    "The future of remote work: trends shaping 2025 and beyond",
    "Quantum computing: what businesses need to know now",
    "Sustainable tech: how green AI is reducing carbon footprints",
    "Cybersecurity threats every startup should prepare for",
    "The rise of no-code tools and what it means for developers",
]

# ── Session state
for k, v in [("history", []), ("result", None), ("running", False), ("topic_in", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar
with st.sidebar:
    st.title("✍️ Blog Writer")
    client, provider = get_client()
    if client:
        st.success(f"✅ {provider} connected")
    else:
        st.error(f"❌ {provider}")
        st.info("Add to Streamlit secrets:\n```\nGROQ_API_KEY = 'gsk_...'\n```\nFree at console.groq.com")

    st.divider()
    st.subheader("⚙️ Settings")
    tone       = st.selectbox("Tone", ["Professional", "Casual", "Technical", "Academic"])
    word_count = st.slider("Target words", 300, 2000, 800, step=100)
    st.info("⏱ ~20–60 seconds\n(2 LLM calls · Groq speed)")
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
    st.subheader("🔄 Pipeline")
    st.markdown("🔍 **Step 1** · DuckDuckGo (3 searches)")
    st.markdown("🔬 **Step 2** · Groq — Research synthesis")
    st.markdown("✍️ **Step 3** · Groq — Write + SEO")

# ── Main UI
st.title("✍️ Multi-Agent Blog Writer")
st.caption("Search → Research → Write · Direct Groq API · No framework overhead")

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
    disabled=st.session_state.running,
)

c1, c2, c3 = st.columns(3)
c1.metric("Tone", tone)
c2.metric("Target Words", word_count)
c3.metric("LLM Calls", "2")

generate_btn = st.button(
    "🚀 Generate Blog Post",
    disabled=st.session_state.running or not client or not topic.strip(),
    type="primary",
    use_container_width=True,
)
if not client:
    st.warning("Add GROQ_API_KEY in Streamlit secrets. Free at console.groq.com")

# ── Generation
if generate_btn and topic.strip() and client:
    st.session_state.running = True
    st.session_state.topic_in = topic
    st.divider()
    st.subheader("⚡ Generating...")

    col1, col2, col3 = st.columns(3)
    with col1: s1 = st.empty(); s1.info("🔍 **Step 1**\nSearching web...")
    with col2: s2 = st.empty(); s2.warning("🔬 **Step 2**\nResearch — waiting")
    with col3: s3 = st.empty(); s3.warning("✍️ **Step 3**\nWriting — waiting")

    status = st.empty()
    prog   = st.progress(0)
    t0     = time.time()

    try:
        status.info("🔍 Step 1 — Searching DuckDuckGo (3 queries)...")
        prog.progress(10)

        # Run pipeline with visible progress updates
        # Step 1+2: search + research
        q1 = web_search(f"{topic} facts statistics 2025")
        q2 = web_search(f"{topic} latest developments trends")
        q3 = web_search(f"{topic} expert opinion challenges")
        raw_search = f"{q1}\n\n{q2}\n\n{q3}"
        s1.success("🔍 **Step 1**\nSearch ✅")
        prog.progress(30)

        status.info("🔬 Step 2 — Synthesising research with Groq...")
        research_notes = call_groq(
            client,
            system=(
                "You are a senior research analyst. Given raw web search snippets, "
                "produce structured research notes in Markdown. "
                "Include only facts present in the snippets. "
                "Format: ## Key Facts & Statistics | ## Recent Developments | "
                "## Expert Perspectives | ## Sources"
            ),
            user=(
                f"Topic: {topic}\n\nRaw search results:\n{raw_search}\n\n"
                "Write structured research notes (minimum 300 words). "
                "Every statistic must have its source URL."
            ),
            max_tokens=2048,
        )
        s2.success("🔬 **Step 2**\nResearch ✅")
        prog.progress(65)

        status.info("✍️ Step 3 — Writing blog post with Groq...")
        blog_raw = call_groq(
            client,
            system=(
                f"You are a professional blog writer with 15 years of experience. "
                f"Write in a {tone} tone. Use ONLY facts from the research notes provided. "
                "Output the blog post in Markdown (H1 title, H2 sections), "
                "then on new lines append exactly:\n"
                "[SEO_TITLE] (under 60 chars)\n"
                "[META_DESC] (under 160 chars)\n"
                "[TAGS] tag1, tag2, tag3, tag4, tag5\n"
                "Nothing after [TAGS]."
            ),
            user=(
                f"Research notes:\n{research_notes}\n\n"
                f"Write a complete {tone} blog post about '{topic}' "
                f"targeting approximately {word_count} words. "
                "Include at least 3 statistics with sources. "
                "Then append the SEO metadata markers."
            ),
            max_tokens=4096,
        )
        s3.success("✍️ **Step 3**\nWriting ✅")
        prog.progress(100)

        elapsed = round(time.time() - t0, 1)
        status.success(f"✅ Done in {elapsed}s")

        parsed = parse_output(blog_raw)
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
            "time":   datetime.now().strftime("%H:%M"),
        })
        st.session_state.result  = parsed
        st.session_state.running = False
        st.rerun()

    except Exception as e:
        status.error(f"❌ Error: {str(e)[:300]}")
        s1.error("Step 1 —"); s2.error("Step 2 —"); s3.error("Step 3 —")
        st.session_state.running = False

# ── Display result
if st.session_state.result:
    res = st.session_state.result
    st.divider()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words Written",  res.get("word_count_actual", "—"))
    m2.metric("Target Words",   res.get("word_count_target", "—"))
    m3.metric("Tags",           len(res.get("tags", [])))
    m4.metric("Time",           f"{res.get('time_taken','—')}s")

    st.subheader("📄 Generated Blog Post")
    dl_col, _ = st.columns([1, 3])
    with dl_col:
        full_md = (
            res["blog_post"]
            + f"\n\n---\n**SEO Title:** {res['seo_title']}\n"
            + f"**Meta Desc:** {res['meta_desc']}\n"
            + f"**Tags:** {', '.join(res['tags'])}\n"
        )
        st.download_button(
            "⬇️ Download .md", full_md,
            file_name="blog_post.md", mime="text/markdown",
            use_container_width=True,
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
        with st.expander("🔬 Research Notes"):
            st.markdown(res["research_notes"])
