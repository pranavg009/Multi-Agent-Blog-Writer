import streamlit as st
from groq import Groq
from ddgs import DDGS
import time, re, os, threading
from datetime import datetime

st.set_page_config(
    page_title="Multi-Agent Blog Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# TONE DEFINITIONS — specific instructions per tone
# ============================================================
TONE_GUIDE = {
    "Professional": (
        "formal and authoritative. Use third-person where appropriate. "
        "No contractions (use 'do not' not 'don't'). "
        "Cite statistics precisely. Confident, measured sentences. "
        "No exclamation marks. Suitable for a business executive audience."
    ),
    "Casual": (
        "conversational and friendly. Use first-person ('we', 'you'). "
        "Contractions are encouraged (it's, you'll, we've). "
        "Use relatable analogies and everyday language. "
        "Short punchy sentences mixed with longer ones. "
        "Occasional rhetorical questions to engage the reader."
    ),
    "Technical": (
        "precise and detail-oriented. Assume an expert audience. "
        "Use correct technical terminology without over-explaining basics. "
        "Include specifics: version numbers, metrics, percentages where available. "
        "Code-style precision in descriptions. Dense information per sentence."
    ),
    "Academic": (
        "scholarly and evidence-based. Formal language throughout. "
        "Present arguments with supporting evidence before conclusions. "
        "Use hedging language ('research suggests', 'evidence indicates'). "
        "Objective and analytical. Avoid sensationalism."
    ),
}

# ============================================================
# GROQ CLIENT
# ============================================================
def get_client():
    key = ""
    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return None, "GROQ_API_KEY not found"
    try:
        return Groq(api_key=key), "Groq · Llama 3.3 70B"
    except Exception as e:
        return None, str(e)[:300]


def call_llm(client, system_prompt, user_prompt, max_tokens=4096):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# SEARCH — hard threading timeout, never hangs
# ============================================================
def web_search(query: str) -> str:
    result = [None]

    def _search():
        try:
            with DDGS(timeout=8) as ddgs:
                hits = list(ddgs.text(query, max_results=5))
            if not hits:
                result[0] = f"No results found for: '{query}'"
                return
            lines = [f"Search results for: '{query}'\n{'='*50}"]
            for i, r in enumerate(hits, 1):
                lines.append(
                    f"{i}. {r.get('title', '')}\n"
                    f"   URL: {r.get('href', '')}\n"
                    f"   {r.get('body', '')[:400]}"
                )
            result[0] = "\n".join(lines)
        except Exception as e:
            result[0] = f"Search error for '{query}': {e}"

    t = threading.Thread(target=_search, daemon=True)
    t.start()
    t.join(12)  # hard 12-second cap per query

    if result[0]:
        return result[0]
    return (
        f"Search timed out for '{query}'. "
        "Use your knowledge to fill this gap and note it clearly."
    )


# ============================================================
# AGENT 1 — RESEARCHER
# ============================================================
def run_researcher(client, topic) -> str:
    queries = [
        f"{topic} statistics data 2024 2025",
        f"{topic} latest trends developments",
        f"{topic} challenges expert analysis",
    ]
    all_results = []
    for q in queries:
        all_results.append(web_search(q))
        time.sleep(0.5)

    combined = "\n\n".join(all_results)

    system = (
        "You are a Senior Research Analyst. "
        "Extract real facts from search results. "
        "Never fabricate statistics — if a stat isn't in the results, skip it. "
        "Always record source URLs next to each fact."
    )
    user = (
        f"Topic: {topic}\n\n"
        f"Raw search results:\n{combined}\n\n"
        "Produce structured research notes in Markdown with exactly these 4 sections:\n\n"
        "## Key Facts & Statistics\n"
        "(list each fact with its source URL in brackets)\n\n"
        "## Recent Developments\n"
        "(what has changed recently)\n\n"
        "## Expert Perspectives & Challenges\n"
        "(what experts say, key problems)\n\n"
        "## All Sources\n"
        "(numbered list of all URLs found)\n\n"
        "Only include facts actually present in the search results above. "
        "Minimum 300 words of notes."
    )
    return call_llm(client, system, user, max_tokens=2000)


# ============================================================
# AGENT 2 — WRITER  (word count enforced via section targets)
# ============================================================
def run_writer(client, topic, tone, word_count, research_notes) -> str:
    tone_instruction = TONE_GUIDE.get(tone, TONE_GUIDE["Professional"])

    # Calculate per-section word targets
    n_body_sections = 4 if word_count >= 1000 else 3
    intro_target     = max(80,  word_count // 7)
    conclusion_target = max(80, word_count // 8)
    body_budget      = word_count - intro_target - conclusion_target
    section_target   = body_budget // n_body_sections

    section_plan = "\n".join(
        [f"- Body Section {i+1} (H2): ~{section_target} words"
         for i in range(n_body_sections)]
    )

    system = (
        f"You are a professional blog writer. "
        f"Your current writing style is: {tone_instruction}\n\n"
        f"WORD COUNT RULE: You MUST write approximately {word_count} words "
        f"(acceptable range: {int(word_count*0.9)}–{int(word_count*1.1)}). "
        f"Count every word you write. Do not stop early."
    )
    user = (
        f"Research notes (use ONLY these facts — do not invent statistics):\n"
        f"{research_notes}\n\n"
        f"{'='*60}\n"
        f"Write a {tone} blog post about: '{topic}'\n\n"
        f"MANDATORY word count: {word_count} words\n\n"
        f"Section-by-section targets (you must hit each one):\n"
        f"- H1 Title: one strong headline\n"
        f"- Introduction: ~{intro_target} words\n"
        f"{section_plan}\n"
        f"- Conclusion (H2): ~{conclusion_target} words\n\n"
        f"Tone: {tone} — meaning {tone_instruction}\n\n"
        f"Rules:\n"
        f"1. Use ONLY facts from the research notes above\n"
        f"2. Include at least 3 statistics with their sources\n"
        f"3. Every H2 section must have at least 2 full paragraphs\n"
        f"4. Maintain {tone} tone from first word to last\n"
        f"5. Full Markdown output only\n"
        f"6. Conclusion must include a clear call-to-action\n\n"
        f"Start writing now. Do not add any preamble — begin with the H1 title:"
    )

    draft = call_llm(client, system, user, max_tokens=4096)

    # Expansion pass — if output is too short, ask for more
    actual = count_words(draft)
    if actual < int(word_count * 0.82):
        shortage = word_count - actual
        expand_system = (
            f"You are an editor. Expand this blog post by adding {shortage} more words. "
            f"Maintain the exact same {tone} tone: {tone_instruction}"
        )
        expand_user = (
            f"This post is {actual} words but needs {word_count} words "
            f"({shortage} words short).\n\n"
            f"Expand it by:\n"
            f"- Adding more depth and examples to each section\n"
            f"- Expanding thin paragraphs (anything under 3 sentences)\n"
            f"- Adding a relevant real-world example or case study\n\n"
            f"IMPORTANT: Keep the same {tone} tone and all existing facts.\n\n"
            f"Current post:\n{draft}\n\n"
            f"Rewrite the ENTIRE post with the additions included:"
        )
        draft = call_llm(client, expand_system, expand_user, max_tokens=4096)

    return draft


# ============================================================
# AGENT 3 — EDITOR + SEO
# ============================================================
def run_editor(client, blog_draft, topic, tone) -> str:
    tone_instruction = TONE_GUIDE.get(tone, TONE_GUIDE["Professional"])

    system = (
        "You are a senior editor and SEO specialist with 15 years experience. "
        "You polish writing and write high-performing SEO metadata."
    )
    user = (
        f"Blog post to edit:\n{blog_draft}\n\n"
        f"{'='*60}\n"
        f"TASK 1 — Edit the blog post:\n"
        f"- Fix any grammar or awkward phrasing\n"
        f"- Ensure consistent {tone} tone throughout: {tone_instruction}\n"
        f"- Improve flow between paragraphs\n"
        f"- Do NOT change facts, statistics, or structure\n\n"
        f"TASK 2 — Append SEO metadata after the post using these EXACT markers:\n\n"
        f"[SEO_TITLE] Write a compelling title under 60 characters. "
        f"Include the main keyword. Make it click-worthy. "
        f"Example format: 'How AI Is Reshaping Software in 2025'\n\n"
        f"[META_DESC] Write a meta description 140-160 characters. "
        f"Include a benefit and a keyword. "
        f"Example: 'Discover how AI agents are transforming software development — "
        f"from automated testing to full code generation. What it means for developers.'\n\n"
        f"[TAGS] Exactly 5 comma-separated tags. "
        f"Mix broad and specific: e.g. 'artificial intelligence, software development, "
        f"AI tools, developer productivity, tech trends 2025'\n\n"
        f"Output the full edited post first, then the 3 metadata lines. "
        f"Nothing after the [TAGS] line."
    )
    return call_llm(client, system, user, max_tokens=4096)


# ============================================================
# PARSER
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
        res["meta_desc"] = "AI-generated blog post"
        res["tags"]      = ["ai", "blog", "content", "writing", "technology"]

    return res


def count_words(text: str) -> int:
    return len(re.sub(r'[#*`_\[\]()\-]', '', text).split())


# ============================================================
# FULL PIPELINE
# ============================================================
def run_pipeline(client, topic, tone, word_count, on_writer, on_editor):
    research_notes = run_researcher(client, topic)
    on_writer()
    blog_draft = run_writer(client, topic, tone, word_count, research_notes)
    on_editor()
    final_output = run_editor(client, blog_draft, topic, tone)
    return final_output, research_notes


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
for k, v in [("history",[]),("result",None),("running",False),("topic_in","")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("✍️ Blog Writer")
    client, provider = get_client()
    if client:
        st.success(f"✅ {provider} connected")
    else:
        st.error(f"❌ {provider}")
        st.info(
            "Add to Streamlit secrets:\n\n"
            "```\nGROQ_API_KEY = 'gsk_...'\n```\n\n"
            "Free at console.groq.com — no credit card needed."
        )
    st.divider()
    st.subheader("⚙️ Settings")
    tone       = st.selectbox("Tone", ["Professional", "Casual", "Technical", "Academic"])
    word_count = st.slider("Target words", 300, 2000, 800, step=100)
    st.info("⏱ Estimated: 45–90 seconds\n")
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
    st.markdown("🔬 **Step 1 — Researcher** · 3 web searches → 1 LLM call")
    st.markdown("✍️ **Step 2 — Writer** · Research → draft with section targets")
    st.markdown("📝 **Step 3 — Editor** · Polish + SEO metadata")

# ============================================================
# MAIN UI
# ============================================================
st.title("✍️ Multi-Agent Blog Writer")
st.caption("Researcher → Writer → Editor · Groq Llama 3.3 70B · 3 direct API calls")

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

m1, m2, m3 = st.columns(3)
m1.metric("Tone",         tone)
m2.metric("Target Words", word_count)
m3.metric("Agents",    "3")

generate_btn = st.button(
    "🚀 Generate Blog Post",
    disabled=st.session_state.running or not client or not topic.strip(),
    type="primary",
    use_container_width=True,
)
if not client:
    st.warning("Add GROQ_API_KEY in Streamlit secrets. Free at console.groq.com")

# ============================================================
# GENERATION
# ============================================================
if generate_btn and topic.strip() and client:
    st.session_state.running  = True
    st.session_state.topic_in = topic
    st.divider()
    st.subheader("⚡ Pipeline Running")

    s1, s2, s3 = st.columns(3)
    step1 = s1.empty(); step2 = s2.empty(); step3 = s3.empty()
    step1.info("🔬 **Step 1**\nResearcher\nSearching...")
    step2.warning("✍️ **Step 2**\nWriter\nWaiting...")
    step3.warning("📝 **Step 3**\nEditor\nWaiting...")

    status  = st.empty()
    prog    = st.progress(10)
    start_t = time.time()
    status.info("🔬 Researcher running 3 web searches...")

    def on_writer():
        step1.success("🔬 **Step 1**\nResearcher\n✅ Done")
        step2.info("✍️ **Step 2**\nWriter\nDrafting...")
        prog.progress(45)
        status.info("✍️ Writer drafting post (may expand if too short)...")

    def on_editor():
        step2.success("✍️ **Step 2**\nWriter\n✅ Done")
        step3.info("📝 **Step 3**\nEditor\nPolishing...")
        prog.progress(80)
        status.info("📝 Editor polishing + generating SEO metadata...")

    try:
        raw, research_notes = run_pipeline(
            client, topic, tone, word_count, on_writer, on_editor
        )
        elapsed = round(time.time() - start_t, 1)

        step3.success("📝 **Step 3**\nEditor\n✅ Done")
        prog.progress(100)
        status.success(f"✅ Done in {elapsed}s")

        parsed = parse_output(raw)
        parsed.update({
            "topic":             topic,
            "tone":              tone,
            "word_count_target": word_count,
            "word_count_actual": count_words(parsed["blog_post"]),
            "research_notes":    research_notes,
            "time_taken":        elapsed,
        })
        st.session_state.history.append({
            "id":    str(time.time()),
            "topic": topic,
            "result": parsed,
            "time":  datetime.now().strftime("%H:%M"),
        })
        st.session_state.result  = parsed
        st.session_state.running = False
        st.rerun()

    except Exception as e:
        status.error(f"❌ Error: {str(e)[:300]}")
        step1.error("Failed"); step2.error("—"); step3.error("—")
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
    c4.metric("Time",           f"{res.get('time_taken','—')}s")

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
        with st.expander("🔬 Research Sources Used"):
            st.markdown(res["research_notes"])
