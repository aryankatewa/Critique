import re
import streamlit as st
from google import genai
from google.genai import types

# =========================================================
# REQUIRED MODEL (hard requirement)
# =========================================================
MODEL_NAME = "gemma-3-27b-it"


# =========================================================
# Streamlit config MUST be first Streamlit call
# =========================================================
st.set_page_config(
    page_title="Critique — Reflective UI Critique",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Styling (light, studio-crit feel)
# =========================================================
st.markdown(
    """
<style>
    .main .block-container { padding-top: 1.75rem; max-width: 1200px; }
    .card {
        background: #FFFFFF;
        color: #0F172A;
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 0 rgba(15, 23, 42, 0.04);
    }
    .muted { color: #475569; }
    .small { font-size: 0.92rem; }
    hr { margin: 1.1rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# Identity loading
# =========================================================
@st.cache_data(show_spinner=False)
def load_identity(filename: str = "identity.txt") -> str:
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        st.error(f"Identity file '{filename}' not found. Make sure it sits next to app.py.")
        st.stop()


def extract_tag(identity_text: str, tag: str) -> str:
    """Extract <Tag>...</Tag> contents from the identity file."""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    m = re.search(pattern, identity_text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_developer_name(identity_text: str) -> str:
    """
    Best-effort: try to find 'created by' / 'by <name>' style attribution in the <Role>.
    If not found, return a sensible fallback.
    """
    role = extract_tag(identity_text, "Role") or identity_text
    # Common patterns: "created by X", "by X", "(c) X"
    patterns = [
        r"created by\s+([^\.\n]+)",
        r"designed by\s+([^\.\n]+)",
        r"\(c\)\s*([^\|\n]+)",
        r"\bby\s+([A-Z][A-Za-z0-9\-\.\s]+)\b",
    ]
    for p in patterns:
        m = re.search(p, role, flags=re.IGNORECASE)
        if m:
            name = m.group(1).strip(" -|")
            # Keep it tidy (avoid grabbing too much)
            return name[:60].strip()
    return "Identity Author"


# =========================================================
# GenAI wiring (per skill)
# =========================================================
def ensure_genai_client() -> bool:
    if "genai_client" not in st.session_state or st.session_state.genai_client is None:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.genai_client = genai.Client(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Failed to initialize AI. Check GEMINI_API_KEY in Streamlit secrets. Details: {e}")
            return False
    return True


def get_gemma_config():
    return types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=2000,
    )


def build_critique_prompt(identity: str, form_data: dict) -> str:
    """
    Prompt enforces identity rules:
    - Observations vs interpretations
    - Reflective questions first
    - No prescriptive “correct fix”
    - Avoid evaluative language (good/bad/better/worse)
    """
    return f"""SYSTEM IDENTITY & GUIDELINES:
{identity}

TASK:
You are reviewing a UI / digital experience. The user is providing context and (optionally) an interface description.
Ground everything in what the user provided. Do not speculate beyond the given input.

HARD CONSTRAINTS:
- Never prescribe final solutions or present design changes as “the correct fix.”
- Avoid evaluative language like “good,” “bad,” “better,” “worse.”
- Clearly separate OBSERVATIONS (what is visible / stated) from INTERPRETATIONS (what it may imply).
- Prioritize REFLECTIVE QUESTIONS before offering directions or frameworks.
- Preserve user authorship: decisions belong to the user.

USER CONTEXT:
Design stage: {form_data.get("stage")}
Audience: {form_data.get("audience")}
Primary user goal/task: {form_data.get("user_goal")}
Platform: {form_data.get("platform")}
What they want critique on: {form_data.get("focus_areas")}

WHAT THEY'RE SHARING (copy, layout notes, screenshot description, link notes, etc.):
{form_data.get("artifact_text")}

OUTPUT FORMAT (use these headings exactly):
1) Observations
- Bullet list of concrete, observable elements from the input.

2) Interpretations (as optional lenses)
- Bullet list. Each item must begin with “One way to interpret this is…”

3) Reflective questions to ask next
- Numbered list (5–10 questions). Keep them specific and actionable.

4) Directions to explore (as experiments, not fixes)
- 3–6 bullets, each framed as an experiment or test, plus a lightweight validation method (e.g., quick usability check).

5) Accessibility & inclusion checks (quick scan)
- Bullet list of checks relevant to the described interface.

Remember: stay calm, studio-critique tone. Do not add a preamble or closing flourish.
"""


def generate_from_form(form_data: dict, identity_context: str) -> str:
    if not ensure_genai_client():
        return "AI client failed to initialize."

    full_prompt = build_critique_prompt(identity_context, form_data)

    try:
        client = st.session_state.genai_client
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[full_prompt],
            config=get_gemma_config(),
        )
        return (response.text or "").strip()
    except Exception as e:
        return f"Error generating critique: {e}"


def stream_refinement_response(user_input: str, identity_context: str, chat_history: list, base_critique: str):
    if not ensure_genai_client():
        yield "AI client failed to initialize."
        return

    history_text = "\n".join(
        [f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in chat_history[-10:]]
    )

    full_prompt = f"""SYSTEM IDENTITY & GUIDELINES:
{identity_context}

CONTEXT:
The assistant previously produced this critique (treat it as draft output, not truth):
{base_critique}

CONVERSATION HISTORY:
{history_text}

USER MESSAGE:
{user_input}

INSTRUCTIONS:
Respond according to the identity. Keep the same constraints (no evaluative language, no “correct fix”).
If the user asks for changes, frame them as options/experiments and preserve authorship.
"""

    try:
        client = st.session_state.genai_client
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=[full_prompt],
            config=get_gemma_config(),
        ):
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error: {e}"


# =========================================================
# Session state
# =========================================================
def init_session_state():
    defaults = {
        "genai_client": None,
        "has_interacted": False,
        "last_result": "",
        "chat_messages": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()

identity = load_identity("identity.txt")
developer_name = extract_developer_name(identity)

app_title = "Critique"
role_summary = extract_tag(identity, "Role")
goal_summary = extract_tag(identity, "Goal")

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.markdown(f"## :material/psychology: {app_title}")
    st.caption("Reflective, research-informed UI critique — grounded in what you provide.")

    with st.expander(":material/info: About this app", expanded=False):
        st.markdown(
            f"""
<div class="small muted">
<b>{app_title}</b> — identity-driven critique assistant<br/>
Developed by <b>{developer_name}</b><br/>
Powered by <b>Gemma 3.27B IT</b> (<code>{MODEL_NAME}</code>)
</div>
""",
            unsafe_allow_html=True,
        )
        if goal_summary:
            st.markdown("**Goal (from identity):**")
            st.write(goal_summary)

    st.divider()
    if st.button(
        "Reset session",
        icon=":material/refresh:",
        use_container_width=True,
        help="Clears the current critique draft and chat refinement history.",
    ):
        st.session_state.has_interacted = False
        st.session_state.last_result = ""
        st.session_state.chat_messages = []
        st.rerun()


# =========================================================
# Landing
# =========================================================
st.markdown(f"# {app_title}")
if role_summary:
    st.markdown(f"<div class='muted'>{role_summary}</div>", unsafe_allow_html=True)

if not st.session_state.has_interacted:
    st.markdown(
        """
<div class="card">
<b>How this works</b><br/>
<span class="muted">
Start with a structured critique request (form) so the feedback stays grounded.
Then, if you want, use the refinement chat to iterate on the critique together.
</span>
</div>
""",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "Try an example",
            icon=":material/play_circle:",
            use_container_width=True,
            help="Loads a short sample prompt so you can see the output format.",
        ):
            st.session_state.has_interacted = True
            st.session_state.example_mode = True
            st.rerun()
    with col2:
        if st.button(
            "Start fresh",
            icon=":material/edit:",
            use_container_width=True,
            help="Begin with your own design/context input.",
        ):
            st.session_state.has_interacted = True
            st.session_state.example_mode = False
            st.rerun()

    st.stop()


# =========================================================
# Tabs: Form critique + Refinement chat
# =========================================================
tab1, tab2 = st.tabs(
    [
        ":material/assignment: Critique request",
        ":material/forum: Refinement chat",
    ]
)

# ---------------------------
# Tab 1: Critique Request
# ---------------------------
with tab1:
    st.markdown("### Provide context (ground the critique)")
    with st.expander(":material/help: Tips for describing the interface", expanded=False):
        st.markdown(
            """
- If you have a screenshot, describe what’s on screen: sections, headings, buttons, labels, states (empty/error/loading).
- Include the primary task the UI supports (what users come here to do).
- Paste key UI copy (headlines, button labels, helper text) to make content critique possible.
"""
        )

    example_prefill = st.session_state.get("example_mode", False)

    if example_prefill and not st.session_state.last_result:
        example_artifact = """Screen: mobile signup step.
Top: "Create your account" header.
Fields: Email, Password, Confirm Password.
Password helper text below field: "Must be 8+ characters."
Primary button: "Continue" (full width).
Secondary text link below: "Already have an account? Log in"
Error state: when password too short, red text appears but no icon; button remains enabled.
"""
    else:
        example_artifact = ""

    with st.form("critique_form", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            stage = st.selectbox(
                "Design stage",
                ["Early (exploring)", "Mid (iterating)", "Late (polishing)"],
                help="Helps calibrate critique depth and specificity.",
            )
        with c2:
            platform = st.selectbox(
                "Platform",
                ["Web", "Mobile", "Tablet", "Desktop", "Other / Mixed"],
                help="Used as a lens for common interaction expectations.",
            )
        with c3:
            focus = st.multiselect(
                "Focus areas",
                [
                    "Hierarchy & scannability",
                    "Information architecture",
                    "Interaction & feedback",
                    "Content & clarity",
                    "Consistency & patterns",
                    "Accessibility & inclusion",
                ],
                default=["Hierarchy & scannability", "Interaction & feedback"],
                help="Pick what you want the critique to emphasize.",
            )

        audience = st.text_input(
            "Who is this for? (audience)",
            placeholder="e.g., first-time users, returning customers, internal admins…",
            help="A short description of who uses this interface.",
        )
        user_goal = st.text_input(
            "What is the user's primary goal on this screen?",
            placeholder="e.g., create an account, compare plans, check out…",
            help="The main task the UI should help users complete.",
        )
        artifact_text = st.text_area(
            "Describe what’s on the screen (or paste key UI copy)",
            value=example_artifact,
            height=220,
            help="Describe layout, labels, states, and any visible feedback. Paste UI copy when possible.",
        )

        submitted = st.form_submit_button(
            "Generate critique draft",
            icon=":material/auto_awesome:",
            help="Generates a structured critique using the identity rules.",
            use_container_width=True,
        )

    if submitted:
        form_data = {
            "stage": stage,
            "platform": platform,
            "focus_areas": ", ".join(focus) if focus else "Not specified",
            "audience": audience or "Not specified",
            "user_goal": user_goal or "Not specified",
            "artifact_text": artifact_text.strip() or "Not provided",
        }

        with st.spinner("Generating critique…"):
            result = generate_from_form(form_data, identity)

        st.session_state.last_result = result
        st.session_state.chat_messages = []  # start fresh refinement thread
        st.success("Draft generated. You can refine it in the next tab.")
        st.divider()

    if st.session_state.last_result:
        st.markdown("### Critique draft")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(st.session_state.last_result)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------
# Tab 2: Refinement Chat
# ---------------------------
with tab2:
    st.markdown("### Iterate on the critique (keep authorship)")
    if not st.session_state.last_result:
        st.info("Generate a critique draft first in the “Critique request” tab.")
    else:
        # Render chat history
        for m in st.session_state.chat_messages:
            avatar = ":material/person:" if m["role"] == "user" else ":material/smart_toy:"
            with st.chat_message(m["role"], avatar=avatar):
                st.markdown(m["content"])

        user_msg = st.chat_input(
            "Ask for clarification, alternative framings, or deeper questions…",
            max_chars=1200,
        )

        if user_msg:
            st.session_state.chat_messages.append({"role": "user", "content": user_msg})
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(user_msg)

            with st.chat_message("assistant", avatar=":material/smart_toy:"):
                stream = stream_refinement_response(
                    user_input=user_msg,
                    identity_context=identity,
                    chat_history=st.session_state.chat_messages,
                    base_critique=st.session_state.last_result,
                )
                assistant_text = st.write_stream(stream)

            st.session_state.chat_messages.append({"role": "assistant", "content": assistant_text})


# =========================================================
# Footer
# =========================================================
st.divider()
st.caption(
    f"Developed by {developer_name} • Powered by Gemma 3.27B IT ({MODEL_NAME}) • Built with Streamlit"
)
