import re
from datetime import datetime
import streamlit as st
from google import genai
from google.genai import types

# ------------------------------------------------------------
# Page config MUST be the first Streamlit call (best practice).
# ------------------------------------------------------------
st.set_page_config(
    page_title="Critique",
    page_icon=":material/psychology:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_NAME = "gemma-3-27b-it"


# -----------------------------
# Identity loading + parsing
# -----------------------------
@st.cache_data(show_spinner=False)
def load_identity(filename: str = "identity.txt") -> str:
    """Load the identity file that defines the assistant's behavior."""
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()


def _extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    return (m.group(1).strip() if m else "")


def parse_identity(identity_text: str) -> dict:
    """Parse xml-like identity tags into a dict."""
    return {
        "role": _extract_tag(identity_text, "Role"),
        "goal": _extract_tag(identity_text, "Goal"),
        "rules": _extract_tag(identity_text, "Rules"),
        "knowledge": _extract_tag(identity_text, "Knowledge"),
        "specialized_actions": _extract_tag(identity_text, "SpecializedActions"),
        "guidelines": _extract_tag(identity_text, "Guidelines"),
    }


def extract_developer_name(identity_text: str) -> str:
    """
    Extract developer/creator name if present.
    Heuristic: look for 'created by', 'developed by', 'by <Name>' in <Role>.
    If missing, return a safe placeholder.
    """
    role = _extract_tag(identity_text, "Role")
    patterns = [
        r"created by\s+([A-Za-z][A-Za-z .,'-]{1,60})",
        r"developed by\s+([A-Za-z][A-Za-z .,'-]{1,60})",
        r"\bby\s+([A-Za-z][A-Za-z .,'-]{1,60})\b",
    ]
    for p in patterns:
        m = re.search(p, role, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().strip(".")
    return "Not specified in identity.txt"


# -----------------------------
# GenAI client + config
# -----------------------------
def ensure_genai_client() -> bool:
    """Initialize and persist the GenAI client. Returns True if successful."""
    if "genai_client" not in st.session_state or st.session_state.genai_client is None:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
            st.session_state.genai_client = genai.Client(api_key=api_key)
            return True
        except Exception as e:
            st.error(f"Failed to initialize AI: {e}")
            return False
    return True


def get_gemma_config():
    """Optimized config for Gemma 3. thinking_config is NOT supported."""
    return types.GenerateContentConfig(
        temperature=0.7,
        max_output_tokens=2000,
    )


# -----------------------------
# Session state
# -----------------------------
def init_session_state():
    defaults = {
        "has_interacted": False,
        "messages": [],  # chat messages: {"role": "user"/"assistant", "content": str}
        "genai_client": None,
        "last_result": None,
        "last_intake": None,  # dict of most recent structured input
        "notes": [],  # saved critiques (timestamp, stage, focus, result)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


# -----------------------------
# Prompt builders
# -----------------------------
def build_structured_prompt(identity_context: str, form_data: dict) -> str:
    """Prompt for the structured critique session."""
    # Keep the assistant anchored to reflective critique behavior.
    # Also: enforce "no evaluative language" explicitly.
    return f"""SYSTEM IDENTITY & GUIDELINES:
{identity_context}

TASK:
You are Critique. Provide a reflective design critique that strengthens the user's judgment.
Follow the Rules and Guidelines strictly:
- Do NOT use evaluative language (avoid "good/bad/better/worse").
- Do NOT prescribe final solutions. Offer lenses, questions, tradeoffs, and directions.
- Prioritize questions over statements.
- Preserve user authorship: the user decides.

USER CONTEXT (structured intake):
Design stage: {form_data.get("stage")}
Artifact type: {form_data.get("artifact")}
Audience/users: {form_data.get("audience")}
Primary goal of the work: {form_data.get("goal")}
Key constraints: {form_data.get("constraints")}
What the user wants critique on: {form_data.get("focus")}
Work description (what exists so far):
{form_data.get("work")}

OUTPUT FORMAT (Markdown):
Use exactly these sections and headings:

## Intent check
(2–4 bullets: what you think the user is aiming for, framed as hypotheses)

## Reflective questions
(8–12 questions, grouped if helpful)

## Lenses to apply
(3–5 lenses: heuristics, accessibility, UX principles, ethics/emotion; frame each as a lens)

## Tradeoffs worth naming
(3–6 tradeoffs, neutral framing)

## Next iteration directions
(3–5 directions framed as experiments, not fixes)

## What I’d ask you next
(3–5 short questions to continue the critique dialogue)
"""


def generate_from_form(form_data: dict, identity_context: str) -> str:
    """Generate a critique response from structured form input."""
    if not ensure_genai_client():
        return "AI initialization failed. Check GEMINI_API_KEY in Streamlit secrets."

    full_prompt = build_structured_prompt(identity_context, form_data)

    try:
        client = st.session_state.genai_client
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[full_prompt],
            config=get_gemma_config(),
        )
        return (response.text or "").strip()
    except Exception as e:
        return f"Error generating response: {str(e)}"


def generate_chat_response(user_input: str, identity_context: str, chat_history: list, last_intake: dict | None, last_result: str | None):
    """Generate a streaming response for conversational follow-ups."""
    if not ensure_genai_client():
        yield "AI initialization failed. Check GEMINI_API_KEY in Streamlit secrets."
        return

    history_text = "\n".join(
        [
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in chat_history[-10:]
        ]
    )

    intake_text = ""
    if last_intake:
        intake_text = "\n".join(
            [
                f"Design stage: {last_intake.get('stage')}",
                f"Artifact type: {last_intake.get('artifact')}",
                f"Audience/users: {last_intake.get('audience')}",
                f"Goal: {last_intake.get('goal')}",
                f"Constraints: {last_intake.get('constraints')}",
                f"Critique focus: {last_intake.get('focus')}",
            ]
        )

    full_prompt = f"""SYSTEM IDENTITY & GUIDELINES:
{identity_context}

CONTEXT:
You are continuing an ongoing critique dialogue.
Rules to enforce:
- Questions before statements.
- No evaluative language ("good/bad/better/worse").
- No final prescriptions; suggest experiments and lenses.
- User retains authorship.

LAST INTAKE (if any):
{intake_text if intake_text else "No structured intake provided yet."}

LAST CRITIQUE (if any):
{last_result if last_result else "No prior critique yet."}

CONVERSATION HISTORY:
{history_text}

USER MESSAGE:
{user_input}

Respond in Markdown. Keep it concise and reflective. If the user asks for the Rules/identity, briefly explain your role and stop.
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
        yield f"Error generating response: {str(e)}"


# -----------------------------
# UI helpers
# -----------------------------
def inject_css():
    st.markdown(
        """
<style>
.main .block-container { padding-top: 1.5rem; max-width: 1200px; }
.card {
  background: #FFFFFF;
  color: #1F2937;
  border-radius: 14px;
  padding: 1.1rem 1.1rem;
  border: 1px solid rgba(15, 23, 42, 0.12);
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}
.muted { color: rgba(31, 41, 55, 0.72); font-size: 0.95rem; }
.hr { height: 1px; background: rgba(15, 23, 42, 0.10); margin: 0.75rem 0; }
.small { font-size: 0.9rem; }
.kbd {
  border: 1px solid rgba(15, 23, 42, 0.15);
  border-bottom-width: 2px;
  padding: 0.05rem 0.35rem;
  border-radius: 6px;
  font-size: 0.85rem;
  background: rgba(15, 23, 42, 0.03);
}
</style>
""",
        unsafe_allow_html=True,
    )


def landing(identity_parts: dict):
    st.markdown("## Welcome to **Critique**")
    st.markdown(
        """
A reflective critique partner for design students and early-career designers.
Use it to clarify intent, surface assumptions, and explore tradeoffs—without turning feedback into a verdict.
"""
    )

    with st.expander(":material/info: What this assistant will and won’t do", expanded=False):
        st.markdown(
            """
**Will do**
- Ask reflective questions to strengthen your thinking
- Offer optional lenses (heuristics, accessibility, UX principles)
- Name tradeoffs and suggest experiments for iteration

**Won’t do**
- Declare your work “good” or “bad”
- Prescribe final solutions as “the right answer”
- Replace your judgment
"""
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start a critique session", icon=":material/psychology:", use_container_width=True,
                     help="Open the structured critique form to get thoughtful, stage-aware feedback."):
            st.session_state.has_interacted = True
            st.rerun()
    with col2:
        if st.button("Try an example", icon=":material/auto_fix_high:", use_container_width=True,
                     help="Loads a sample project description so you can see the critique style."):
            st.session_state.has_interacted = True
            st.session_state.example_mode = True
            st.rerun()


def sidebar_about(identity_text: str):
    dev = extract_developer_name(identity_text)
    parts = parse_identity(identity_text)

    with st.sidebar:
        st.markdown("### Critique")
        st.caption("Reflective design critique assistant")

        with st.expander(":material/info: About", expanded=False):
            st.markdown(
                f"""
**Purpose**  
{parts.get("goal") or "—"}

**Developed by**  
{dev}

**Model**  
`{MODEL_NAME}`

**Identity source**  
Loaded from `identity.txt`
"""
            )

        with st.expander(":material/rule: Identity (read-only)", expanded=False):
            st.code(identity_text, language="xml")


# -----------------------------
# Main app
# -----------------------------
inject_css()
identity_context = load_identity("identity.txt")
identity_parts = parse_identity(identity_context)

sidebar_about(identity_context)

if not st.session_state.has_interacted:
    # If user chose example mode, pre-fill some defaults via session_state keys (not inside form values).
    landing(identity_parts)
    st.stop()

# Top header
st.markdown(
    """
<div class="card">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:1rem;">
    <div>
      <div style="font-size:1.4rem; font-weight:700;">Critique</div>
      <div class="muted">A reflective critique partner — questions, lenses, tradeoffs, iteration.</div>
    </div>
    <div class="small muted">Tip: use the <span class="kbd">Crit Session</span> tab to start, then refine in <span class="kbd">Follow‑up Chat</span>.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_crit, tab_chat, tab_notes = st.tabs(
    [":material/psychology: Crit Session", ":material/chat: Follow‑up Chat", ":material/history: Session Notes"]
)

# -----------------------------
# Crit Session tab (structured form)
# -----------------------------
with tab_crit:
    st.markdown("### Structured critique intake")

    # Example mode defaults (safe: initialize session_state keys used by widgets)
    if "example_mode" in st.session_state and st.session_state.example_mode:
        if "ex_seeded" not in st.session_state:
            st.session_state.ex_seeded = True
            st.session_state.stage = "Early ideation"
            st.session_state.artifact = "App concept / feature"
            st.session_state.audience = "Design students submitting project pitches"
            st.session_state.goal_text = "Help students get reflective critique quickly while preserving authorship"
            st.session_state.constraints = "Must avoid prescriptive 'fixes' and evaluative language; keep responses structured"
            st.session_state.focus = "How to structure the critique so it stays reflective"
            st.session_state.work = (
                "I’m building a bot called Critique that asks reflective questions and applies design lenses. "
                "I want the feedback to adapt by stage (early/mid/late) and emphasize tradeoffs and iteration."
            )

    with st.form("critique_intake"):
        c1, c2 = st.columns(2)

        with c1:
            stage = st.selectbox(
                "Design stage",
                ["Early ideation", "Mid-fidelity", "Late stage"],
                key="stage",
                help="Sets the critique depth and focus. Early = exploratory questions; late = edge cases and consequences.",
            )
            artifact = st.selectbox(
                "Artifact type",
                ["App concept / feature", "Wireframe / layout", "Prototype flow", "Copy / content", "Other"],
                key="artifact",
                help="Helps choose the most useful critique lenses.",
            )
            audience = st.text_input(
                "Audience / users",
                key="audience",
                help="Who is this for? If unknown, describe who you imagine using it.",
            )

        with c2:
            goal = st.text_input(
                "Primary goal of the work",
                key="goal_text",
                help="What are you trying to accomplish for the user or stakeholder?",
            )
            constraints = st.text_area(
                "Key constraints",
                key="constraints",
                height=92,
                help="Constraints can be time, platform, accessibility requirements, brand, ethics, scope, etc.",
            )

        focus = st.text_area(
            "What do you want critique on (today)?",
            key="focus",
            height=92,
            help="Example: hierarchy, onboarding clarity, accessibility, interaction feedback, edge cases, tone, trust, etc.",
        )
        work = st.text_area(
            "Describe what exists so far",
            key="work",
            height=160,
            help="Paste a description of your concept/flow, or describe the screen(s). If you have a link, describe what’s at the link.",
        )

        submitted = st.form_submit_button("Generate critique", icon=":material/auto_fix_high:", help="Creates a structured critique response.")

    if submitted:
        form_data = {
            "stage": stage,
            "artifact": artifact,
            "audience": audience,
            "goal": goal,
            "constraints": constraints,
            "focus": focus,
            "work": work,
        }
        st.session_state.last_intake = form_data

        with st.spinner("Generating critique…"):
            result = generate_from_form(form_data, identity_context)

        st.session_state.last_result = result
        st.session_state.notes.append(
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "stage": stage,
                "focus": focus[:80] + ("…" if len(focus) > 80 else ""),
                "result": result,
            }
        )

        st.markdown("### Output")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(result)
        st.markdown("</div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start a follow‑up chat", icon=":material/chat:", use_container_width=True,
                         help="Switch to the chat tab to continue the critique dialogue."):
                st.session_state.active_tab = "chat"
                st.rerun()
        with col_b:
            if st.button("Reset form", icon=":material/refresh:", use_container_width=True,
                         help="Clears the current inputs."):
                for key in ["stage", "artifact", "audience", "goal_text", "constraints", "focus", "work"]:
                    if key in st.session_state:
                        st.session_state[key] = ""
                st.session_state.last_intake = None
                st.session_state.last_result = None
                st.rerun()

# -----------------------------
# Follow-up Chat tab
# -----------------------------
with tab_chat:
    st.markdown("### Follow‑up chat")

    if not st.session_state.messages:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "If you want, tell me what you’re uncertain about (intent, audience, tradeoffs, constraints), and I’ll help you think through it.",
            }
        )

    # Render history
    for m in st.session_state.messages:
        avatar = ":material/person:" if m["role"] == "user" else ":material/smart_toy:"
        with st.chat_message(m["role"], avatar=avatar):
            st.markdown(m["content"])

    prompt = st.chat_input("Ask a follow‑up question…", help="Use this to refine the critique, explore tradeoffs, or clarify intent.")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar=":material/smart_toy:"):
            stream = generate_chat_response(
                user_input=prompt,
                identity_context=identity_context,
                chat_history=st.session_state.messages,
                last_intake=st.session_state.last_intake,
                last_result=st.session_state.last_result,
            )
            response_text = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response_text})

    with st.expander(":material/tune: Chat utilities", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear chat", icon=":material/delete:", use_container_width=True,
                         help="Removes the chat history for this session."):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Bring last critique into chat", icon=":material/summarize:", use_container_width=True,
                         help="Adds the last structured critique output into the chat context as a message."):
                if st.session_state.last_result:
                    st.session_state.messages.append({"role": "assistant", "content": st.session_state.last_result})
                    st.rerun()
                else:
                    st.info("No structured critique found yet. Run a Crit Session first.")

# -----------------------------
# Notes tab
# -----------------------------
with tab_notes:
    st.markdown("### Session notes (saved critiques)")

    if not st.session_state.notes:
        st.info("No saved critiques yet. Generate one in the Crit Session tab.")
    else:
        for i, note in enumerate(reversed(st.session_state.notes), start=1):
            with st.expander(f":material/description: Critique #{i} • {note['timestamp']} • {note['stage']}", expanded=False):
                st.caption(f"Focus: {note['focus']}")
                st.markdown(note["result"])

        # Export
        export_text = "\n\n---\n\n".join(
            [f"[{n['timestamp']}] {n['stage']} • {n['focus']}\n\n{n['result']}" for n in st.session_state.notes]
        )
        st.download_button(
            "Download session notes (.md)",
            data=export_text,
            file_name="critique_session_notes.md",
            mime="text/markdown",
            icon=":material/download:",
            help="Exports all saved critiques for this session as a Markdown file.",
            use_container_width=True,
        )

# Footer attribution
st.markdown(
    """
<div class="muted" style="margin-top:1.25rem; text-align:center;">
  Developed using an <code>identity.txt</code>-driven approach • Powered by <code>gemma-3-27b-it</code>
</div>
""",
    unsafe_allow_html=True,
)

