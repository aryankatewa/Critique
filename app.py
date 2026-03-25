import os
from pathlib import Path

import streamlit as st
from google import genai
from google.genai import types

APP_TITLE = "Critique"
DEFAULT_MODEL = "gemini-3-flash-preview"


def load_identity(path: str = "identity.txt") -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def build_user_prompt(
    mode: str,
    audience: str,
    depth: str,
    url: str,
    figma_url: str,
    notes: str,
) -> str:
    return f"""
Critique mode: {mode}
Audience style: {audience}
Depth: {depth}

Optional webpage URL:
{url or "None"}

Optional Figma prototype URL:
{figma_url or "None"}

Additional context from user:
{notes or "None"}

Please analyze the provided inputs and return the critique in the required structured format.
""".strip()


def get_client() -> genai.Client:
    api_key = st.secrets["GEMINI_API_KEY"]
    return genai.Client(api_key=api_key)


def build_contents(prompt_text: str, uploaded_files) -> list:
    contents = [prompt_text]

    for file in uploaded_files:
        image_bytes = file.read()
        contents.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type=file.type,
            )
        )

    return contents


def generate_critique(identity_text: str, prompt_text: str, uploaded_files):
    client = get_client()

    contents = build_contents(prompt_text, uploaded_files)

    response = client.models.generate_content(
        model=DEFAULT_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=identity_text,
            temperature=0.6,
            max_output_tokens=2500,
        ),
    )
    return response.text


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Multimodal UX critique for screenshots, flows, URLs, and public Figma links.")

    identity_text = load_identity()

    with st.sidebar:
        st.header("Critique Settings")
        mode = st.selectbox(
            "Mode",
            [
                "Single screen critique",
                "Flow critique",
                "Landing page critique",
                "Form usability critique",
                "Accessibility quick scan",
            ],
        )
        audience = st.selectbox(
            "Audience style",
            [
                "UX designer language",
                "Plain language",
                "Stakeholder summary",
            ],
        )
        depth = st.selectbox(
            "Depth",
            ["Brief", "Standard", "Deep"],
        )

    col1, col2 = st.columns(2)

    with col1:
        uploaded_files = st.file_uploader(
            "Upload screenshots",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )

        url = st.text_input("Public webpage URL")
        figma_url = st.text_input("Public Figma prototype link")
        notes = st.text_area("Additional context", height=180)

        run = st.button("Generate critique", use_container_width=True)

    with col2:
        st.markdown("### What this tool critiques")
        st.markdown("""
- visual hierarchy
- usability and interaction clarity
- task flow continuity
- messaging and content clarity
- accessibility and inclusion
- UX tensions and research questions
        """)

    if run:
        prompt_text = build_user_prompt(
            mode=mode,
            audience=audience,
            depth=depth,
            url=url,
            figma_url=figma_url,
            notes=notes,
        )

        with st.spinner("Generating critique..."):
            try:
                result = generate_critique(identity_text, prompt_text, uploaded_files or [])
                st.markdown("## Critique")
                st.markdown(result)
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
