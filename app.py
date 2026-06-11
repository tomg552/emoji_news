"""
app.py — Streamlit UI for the Emoji News Script translator.

Run:
    pip install streamlit requests
    streamlit run app.py

Expects emoji_news_ontology.json, emoji_script.py and translator.py alongside
this file. Endpoint settings live in the sidebar and can be pre-seeded with
environment variables: EMOJI_LLM_BASE_URL, EMOJI_LLM_MODEL, EMOJI_LLM_API_KEY.
"""

import json
import os

import streamlit as st

from emoji_script import load_ontology
from translator import OpenAICompatibleClient, Translator, build_story_schema

st.set_page_config(page_title="Emoji News Translator", page_icon="📰", layout="wide")


@st.cache_resource
def get_ontology():
    return load_ontology()


ontology = get_ontology()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Model endpoint")
    base_url = st.text_input(
        "Base URL (OpenAI-compatible, ends in /v1)",
        value=os.environ.get("EMOJI_LLM_BASE_URL", ""),
        placeholder="https://your-endpoint.run.app/v1",
    )
    model = st.text_input(
        "Model name",
        value=os.environ.get("EMOJI_LLM_MODEL", "tgi"),
        help="vLLM: the served model id. TGI: usually 'tgi'.",
    )
    api_key = st.text_input(
        "API key (optional)",
        value=os.environ.get("EMOJI_LLM_API_KEY", ""),
        type="password",
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_retries = st.number_input("Validation retries", 0, 5, 2)
    schema_mode = st.selectbox(
        "Constrained decoding",
        ["off", "vllm", "tgi"],
        help="vllm: OpenAI-style response_format json_schema. "
             "tgi: TGI grammar via response_format json_object. "
             "off: rely on the validate-and-retry loop only.",
    )

    st.divider()
    st.caption(
        f"Ontology v{ontology['meta'].get('version', '?')} — "
        f"{len(ontology['concepts'])} concepts, "
        f"{len(ontology.get('actions', {}))} actions, "
        f"{len(ontology['compounds'])} compounds, "
        f"{len(ontology['grammar'])} grammar markers"
    )

# ---------------------------------------------------------------- main
st.title("📰 ➡️ 🧩 Emoji News Translator")

article = st.text_area(
    "Paste a news article (or a few paragraphs)",
    height=260,
    placeholder="The health secretary today announced...",
)

col_run, col_clear = st.columns([1, 5])
run = col_run.button("Translate", type="primary", disabled=not article.strip())

if run:
    if not base_url:
        st.error("Set the model endpoint base URL in the sidebar first.")
        st.stop()

    client = OpenAICompatibleClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        schema_style=None if schema_mode == "off" else schema_mode,
        json_schema=None if schema_mode == "off" else build_story_schema(ontology),
    )

    status = st.status("Translating…", expanded=True)
    translator = Translator(
        client=client,
        ontology=ontology,
        max_retries=int(max_retries),
        on_event=lambda msg: status.write(msg),
    )

    try:
        result = translator.translate(article)
    except Exception as exc:  # endpoint/network failures surfaced plainly
        status.update(label="Endpoint error", state="error")
        st.exception(exc)
        st.stop()

    if result.ok:
        status.update(label=f"Done in {result.attempts} attempt(s)", state="complete")
        st.subheader("Emoji rendering")
        for line in result.emoji.splitlines():
            st.markdown(f"<div style='font-size:2rem; line-height:2.6rem'>{line}</div>",
                        unsafe_allow_html=True)
        st.code(result.emoji, language=None)
    else:
        status.update(label="Failed validation after all retries", state="error")
        st.error("The model never produced a valid structured story. "
                 "Last attempt and errors are below for inspection.")

    left, right = st.columns(2)
    with left:
        with st.expander("Structured story (intermediate representation)",
                         expanded=not result.ok):
            st.json(result.story or {})
    with right:
        with st.expander("Deterministic alias-scan grounding"):
            st.caption("What the longest-match index found directly in the "
                       "source text — compare against the refs the model chose.")
            st.table(result.grounding or [{"surface": "(no matches)"}])

    with st.expander("Run log"):
        st.text("\n".join(result.log))

    if result.ok:
        st.download_button(
            "Download structured story JSON",
            data=json.dumps(result.story, ensure_ascii=False, indent=2),
            file_name="story.json",
            mime="application/json",
        )
