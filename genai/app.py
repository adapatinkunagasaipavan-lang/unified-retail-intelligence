"""
Streamlit chat UI for the Retail Intelligence Assistant.

Run with:
    streamlit run genai/app.py

No API key required -- this wraps the local, offline text-to-SQL engine
and churn explainer built in genai/text_to_sql and genai/agents.
"""
import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agents"))
from router import Agent  # noqa: E402

st.set_page_config(page_title="Retail Intelligence Assistant", page_icon="📊")

st.title("📊 Retail Intelligence Assistant")
st.caption(
    "Ask about sales, categories, or customer churn risk. "
    "Every answer is grounded in a real query against the Gold tables -- "
    "no hallucination, no API key required."
)

GOLD_DIR = st.sidebar.text_input("Gold tables directory", value="data/lake/gold")

with st.sidebar:
    st.markdown("### Example questions")
    st.markdown(
        "- What were the top 5 selling categories?\n"
        "- What is the total sales for Electronics?\n"
        "- What is the churn rate?\n"
        "- What is the total revenue?\n"
        "- Why is customer CUST000279 high risk?"
    )


@st.cache_resource(show_spinner="Loading Gold tables and Production model...")
def get_agent(gold_dir: str):
    return Agent(gold_dir=gold_dir)


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sql"):
            with st.expander("Show generated SQL"):
                st.code(msg["sql"], language="sql")
        if msg.get("evidence") is not None:
            with st.expander("Show evidence"):
                st.dataframe(msg["evidence"])

question = st.chat_input("Ask a question about sales or customer churn...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    try:
        agent = get_agent(GOLD_DIR)
        result = agent.ask(question)
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(
                f"Couldn't answer that -- {e}\n\n"
                f"Make sure you've run the pipeline first "
                f"(`python pipelines/run_pipeline.py`) so the Gold tables and "
                f"the Production model exist."
            )
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {e}"
        })
    else:
        with st.chat_message("assistant"):
            st.markdown(result.answer)
            sql = getattr(result, "sql", None)
            if sql:
                with st.expander("Show generated SQL"):
                    st.code(sql, language="sql")
            evidence = getattr(result, "data", None)
            if evidence is None:
                evidence = getattr(result, "evidence", None)
            if evidence is not None:
                with st.expander("Show evidence"):
                    st.dataframe(evidence)

        st.session_state.messages.append({
            "role": "assistant",
            "content": result.answer,
            "sql": sql,
            "evidence": evidence,
        })
