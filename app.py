"""
Streamlit application for the Retail Q&A Tool.
"""

import time
import streamlit as st

from langchain_helper import db, get_few_shot_db_chain


# Gemini 3.1 Flash Lite: 15 RPM
# Each question makes 2 API calls (SQL generation + answer generation)
# Limit users to 7 questions per minute.
MAX_REQUESTS = 7
WINDOW = 60  # seconds


st.set_page_config(
    page_title="Retail Q&A Tool",
    page_icon="👕",
    layout="centered",
)

st.title("👕 Retail Q&A Tool")
st.caption("Ask questions about the retail inventory database using natural language.")


def generate_answer(question: str) -> str:
    """Generate a natural language answer for the given question."""

    sql_chain, answer_chain = get_few_shot_db_chain()

    sql_query = sql_chain.invoke(
        {
            "input": question,
            "table_info": db.get_table_info(),
        }
    )

    sql_result = db.run(sql_query)

    answer = answer_chain.invoke(
        {
            "input": question,
            "sql_result": sql_result,
        }
    )

    return answer


question = st.text_input(
    "Enter your question",
    placeholder="e.g. Which product earns the most after discounts?",
)

# Initialize request history
if "request_times" not in st.session_state:
    st.session_state.request_times = []

if question:

    now = time.time()

    # Keep only requests from the last minute
    st.session_state.request_times = [
        t for t in st.session_state.request_times
        if now - t < WINDOW
    ]

    if len(st.session_state.request_times) >= MAX_REQUESTS:
        st.warning("Rate limit exceeded. Please wait a minute before asking more questions.")
        st.stop()

    st.session_state.request_times.append(now)

    with st.spinner("Generating answer..."):

        try:
            response = generate_answer(question)

            st.subheader("Answer")
            st.write(response)

        except Exception as e:
            st.error("Something went wrong while processing your request.")
            st.exception(e)